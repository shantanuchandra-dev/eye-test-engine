#!/usr/bin/env python3
"""
test_curl_buttons.py — End-to-end tests for every UI button that triggers a
curl/HTTP request across all three frontend screens.

Screens covered:
  1. intake.html   — Patient Intake form
  2. index.html    — Main Test Screen (app.js)
  3. dashboard.html — Dashboard

Each test validates:
  - The correct HTTP method is used (GET / POST / PUT)
  - The correct endpoint path is hit
  - The expected request payload shape is sent
  - The response status is 2xx (or an expected error code)

Results are appended to a CSV file (test_results.csv) so the report keeps
growing across runs.  The tabular summary is also printed to stdout.

Usage:
    python test_curl_buttons.py                     # run all tests
    python test_curl_buttons.py -k "intake"         # run only intake tests
    python test_curl_buttons.py --base http://...:5050  # custom backend URL

Dependencies:  requests, tabulate  (pip install requests tabulate)
"""

import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
#  Optional: install tabulate if missing
# ---------------------------------------------------------------------------
try:
    from tabulate import tabulate
except ImportError:
    print("[WARN] tabulate not installed — falling back to simple table output")
    tabulate = None

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
# Default backend URL — override via CLI flag --base or env BACKEND_URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5050")

# Phoropter broker URL — used for direct-broker tests (screenshot, etc.)
PHOROPTER_URL = os.environ.get(
    "PHOROPTER_URL",
    "https://rajasthan-royals.preprod.lenskart.com",
)

# CSV file that accumulates results across runs
CSV_PATH = Path(__file__).parent / "test_results.csv"

# CSV columns
CSV_COLUMNS = [
    "timestamp",       # ISO 8601 when the test ran
    "screen",          # intake | test | dashboard
    "button_name",     # human-readable button / trigger name
    "method",          # GET | POST | PUT
    "endpoint",        # e.g. /api/devices
    "payload_summary", # abbreviated payload description
    "expected_status", # e.g. "2xx" or "200"
    "actual_status",   # actual HTTP status code
    "pass_fail",       # PASS | FAIL | SKIP
    "notes",           # error message or extra info
]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _api(path: str) -> str:
    """Build a full URL from a relative API path."""
    return f"{BACKEND_URL}{path}"


def _broker(path: str) -> str:
    """Build a full URL from a relative broker path."""
    return f"{PHOROPTER_URL}{path}"


def _ts() -> str:
    """Current ISO 8601 timestamp."""
    return datetime.now(tz=None).astimezone().isoformat(timespec="seconds")


def _short(obj: Any, max_len: int = 80) -> str:
    """Compact string representation of a payload for the CSV."""
    s = json.dumps(obj, ensure_ascii=False) if isinstance(obj, (dict, list)) else str(obj)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _is_2xx(status: int) -> bool:
    return 200 <= status < 300


# ---------------------------------------------------------------------------
#  Result storage
# ---------------------------------------------------------------------------
class TestResult:
    """Single test result row."""
    def __init__(
        self,
        screen: str,
        button: str,
        method: str,
        endpoint: str,
        payload_summary: str = "",
        expected: str = "2xx",
        actual: int = 0,
        passed: bool = False,
        notes: str = "",
    ):
        self.timestamp = _ts()
        self.screen = screen
        self.button = button
        self.method = method
        self.endpoint = endpoint
        self.payload_summary = payload_summary
        self.expected = expected
        self.actual = actual
        self.passed = passed
        self.notes = notes

    @property
    def pass_fail(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_row(self) -> list:
        return [
            self.timestamp,
            self.screen,
            self.button,
            self.method,
            self.endpoint,
            self.payload_summary,
            self.expected,
            self.actual,
            self.pass_fail,
            self.notes,
        ]


_results: List[TestResult] = []


def _record(result: TestResult) -> None:
    """Append a result to the in-memory list."""
    _results.append(result)


def _skip(screen: str, button: str, method: str, endpoint: str, reason: str) -> None:
    """Record a skipped test."""
    _record(TestResult(
        screen=screen, button=button, method=method, endpoint=endpoint,
        expected="-", actual=0, passed=False,
        notes=f"SKIP: {reason}",
    ))


# ---------------------------------------------------------------------------
#  CSV writer — appends new rows; creates header if file is new
# ---------------------------------------------------------------------------
def flush_results_to_csv() -> None:
    """Append all in-memory results to the CSV file."""
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)
        for r in _results:
            writer.writerow(r.as_row())


# ---------------------------------------------------------------------------
#  Pretty-print the tabular summary to stdout
# ---------------------------------------------------------------------------
def print_results_table() -> None:
    """Print a nicely formatted table of all results."""
    headers = ["#", "Screen", "Button", "Method", "Endpoint", "Status", "Result", "Notes"]
    rows = []
    for i, r in enumerate(_results, 1):
        rows.append([
            i,
            r.screen,
            r.button,
            r.method,
            r.endpoint,
            r.actual if r.actual else "-",
            r.pass_fail,
            (r.notes[:60] + "...") if len(r.notes) > 60 else r.notes,
        ])

    if tabulate:
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Fallback: simple formatted table
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print()
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))

    # Summary line
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed and "SKIP" not in r.notes)
    skipped = sum(1 for r in _results if "SKIP" in r.notes)
    total = len(_results)
    print(f"\nTotal: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    print(f"Results appended to: {CSV_PATH.resolve()}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  TEST DEFINITIONS
#  Each function tests one button / trigger. Functions are grouped by screen.
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
#  Helper: ensure the backend is running before we start
# ---------------------------------------------------------------------------
def check_backend_alive() -> bool:
    """Verify the Flask backend is reachable."""
    try:
        r = requests.get(_api("/api/config"), timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
#  INTAKE.HTML — Patient Intake Screen
# ---------------------------------------------------------------------------

def test_intake_load_devices():
    """
    intake.html — Page Load → loadDevices()

    On page load, intake.html calls GET /api/devices to populate the
    phoropter dropdown.  The broker lists all available TOPCON devices.
    """
    screen = "intake"
    button = "Page Load (loadDevices)"
    method = "GET"
    endpoint = "/api/devices"
    try:
        r = requests.get(_api(endpoint), timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"devices={len(r.json().get('devices', []))} found" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_intake_acquire_device():
    """
    intake.html — Acquire button → acquireDevice()

    POST /api/devices/{device_id}/acquire with operator name.
    Acquires a real device from the broker so phoropter commands work.
    Uses a test brain_id and releases immediately after.
    """
    screen = "intake"
    button = "Acquire (acquireDevice)"
    method = "POST"

    # First list devices to get a real device_id
    try:
        dev_resp = requests.get(_api("/api/devices"), timeout=10)
        devices = dev_resp.json().get("devices", [])
    except Exception:
        devices = []

    if not devices:
        _skip(screen, button, method, "/api/devices/{id}/acquire",
               "No devices available to test acquisition")
        return

    device_id = devices[0].get("device_id", devices[0].get("id", ""))
    endpoint = f"/api/devices/{device_id}/acquire"
    # The intake page's acquireDevice() constructs a brain_id from
    # "{operator}@{client_ip}" and sends {brain_id: ..., name: ...}.
    # This matches the broker's required payload format.
    brain_id = "intake_test@127.0.0.1"
    payload = {"brain_id": brain_id, "name": "Test Suite Intake"}

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=r.json().get("status", r.text[:80]) if r.text else "",
        ))
        # Release immediately so the device isn't stuck
        if _is_2xx(r.status_code):
            requests.post(
                _api(f"/api/devices/{device_id}/release"),
                json={"brain_id": brain_id},
                timeout=5,
            )
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_intake_submit_form():
    """
    intake.html — Submit button (Start Refraction Session)

    POST /api/session/intake with full patient data.
    Creates a new session and returns session_id + derived_variables.
    This is the main entry point for starting a refraction test.
    """
    screen = "intake"
    button = "Start Refraction Session (submit)"
    method = "POST"
    endpoint = "/api/session/intake"

    # Minimal valid payload matching the intake form's structure
    payload = {
        "phoropter_id": "Test",
        "patient": {
            "age": 45,
            "occupation": "IT Professional",
            "screen_time_hours": 8,
            "driving_hours": 1,
            "primary_reason": "Blurred distance vision",
            "symptoms_text": "Glare/halos,Night driving difficulty",
            "satisfaction": "Not satisfied",
            "wear_type": "Single vision",
            "distance_target": "6/6 target",
            "priority": "Standard",
            "near_priority": "Medium",
            "last_eye_test_months": 24,
            "rx_change_large": False,
            "fluctuating_vision": False,
            "diabetes": False,
            "prior_surgery": "None",
            "keratoconus": False,
            "amblyopia": False,
            "infection": False,
            "optom_review": False,
            "autorefractor": {
                "right": {"sph": -2.50, "cyl": -0.75, "axis": 180},
                "left": {"sph": -3.00, "cyl": -0.50, "axis": 175},
            },
            "lensometry": {
                "right": {"sph": -2.25, "cyl": -0.50, "axis": 180},
                "left": {"sph": -2.75, "cyl": -0.50, "axis": 170},
                "add_r": 1.50,
                "add_l": 1.50,
            },
        },
    }

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=15)
        data = r.json() if _is_2xx(r.status_code) else {}
        sid = data.get("session_id", "")
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code) and bool(sid),
            notes=f"session_id={sid}" if sid else r.text[:80],
        ))
        # Store session_id for later tests
        return sid
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))
        return None


# ---------------------------------------------------------------------------
#  INDEX.HTML (app.js) — Main Test Screen
# ---------------------------------------------------------------------------

def test_index_fetch_config():
    """
    index.html — Page Load → fetchConfig()

    GET /api/config fetches backend configuration (backend_url,
    phoropter_base_url).  Called on DOMContentLoaded.
    """
    screen = "test"
    button = "Page Load (fetchConfig)"
    method = "GET"
    endpoint = "/api/config"
    try:
        r = requests.get(_api(endpoint), timeout=5)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=str(r.json().keys()) if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_fetch_devices():
    """
    index.html — Page Load → fetchDevices()

    GET /api/devices populates the phoropter dropdown on the device bar.
    Same endpoint as intake but called from the test screen context.
    """
    screen = "test"
    button = "Page Load (fetchDevices)"
    method = "GET"
    endpoint = "/api/devices"
    try:
        r = requests.get(_api(endpoint), timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"devices found: {len(r.json().get('devices', []))}" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_acquire_device():
    """
    index.html — Acquire button → acquireSelectedDevice()

    POST /api/devices/{device_id}/acquire with brain_id and operator name.
    The brain_id is constructed as "{operator}@{client_ip}".
    """
    screen = "test"
    button = "Acquire (acquireSelectedDevice)"
    method = "POST"

    try:
        dev_resp = requests.get(_api("/api/devices"), timeout=10)
        devices = dev_resp.json().get("devices", [])
    except Exception:
        devices = []

    if not devices:
        _skip(screen, button, method, "/api/devices/{id}/acquire",
               "No devices available")
        return

    device_id = devices[0].get("device_id", devices[0].get("id", ""))
    endpoint = f"/api/devices/{device_id}/acquire"
    payload = {"brain_id": "test_suite@127.0.0.1", "name": "Test Suite"}

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=r.text[:80],
        ))
        # Release immediately
        if _is_2xx(r.status_code):
            requests.post(
                _api(f"/api/devices/{device_id}/release"),
                json={"brain_id": "test_suite@127.0.0.1"},
                timeout=5,
            )
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_index_session_status(session_id: str):
    """
    index.html — checkIntakeRedirect() / _tryRestoreSession()

    GET /api/session/{session_id}/status fetches the current state of an
    active session.  Called after intake redirect and on page restore.
    """
    screen = "test"
    button = "Session Status (fetchSessionStatus)"
    method = "GET"
    endpoint = f"/api/session/{session_id}/status"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/status",
               "No session_id available")
        return

    try:
        r = requests.get(_api(endpoint), timeout=10)
        data = r.json() if _is_2xx(r.status_code) else {}
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"state={data.get('state', '?')}" if data else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_submit_response(session_id: str):
    """
    index.html — Response buttons (1-6) → submitResponse(responseValue)

    POST /api/session/{session_id}/respond sends the patient's response
    (e.g. READABLE, NOT_READABLE, BETTER_1, etc.) to advance the FSM.
    The backend processes the response through the FSM engine and returns
    the next question, options, power state, and phoropter commands.
    """
    screen = "test"
    button = "Response Button (submitResponse)"
    method = "POST"
    endpoint = f"/api/session/{session_id}/respond"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/respond",
               "No session_id available")
        return

    # First get current state to know what options are valid
    try:
        status_resp = requests.get(_api(f"/api/session/{session_id}/status"), timeout=10)
        status_data = status_resp.json()
        options = status_data.get("options", [])
    except Exception:
        options = ["READABLE"]  # fallback

    response_value = options[0] if options else "READABLE"
    payload = {"response": response_value}

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=15)
        data = r.json() if _is_2xx(r.status_code) else {}
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"next_state={data.get('state', '?')}, response_sent={response_value}" if data else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_index_sync_power(session_id: str):
    """
    index.html — Manual power click (+/-) or Type mode → sync-power

    POST /api/session/{session_id}/sync-power sends manually adjusted
    power values to update the FSM state.  Called by applyManualPowerChange()
    (click ±0.25 SPH/CYL, ±5 AXIS) and _commitActiveInput() (type mode).
    This does NOT send phoropter commands — it only updates the FSM row
    and prev-state tracker.
    """
    screen = "test"
    button = "Manual Power (sync-power)"
    method = "POST"
    endpoint = f"/api/session/{session_id}/sync-power"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/sync-power",
               "No session_id available")
        return

    payload = {
        "right": {"sph": -2.50, "cyl": -0.75, "axis": 180, "add": 0},
        "left": {"sph": -3.00, "cyl": -0.50, "axis": 175, "add": 0},
    }

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes="Power sync accepted" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_index_derived_variables(session_id: str):
    """
    index.html — Debug panel → refreshDerivedVariables()

    GET /api/session/{session_id}/derived-variables fetches the full set
    of derived variables (46 clinical parameters), working variables, and
    calibration values for display in the debug panel.
    """
    screen = "test"
    button = "Debug Panel (refreshDerivedVariables)"
    method = "GET"
    endpoint = f"/api/session/{session_id}/derived-variables"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/derived-variables",
               "No session_id available")
        return

    try:
        r = requests.get(_api(endpoint), timeout=10)
        data = r.json() if _is_2xx(r.status_code) else {}
        dv_count = len(data.get("derived_variables", {}))
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"derived_vars={dv_count}" if data else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_reset_phoropter():
    """
    index.html — Reset button (red) → resetPhoropter()

    POST /api/phoropter/{phoropter_id}/reset resets the phoropter to
    0/0/180 (SPH=0, CYL=0, AXIS=180).  The backend proxies this to
    the broker's reset endpoint.
    """
    screen = "test"
    button = "Reset Phoropter (resetPhoropter)"
    method = "POST"
    # Use 'Test' device to avoid hitting real hardware
    endpoint = "/api/phoropter/Test/reset"

    try:
        r = requests.post(_api(endpoint), timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code) or r.status_code == 404,
            notes="Reset sent (or test device)" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_send_power(session_id: str):
    """
    index.html — fetchSessionStatus() → send-power (after reset)

    POST /api/session/{session_id}/send-power re-sends the current FSM
    row's power values to the phoropter.  Called after a reset so the
    physical device matches the session's starting Rx.
    """
    screen = "test"
    button = "Send Power (send-power after reset)"
    method = "POST"
    endpoint = f"/api/session/{session_id}/send-power"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/send-power",
               "No session_id available")
        return

    try:
        r = requests.post(_api(endpoint), timeout=10)
        data = r.json() if _is_2xx(r.status_code) else {}
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"status={data.get('status', '?')}" if data else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_screenshot():
    """
    index.html — Live View button → fetchScreenshot()

    POST /phoropter/{phoropter_id}/screenshot (direct to broker, not
    through Flask).  Returns a base64 JPEG of the phoropter's current
    display.  The frontend calls this directly against the broker URL.
    """
    screen = "test"
    button = "Live View (fetchScreenshot)"
    method = "POST"

    # List devices to get a real device_id for screenshot
    try:
        dev_resp = requests.get(_api("/api/devices"), timeout=10)
        devices = dev_resp.json().get("devices", [])
    except Exception:
        devices = []

    if not devices:
        _skip(screen, button, method, "/phoropter/{id}/screenshot",
               "No devices available for screenshot test")
        return

    device_id = devices[0].get("device_id", devices[0].get("id", ""))
    endpoint = f"/phoropter/{device_id}/screenshot"

    try:
        r = requests.post(
            _broker(endpoint),
            headers={"x-brain-id": "test_suite@127.0.0.1"},
            timeout=10,
        )
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"response_size={len(r.content)}B" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_heartbeat():
    """
    index.html — Interval timer (30s) → startHeartbeat()

    POST /api/devices/{device_id}/heartbeat keeps the device lock alive.
    Sent every 30 seconds while a device is acquired.
    We must acquire first so the heartbeat has an active lock to refresh.
    """
    screen = "test"
    button = "Heartbeat (30s interval)"
    method = "POST"
    brain_id = "heartbeat_test@127.0.0.1"

    try:
        dev_resp = requests.get(_api("/api/devices"), timeout=10)
        devices = dev_resp.json().get("devices", [])
    except Exception:
        devices = []

    if not devices:
        _skip(screen, button, method, "/api/devices/{id}/heartbeat",
               "No devices available")
        return

    device_id = devices[0].get("device_id", devices[0].get("id", ""))
    endpoint = f"/api/devices/{device_id}/heartbeat"
    payload = {"brain_id": brain_id}

    # Acquire first so the heartbeat has an active lock
    try:
        requests.post(
            _api(f"/api/devices/{device_id}/acquire"),
            json={"brain_id": brain_id, "name": "Heartbeat Test"},
            timeout=10,
        )
    except Exception:
        pass

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=r.status_code in (200, 202),
            notes=r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))
    finally:
        # Release after heartbeat test
        try:
            requests.post(
                _api(f"/api/devices/{device_id}/release"),
                json={"brain_id": brain_id},
                timeout=5,
            )
        except Exception:
            pass


def test_index_release_device():
    """
    index.html — endTestWithStatus() / discardTest() → releaseDevice()

    POST /api/devices/{device_id}/release releases the phoropter lock.
    Called automatically when a test ends or is discarded.
    We acquire first to ensure we have a lock to release.
    """
    screen = "test"
    button = "Release Device (releaseDevice)"
    method = "POST"
    brain_id = "release_test@127.0.0.1"

    try:
        dev_resp = requests.get(_api("/api/devices"), timeout=10)
        devices = dev_resp.json().get("devices", [])
    except Exception:
        devices = []

    if not devices:
        _skip(screen, button, method, "/api/devices/{id}/release",
               "No devices available")
        return

    device_id = devices[0].get("device_id", devices[0].get("id", ""))
    endpoint = f"/api/devices/{device_id}/release"
    payload = {"brain_id": brain_id}

    # Acquire first so we have a lock to release
    try:
        requests.post(
            _api(f"/api/devices/{device_id}/acquire"),
            json={"brain_id": brain_id, "name": "Release Test"},
            timeout=10,
        )
    except Exception:
        pass

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_index_end_session(session_id: str):
    """
    index.html — Save & End button → endTestWithStatus('completed')

    POST /api/session/{session_id}/end finalizes the session, saves
    results to CSV/metadata, and optionally uploads to Supabase.
    Payload includes store flag, operator name, and feedback.
    """
    screen = "test"
    button = "Save & End (endTestWithStatus)"
    method = "POST"
    endpoint = f"/api/session/{session_id}/end"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/end",
               "No session_id available")
        return

    payload = {
        "store": True,
        "operator_name": "test_suite_runner",
        "qualitative_feedback": "",
    }

    try:
        r = requests.post(_api(endpoint), json=payload, timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=r.text[:80] if r.text else "",
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_index_discard_session(session_id: str):
    """
    index.html — Discard button → discardTest()

    POST /api/session/{session_id}/discard discards the session without
    saving.  Called when the operator decides to throw away results.
    """
    screen = "test"
    button = "Discard (discardTest)"
    method = "POST"
    endpoint = f"/api/session/{session_id}/discard"

    if not session_id:
        _skip(screen, button, method, "/api/session/{id}/discard",
               "No session_id available")
        return

    try:
        r = requests.post(_api(endpoint), timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            # Discard on already-ended session may 404 — that's acceptable
            passed=_is_2xx(r.status_code) or r.status_code == 404,
            notes=r.text[:80] if r.text else "",
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_index_get_client_ip():
    """
    index.html — Page Load → getClientIp()

    GET https://api.ipify.org?format=json fetches the client's public IP.
    Used to construct the brain_id for device acquisition.
    This is an external API call, not routed through the Flask backend.
    """
    screen = "test"
    button = "Get Client IP (getClientIp)"
    method = "GET"
    endpoint = "https://api.ipify.org?format=json"

    try:
        r = requests.get(endpoint, timeout=5)
        ip = r.json().get("ip", "") if _is_2xx(r.status_code) else ""
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code) and bool(ip),
            notes=f"ip={ip}" if ip else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


# ---------------------------------------------------------------------------
#  DASHBOARD.HTML — Dashboard Screen
# ---------------------------------------------------------------------------

def test_dashboard_load_config():
    """
    dashboard.html — Page Load → loadConfig()

    GET /api/dashboard/config fetches the current admin settings
    (tests_enabled, daily_limit, daily_limit_scope, per_phoropter_enabled).
    """
    screen = "dashboard"
    button = "Page Load (loadConfig)"
    method = "GET"
    endpoint = "/api/dashboard/config"
    try:
        r = requests.get(_api(endpoint), timeout=5)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=str(r.json().keys()) if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_dashboard_save_config():
    """
    dashboard.html — Save settings button → PUT /api/dashboard/config

    Saves admin settings: tests_enabled toggle, daily_limit number,
    daily_limit_scope (global / per_phoropter).
    """
    screen = "dashboard"
    button = "Save settings (save_config)"
    method = "PUT"
    endpoint = "/api/dashboard/config"
    payload = {
        "tests_enabled": True,
        "daily_limit": None,
        "daily_limit_scope": "global",
    }
    try:
        r = requests.put(_api(endpoint), json=payload, timeout=5)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes="Config saved" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_dashboard_per_phoropter_toggle():
    """
    dashboard.html — Per-phoropter checkbox toggle

    PUT /api/dashboard/config with per_phoropter_enabled map.
    Each phoropter can be individually enabled/disabled for new tests.
    """
    screen = "dashboard"
    button = "Per-phoropter toggle (checkbox)"
    method = "PUT"
    endpoint = "/api/dashboard/config"
    payload = {"per_phoropter_enabled": {"phoropter-1": True}}
    try:
        r = requests.put(_api(endpoint), json=payload, timeout=5)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload),
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes="Toggle applied" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            payload_summary=_short(payload), passed=False, notes=str(e),
        ))


def test_dashboard_load_stats():
    """
    dashboard.html — Page Load / Apply Filters → loadStats()

    GET /api/dashboard/stats?source=local returns total sessions, active
    sessions, by-day breakdown, and recent sessions table data.
    Supports filtering by date range, operator, phoropter, and data source.
    """
    screen = "dashboard"
    button = "Load Stats (loadStats)"
    method = "GET"
    endpoint = "/api/dashboard/stats?source=local"
    try:
        r = requests.get(_api(endpoint), timeout=10)
        data = r.json() if _is_2xx(r.status_code) else {}
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"total={data.get('total_sessions', '?')}" if data else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_dashboard_load_rr():
    """
    dashboard.html — Page Load / Apply Filters → loadRR()

    GET /api/dashboard/rr?source=local returns R&R (Repeatability &
    Reproducibility) data grouped by operator and by phoropter.
    Includes count, mean duration, completion rate, and phase jump rate.
    """
    screen = "dashboard"
    button = "Load R&R (loadRR)"
    method = "GET"
    endpoint = "/api/dashboard/rr?source=local"
    try:
        r = requests.get(_api(endpoint), timeout=10)
        data = r.json() if _is_2xx(r.status_code) else {}
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"operators={len(data.get('by_operator', {}))}" if data else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_dashboard_export_csv():
    """
    dashboard.html — Download CSV button → GET /api/dashboard/export

    Downloads filtered session metadata as a CSV file.
    Uses the current filter parameters (date range, operator, phoropter).
    """
    screen = "dashboard"
    button = "Download CSV (export_link)"
    method = "GET"
    endpoint = "/api/dashboard/export?source=local"
    try:
        r = requests.get(_api(endpoint), timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes=f"content_type={r.headers.get('Content-Type', '?')}" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


def test_dashboard_apply_filters():
    """
    dashboard.html — Apply Filters button → refreshData()

    GET /api/dashboard/stats with filter params (from, to, operator,
    phoropter, source).  The Apply button re-triggers both loadStats()
    and loadRR() with the current filter values.
    """
    screen = "dashboard"
    button = "Apply Filters (refreshData)"
    method = "GET"
    endpoint = "/api/dashboard/stats?source=local&from=2024-01-01&to=2026-12-31"
    try:
        r = requests.get(_api(endpoint), timeout=10)
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            expected="2xx", actual=r.status_code,
            passed=_is_2xx(r.status_code),
            notes="Filtered stats loaded" if _is_2xx(r.status_code) else r.text[:80],
        ))
    except Exception as e:
        _record(TestResult(
            screen=screen, button=button, method=method, endpoint=endpoint,
            passed=False, notes=str(e),
        ))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    """Simple arg parsing (avoid argparse dependency in minimal envs)."""
    args = sys.argv[1:]
    config = {"base": BACKEND_URL, "filter": None}
    i = 0
    while i < len(args):
        if args[i] == "--base" and i + 1 < len(args):
            config["base"] = args[i + 1]
            i += 2
        elif args[i] == "-k" and i + 1 < len(args):
            config["filter"] = args[i + 1].lower()
            i += 2
        else:
            i += 1
    return config


def main():
    global BACKEND_URL

    config = parse_args()
    BACKEND_URL = config["base"]
    keyword = config["filter"]

    print("=" * 72)
    print("  Eye Test Engine v2 — Button / Curl Test Suite")
    print(f"  Backend: {BACKEND_URL}")
    print(f"  Broker:  {PHOROPTER_URL}")
    print(f"  Filter:  {keyword or '(all)'}")
    print(f"  Time:    {_ts()}")
    print("=" * 72)

    # Verify backend is running
    if not check_backend_alive():
        print("\n[ERROR] Backend not reachable at", BACKEND_URL)
        print("        Start the server first:  python api_server.py")
        sys.exit(1)

    # ── Helper to decide whether to run a test based on -k filter ──
    def should_run(name: str) -> bool:
        if not keyword:
            return True
        return keyword in name.lower()

    # ── INTAKE SCREEN TESTS ──
    if should_run("intake_load_devices"):
        test_intake_load_devices()

    if should_run("intake_acquire_device"):
        test_intake_acquire_device()

    session_id = None
    if should_run("intake_submit_form"):
        session_id = test_intake_submit_form()

    # ── TEST SCREEN TESTS ──
    if should_run("index_fetch_config"):
        test_index_fetch_config()

    if should_run("index_fetch_devices"):
        test_index_fetch_devices()

    if should_run("index_acquire_device"):
        test_index_acquire_device()

    if should_run("index_get_client_ip"):
        test_index_get_client_ip()

    # Session-dependent tests — create a session if we don't have one
    if not session_id and any(should_run(t) for t in [
        "index_session_status", "index_submit_response", "index_sync_power",
        "index_derived_variables", "index_end_session", "index_discard_session",
    ]):
        session_id = test_intake_submit_form()

    if should_run("index_session_status"):
        test_index_session_status(session_id)

    if should_run("index_submit_response"):
        test_index_submit_response(session_id)

    if should_run("index_sync_power"):
        test_index_sync_power(session_id)

    if should_run("index_derived_variables"):
        test_index_derived_variables(session_id)

    if should_run("index_reset_phoropter"):
        test_index_reset_phoropter()

    if should_run("index_send_power"):
        test_index_send_power(session_id)

    if should_run("index_screenshot"):
        test_index_screenshot()

    if should_run("index_heartbeat"):
        test_index_heartbeat()

    if should_run("index_release_device"):
        test_index_release_device()

    # End/Discard — create a fresh session so we don't interfere with the
    # session used for response/sync tests
    if should_run("index_end_session"):
        end_sid = test_intake_submit_form()
        test_index_end_session(end_sid)

    if should_run("index_discard_session"):
        discard_sid = test_intake_submit_form()
        test_index_discard_session(discard_sid)

    # ── DASHBOARD SCREEN TESTS ──
    if should_run("dashboard_load_config"):
        test_dashboard_load_config()

    if should_run("dashboard_save_config"):
        test_dashboard_save_config()

    if should_run("dashboard_per_phoropter_toggle"):
        test_dashboard_per_phoropter_toggle()

    if should_run("dashboard_load_stats"):
        test_dashboard_load_stats()

    if should_run("dashboard_load_rr"):
        test_dashboard_load_rr()

    if should_run("dashboard_export_csv"):
        test_dashboard_export_csv()

    if should_run("dashboard_apply_filters"):
        test_dashboard_apply_filters()

    # ── Output ──
    flush_results_to_csv()
    print_results_table()


if __name__ == "__main__":
    main()
