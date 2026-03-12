#!/usr/bin/env python3
"""Flask API server for Eye Test Engine v2."""
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure package dir is on path
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path)
    load_dotenv()
except ImportError:
    pass

from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import urllib.error
import urllib.parse
import urllib.request

from session_orchestrator import SessionOrchestrator

# Load io modules via importlib to avoid stdlib clash
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "io_outputs", str(Path(__file__).resolve().parent / "io" / "outputs.py"))
_outputs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_outputs)
write_session_csv = _outputs.write_session_csv
write_session_metadata = _outputs.write_session_metadata
append_to_combined_log = _outputs.append_to_combined_log
append_to_combined_metadata = _outputs.append_to_combined_metadata
build_session_metadata = _outputs.build_session_metadata
session_csv_string = _outputs.session_csv_string

_remote_spec = _ilu.spec_from_file_location(
    "remote_storage", str(Path(__file__).resolve().parent / "io" / "remote_storage.py"))
_remote = _ilu.module_from_spec(_remote_spec)
_remote_spec.loader.exec_module(_remote)
upload_session_remote = _remote.upload_session

_dash_spec = _ilu.spec_from_file_location(
    "dashboard_data", str(Path(__file__).resolve().parent / "io" / "dashboard_data.py"))
_dashboard = _ilu.module_from_spec(_dash_spec)
_dash_spec.loader.exec_module(_dashboard)

app = Flask(__name__)
CORS(app)

sessions = {}  # session_id -> SessionOrchestrator

PHOROPTER_BASE_URL = os.environ.get("PHOROPTER_BASE_URL", "https://rajasthan-royals.preprod.lenskart.com")
CALIBRATION_PATH = str(Path(__file__).resolve().parent / "config" / "calibration.csv")

_IS_VERCEL = bool(os.environ.get("VERCEL"))
LOGS_DIR = Path("/tmp/eye_test_logs") if _IS_VERCEL else Path(__file__).parent / "logs"
SESSIONS_DIR = LOGS_DIR / "sessions"
COMBINED_LOG_PATH = LOGS_DIR / "combined_log.csv"
COMBINED_META_PATH = LOGS_DIR / "combined_metadata.csv"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Dashboard config (in-memory, matches v1)
_dashboard_config = {
    "tests_enabled": True,
    "daily_limit": None,
    "daily_limit_scope": "global",
    "per_phoropter_enabled": {},
}


def _log_api_command(action: str, payload: dict) -> None:
    print(f"[API] {action}: {json.dumps(payload, ensure_ascii=False)}")


def _request_payload() -> dict:
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload
    raw = request.get_data(as_text=True) or ""
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _proxy_request(method: str, path: str, payload: Optional[dict] = None, query: Optional[dict] = None):
    url = f"{PHOROPTER_BASE_URL}{path}"
    if query:
        cleaned = {k: v for k, v in query.items() if v is not None and v != ""}
        if cleaned:
            url = f"{url}?{urllib.parse.urlencode(cleaned)}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8")
            if body:
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = {"raw": body}
            else:
                parsed = {"status": "success"}
            return jsonify(parsed), resp.getcode()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"error": body or str(e)}
        if "error" not in parsed and "reason" not in parsed:
            parsed["error"] = str(e)
        return jsonify(parsed), e.code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# ── Config ──────────────────────────────────────────
@app.route('/api/config', methods=['GET'])
def get_config():
    backend_url = os.environ.get("BACKEND_URL", "")
    return jsonify({
        "backend_url": backend_url,
        "phoropter_base_url": PHOROPTER_BASE_URL
    })


# ── Device Management ───────────────────────────────
@app.route('/api/devices', methods=['GET'])
def list_devices():
    include_all = request.args.get("all")
    return _proxy_request("GET", "/devices", query={"all": include_all})

@app.route('/api/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    return _proxy_request("GET", f"/devices/{device_id}")

@app.route('/api/devices/<device_id>/acquire', methods=['POST'])
def acquire_device(device_id):
    payload = _request_payload()
    _log_api_command(f"/api/devices/{device_id}/acquire", payload)
    return _proxy_request("POST", f"/devices/{device_id}/acquire", payload=payload)

@app.route('/api/devices/<device_id>/release', methods=['POST'])
def release_device(device_id):
    payload = _request_payload()
    return _proxy_request("POST", f"/devices/{device_id}/release", payload=payload)

@app.route('/api/devices/<device_id>/heartbeat', methods=['POST'])
def heartbeat_device(device_id):
    payload = _request_payload()
    return _proxy_request("POST", f"/devices/{device_id}/heartbeat", payload=payload)

@app.route('/api/brains', methods=['GET'])
def list_brains():
    return _proxy_request("GET", "/brains")

@app.route('/api/events', methods=['GET'])
def list_events():
    limit = request.args.get("limit", "20")
    return _proxy_request("GET", "/events", query={"limit": limit})

@app.route('/api/phoropter/<device_id>/sync-state', methods=['POST'])
def sync_phoropter_state(device_id):
    payload = _request_payload()
    return _proxy_request("POST", f"/phoropter/{device_id}/sync-state", payload=payload)

@app.route('/api/phoropter/<device_id>/reset', methods=['POST'])
def reset_phoropter(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/reset")

@app.route('/api/phoropter/<device_id>/pinhole', methods=['POST'])
def set_pinhole(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/pinhole")

@app.route('/api/phoropter/<device_id>/screenshot', methods=['POST'])
def take_screenshot(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/screenshot")


# ── Session Management ──────────────────────────────
@app.route('/api/session/intake', methods=['POST'])
def session_intake():
    """Start a new session with patient intake data."""
    payload = request.json or {}
    _log_api_command("/api/session/intake", payload)

    session_id = payload.get("session_id", f"session_{int(datetime.now().timestamp() * 1000)}")
    phoropter_id = payload.get("phoropter_id", "phoropter-1")
    patient_data = payload.get("patient", {})
    patient_data["visit_id"] = session_id

    session = SessionOrchestrator(
        base_url=PHOROPTER_BASE_URL,
        phoropter_id=phoropter_id,
        calibration_path=CALIBRATION_PATH,
    )
    sessions[session_id] = session

    try:
        state = session.initialize(patient_data)
    except Exception as e:
        del sessions[session_id]
        return jsonify({"error": f"Failed to initialize: {str(e)}"}), 400

    return jsonify({
        "session_id": session_id,
        "status": "started",
        **state,
    })


@app.route('/api/session/<session_id>/respond', methods=['POST'])
def respond(session_id):
    """Process patient response."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    payload = request.json or {}
    _log_api_command(f"/api/session/{session_id}/respond", payload)
    response_value = payload.get("response") or payload.get("intent", "")

    if not response_value:
        return jsonify({"error": "response required"}), 400

    next_state = session.process_response(response_value)

    return jsonify({
        "session_id": session_id,
        "status": "active",
        **next_state,
    })


@app.route('/api/session/<session_id>/status', methods=['GET'])
def get_status(session_id):
    """Get full current session state."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    session = sessions[session_id]
    state = session._build_response()
    return jsonify({
        "session_id": session_id,
        "status": "active",
        "total_rows": len(session.session_history),
        **state,
    })


@app.route('/api/session/<session_id>/derived-variables', methods=['GET'])
def get_derived_variables(session_id):
    """Get derived variables and working variables for debug panel."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    session = sessions[session_id]
    return jsonify(session.get_derived_variables_display())


@app.route('/api/session/<session_id>/sync-power', methods=['POST'])
def sync_power(session_id):
    """Sync manual power changes."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    session = sessions[session_id]
    payload = request.json or {}
    session.sync_power(payload.get("right", {}), payload.get("left", {}))
    return jsonify({"session_id": session_id, "status": "success"})


@app.route('/api/session/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """End session and write logs."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    payload = _request_payload()
    _log_api_command(f"/api/session/{session_id}/end", payload)

    store = payload.get("store", True)
    if isinstance(store, str):
        store = store.lower() in ("true", "1", "yes")

    session = sessions[session_id]
    session.session_end_time = datetime.now()

    ar = payload.get("ar")
    lenso = payload.get("lenso")
    operator_name = payload.get("operator_name", "")
    customer_name = payload.get("customer_name", "")
    customer_age = payload.get("customer_age", "")
    customer_gender = payload.get("customer_gender", "")
    qualitative_feedback = payload.get("qualitative_feedback", "")

    # Final prescription
    final_rx = {}
    if session.session_history:
        last_row = session.session_history[-1]
        final_rx = {
            "right_eye": {"sph": last_row.r_sph, "cyl": last_row.r_cyl, "axis": last_row.r_axis, "add": last_row.r_add},
            "left_eye": {"sph": last_row.l_sph, "cyl": last_row.l_cyl, "axis": last_row.l_axis, "add": last_row.l_add},
        }

    remote_status = None
    if store:
        phases_completed = list(session._phases_visited)
        try:
            metadata = build_session_metadata(
                session_id=session_id,
                phoropter_id=session.phoropter_id,
                session_start_time=session.session_start_time or datetime.now(),
                session_end_time=session.session_end_time,
                completion_status="completed",
                rows=session.session_history,
                ar=ar, lensometry=lenso,
                phase_jump_count=session._phase_jump_count,
                phases_completed=phases_completed,
                duration_per_phase=session.get_duration_per_phase(),
                operator_name=operator_name,
                customer_name=customer_name,
                customer_age=customer_age,
                customer_gender=customer_gender,
                qualitative_feedback=qualitative_feedback,
            )
            csv_path = SESSIONS_DIR / f"{session_id}.csv"
            meta_path = SESSIONS_DIR / f"{session_id}_metadata.json"
            write_session_csv(session.session_history, csv_path)
            write_session_metadata(metadata, meta_path)
            append_to_combined_log(session.session_history, session_id, COMBINED_LOG_PATH)
            append_to_combined_metadata(metadata, COMBINED_META_PATH)

            if os.environ.get("REMOTE_STORAGE", "").strip().lower() == "supabase":
                csv_content = session_csv_string(session.session_history)
                err = upload_session_remote(session_id, csv_content, metadata)
                if err:
                    remote_status = {"saved": False, "error": err}
                else:
                    remote_status = {"saved": True}
        except Exception as e:
            print(f"[LOG ERROR] {e}")

    if store:
        del sessions[session_id]

    resp = {
        "session_id": session_id,
        "status": "ended",
        "total_rows": len(session.session_history),
        "final_prescription": final_rx,
    }
    if remote_status:
        resp["remote_storage"] = remote_status
    return jsonify(resp)


@app.route('/api/session/<session_id>/discard', methods=['POST'])
def discard_session(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    del sessions[session_id]
    return jsonify({"session_id": session_id, "status": "discarded"})


# ── Dashboard ───────────────────────────────────────
@app.route('/api/dashboard/config', methods=['GET'])
def dashboard_config_get():
    return jsonify(_dashboard_config)

@app.route('/api/dashboard/config', methods=['PUT'])
def dashboard_config_put():
    payload = request.json or {}
    for key in ("tests_enabled", "daily_limit", "daily_limit_scope", "per_phoropter_enabled"):
        if key in payload:
            _dashboard_config[key] = payload[key]
    return jsonify(_dashboard_config)

@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    source = request.args.get("source", "local")
    if source != "local":
        return jsonify({"error": "Only local source supported in v2", "total_sessions": 0, "recent_sessions": []})
    from_date = request.args.get("from")
    to_date = request.args.get("to")
    operator = request.args.get("operator")
    phoropter = request.args.get("phoropter")
    from datetime import date as _date
    rows = _dashboard.load_metadata_rows(COMBINED_META_PATH)
    fd = _date.fromisoformat(from_date) if from_date else None
    td = _date.fromisoformat(to_date) if to_date else None
    filtered = _dashboard.filter_rows(rows, from_date=fd, to_date=td, operator=operator, phoropter=phoropter)
    by_day = {}
    for r in filtered:
        d = (r.get("Start_Time") or "")[:10]
        if d:
            by_day[d] = by_day.get(d, 0) + 1
    recent = filtered[-20:]
    recent_out = []
    for r in recent:
        recent_out.append({
            "session_id": r.get("Session_ID", ""),
            "start_time": r.get("Start_Time", ""),
            "operator": r.get("Operator_Name", ""),
            "phoropter": r.get("Phoropter_ID", ""),
            "completion_status": r.get("Completion_Status", ""),
            "duration_seconds": r.get("Duration_Seconds", ""),
        })
    return jsonify({
        "total_sessions": len(filtered),
        "active_sessions_count": len(sessions),
        "by_day": by_day,
        "recent_sessions": recent_out,
    })

@app.route('/api/dashboard/rr', methods=['GET'])
def dashboard_rr():
    rows = _dashboard.load_metadata_rows(COMBINED_META_PATH)
    return jsonify(_dashboard.get_rr_aggregates(rows))

@app.route('/api/dashboard/export', methods=['GET'])
def dashboard_export():
    rows = _dashboard.load_metadata_rows(COMBINED_META_PATH)
    import io as _io
    import csv as _csv
    cols = _dashboard.export_metadata_columns()
    buf = _io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    from flask import Response
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=metadata_export.csv"})


# ── Frontend ────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).parent / "frontend"

@app.route("/")
def serve_index():
    return send_from_directory(_FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(_FRONTEND_DIR, path)


if __name__ == '__main__':
    print("Starting Eye Test Engine v2 API Server...")
    print("Endpoints:")
    print("  POST /api/session/intake")
    print("  POST /api/session/<id>/respond")
    print("  GET  /api/session/<id>/status")
    print("  GET  /api/session/<id>/derived-variables")
    print("  POST /api/session/<id>/sync-power")
    print("  POST /api/session/<id>/end")
    print("  POST /api/session/<id>/discard")
    print("  GET  /api/devices")
    print("  POST /api/devices/<id>/acquire|release|heartbeat")
    print("  POST /api/phoropter/<id>/reset|pinhole|screenshot|sync-state")
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host=host, port=port, debug=debug)
