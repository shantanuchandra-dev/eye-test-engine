"""
SessionOrchestrator — wraps RefractionFSMEngine + DerivedVariablesEngine.

Responsibilities:
- Build PatientInput from intake form data
- Invoke DerivedVariablesEngine to compute DerivedVariables
- Drive RefractionFSMEngine step-by-step
- Map FSM states → phoropter commands (chart + power + occluder)
- Track session history for logging
- Provide derived/working variables for debug panel
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ete_io.ist_time import ist_now
from typing import Any, Dict, List, Optional

import requests as http_requests

logger = logging.getLogger(__name__)

from fsm.config.calibration_loader import CalibrationLoader
from fsm.engines.refraction_fsm_engine import COMPACT_PROMPT_CONFIG, RefractionFSMEngine
from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription
from fsm.models.derived_variables import DerivedVariables
from fsm.models.fsm_runtime import FSMRuntimeRow


# ── FSM State → Phoropter Mapping ───────────────────────────────────

STATE_EYE_MAP = {
    "B": "RE", "E": "RE", "F": "RE", "G": "RE",
    "D": "LE", "H": "LE", "I": "LE", "J": "LE",
    "K": "BIN", "P": "RE", "Q": "LE", "R": "BIN",
}

STATE_AUX_LENS_MAP = {
    "B": "AuxLensL",   # Occlude left → test right
    "E": "AuxLensL",
    "F": "AuxLensL",
    "G": "AuxLensL",
    "D": "AuxLensR",   # Occlude right → test left
    "H": "AuxLensR",
    "I": "AuxLensR",
    "J": "AuxLensR",
    "K": "BINO",
    "P": "AuxLensL",
    "Q": "AuxLensR",
    "R": "BINO",
}

STATE_CHART_MAP = {
    "B": "snellen",      # Chart from chart_param
    "D": "snellen",
    "E": "jcc",           # chart_19
    "F": "jcc",           # chart_19
    "G": "duochrome",     # chart_17
    "H": "jcc",
    "I": "jcc",
    "J": "duochrome",
    "K": "bino",          # chart_20
    "P": "near",          # Chart5
    "Q": "near",
    "R": "near",
}

# chart_param → phoropter chart_items
# FSM engine emits chart_param without underscores (e.g. "200150"), so we map both forms
CHART_PARAM_TO_ITEMS = {
    "400": {"tab": "Chart1", "chart_items": ["chart_9"]},
    "200_150": {"tab": "Chart1", "chart_items": ["chart_10"]},
    "200150": {"tab": "Chart1", "chart_items": ["chart_10"]},
    "100_80": {"tab": "Chart1", "chart_items": ["chart_11"]},
    "10080": {"tab": "Chart1", "chart_items": ["chart_11"]},
    "70_60_50": {"tab": "Chart1", "chart_items": ["chart_12"]},
    "706050": {"tab": "Chart1", "chart_items": ["chart_12"]},
    "40_30_25": {"tab": "Chart1", "chart_items": ["chart_13"]},
    "403025": {"tab": "Chart1", "chart_items": ["chart_13"]},
    "20_15_10": {"tab": "Chart1", "chart_items": ["chart_14"]},
    "201510": {"tab": "Chart1", "chart_items": ["chart_14"]},
    "20_20_20": {"tab": "Chart1", "chart_items": ["chart_15"]},
    "202020": {"tab": "Chart1", "chart_items": ["chart_15"]},
    "25_20_15": {"tab": "Chart1", "chart_items": ["chart_16"]},
}

SPECIAL_CHART_ITEMS = {
    "jcc": {"tab": "Chart1", "chart_items": ["chart_19"]},
    "duochrome": {"tab": "Chart1", "chart_items": ["chart_17"]},
    "bino": {"tab": "Chart1", "chart_items": ["chart_20"]},
    "near": {"tab": "Chart5", "chart_items": ["chart_5"]},
}

# Phase display names
STATE_PHASE_DISPLAY = {
    "B": "Coarse Sphere RE",
    "D": "Coarse Sphere LE",
    "E": "JCC Axis RE",
    "F": "JCC Power RE",
    "G": "Duochrome RE",
    "H": "JCC Axis LE",
    "I": "JCC Power LE",
    "J": "Duochrome LE",
    "K": "Binocular Balance",
    "P": "Near Add RE",
    "Q": "Near Add LE",
    "R": "Near Binocular",
    "END": "Test Complete",
    "ESCALATE": "Escalation Required",
}


def _snap_ar_pd_mm(raw: Any) -> float:
    """AR intake PD in mm; broker uses 0.5 mm steps (curl_API.md §3)."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = 64.0
    v = round(v * 2.0) / 2.0
    return max(50.0, min(78.0, v))


class SessionOrchestrator:
    """Drives a single eye test session using the FSMv3.1 engine."""

    def __init__(self, calibration_path: str = "config/calibration.csv",
                 phoropter_base_url: str = ""):
        self.calibration = CalibrationLoader(calibration_path)
        self.engine = RefractionFSMEngine(self.calibration)
        self.dv_engine = DerivedVariablesEngine(self.calibration)
        self.phoropter_base_url = phoropter_base_url or os.environ.get(
            "PHOROPTER_BASE_URL", "https://rajasthan-royals.preprod.lenskart.com"
        )
        self.phoropter_auto_dispatch: bool = True  # Send commands automatically
        self.auto_screenshot: bool = True  # Capture screenshot after each command batch (ON by default)

        # Session state
        self.session_id: str = ""
        self.phoropter_id: str = ""
        self.patient_input: Optional[PatientInput] = None
        self.derived_variables: Optional[DerivedVariables] = None
        self.ar_re: Optional[EyePrescription] = None
        self.ar_le: Optional[EyePrescription] = None
        self.current_row: Optional[FSMRuntimeRow] = None

        # History and tracking
        self.session_history: List[dict] = []
        self.session_start_time: Optional[datetime] = None
        self.phase_start_times: Dict[str, float] = {}
        self.duration_per_phase: Dict[str, float] = {}
        self.phases_completed: List[str] = []
        self.phase_jump_count: int = 0
        self.unable_to_read_count: int = 0
        self.prompt_instance_id: int = 0
        self.session_language: str = "en"
        self.failed_voice_attempts: List[dict] = []

        # Phoropter state tracking (for prev_state commands)
        self._prev_re_sph: float = 0.0
        self._prev_re_cyl: float = 0.0
        self._prev_re_axis: float = 180.0
        self._prev_le_sph: float = 0.0
        self._prev_le_cyl: float = 0.0
        self._prev_le_axis: float = 180.0
        self._prev_add_r: float = 0.0
        self._prev_add_l: float = 0.0
        self._prev_aux_lens: str = "BINO"

        # Conversation log (for frontend debug panel)
        self.conversation_log: List[dict] = []
        # CURL command log
        self.curl_log: List[dict] = []

    def initialize(self, patient_data: dict, session_id: str, phoropter_id: str) -> dict:
        """Initialize session from patient intake form data.

        Returns:
            dict with first question, options, power state, etc.
        """
        self.session_id = session_id
        self.phoropter_id = phoropter_id
        self.session_start_time = ist_now()

        # Build PatientInput from form data
        self.patient_input = self._build_patient_input(patient_data)

        # Build AR prescriptions
        self.ar_re = EyePrescription(
            sphere=patient_data.get("ar_re_sph", 0.0),
            cylinder=patient_data.get("ar_re_cyl", 0.0),
            axis=patient_data.get("ar_re_axis", 180.0),
        )
        self.ar_le = EyePrescription(
            sphere=patient_data.get("ar_le_sph", 0.0),
            cylinder=patient_data.get("ar_le_cyl", 0.0),
            axis=patient_data.get("ar_le_axis", 180.0),
        )

        # Derive variables
        self.derived_variables = self.dv_engine.derive(self.patient_input)

        # Initialize FSM
        self.current_row = self.engine.initialize_row(
            visit_id=session_id,
            dv=self.derived_variables,
            ar_re=self.ar_re,
            ar_le=self.ar_le,
        )

        # Track phase entry
        self._track_phase_entry(self.current_row.state)

        # Log conversation start
        self._log_conversation("system", f"Session started. State: {self.current_row.state}")

        # Step 1: Phoropter baseline (curl_API.md §2 preload, §3 PD Control)
        ar_pd_mm = _snap_ar_pd_mm(patient_data.get("ar_pd_mm", 64.0))
        if self.phoropter_auto_dispatch and self.phoropter_id:
            reset_url = f"{self.phoropter_base_url}/phoropter/{self.phoropter_id}/reset"
            if math.isclose(ar_pd_mm, 64.0):
                self._log_conversation(
                    "system",
                    "Phoropter /reset skipped (PD at default 64 mm)",
                )
            else:
                self._post_to_phoropter(reset_url, {})
                self._log_conversation("system", "Phoropter reset to 0/0/180 (before PD adjust)")
                self._post_to_phoropter(
                    self._phoropter_url(),
                    {
                        "test_cases": [
                            {"case_id": "test_pd_auto", "pd": float(ar_pd_mm)},
                        ]
                    },
                )
                self._log_conversation(
                    "system",
                    f"Phoropter PD set to {ar_pd_mm} mm (test_pd_auto)",
                )

        # Step 2: Set prev-state to 0/0/180 (post-reset state) so the broker
        # can compute the correct delta clicks from zero to the starting Rx
        self._prev_re_sph = 0.0
        self._prev_re_cyl = 0.0
        self._prev_re_axis = 180.0
        self._prev_le_sph = 0.0
        self._prev_le_cyl = 0.0
        self._prev_le_axis = 180.0
        self._prev_add_r = 0.0
        self._prev_add_l = 0.0
        self._prev_aux_lens = "BINO"

        # Step 3: Send chart + power (from 0/0/180 → starting Rx) + JCC eye mode
        phoropter_result = self.dispatch_phoropter_commands(is_phase_entry=True)

        # Step 4: Update prev-state to the starting Rx (for subsequent delta commands)
        self._prev_re_sph = self.current_row.re_sph or 0.0
        self._prev_re_cyl = self.current_row.re_cyl or 0.0
        self._prev_re_axis = self.current_row.re_axis or 180.0
        self._prev_le_sph = self.current_row.le_sph or 0.0
        self._prev_le_cyl = self.current_row.le_cyl or 0.0
        self._prev_le_axis = self.current_row.le_axis or 180.0
        self._prev_aux_lens = STATE_AUX_LENS_MAP.get(self.current_row.state, "BINO")
        self._log_conversation("system", f"Phoropter init: {phoropter_result}")

        return self._build_response()

    def process_response(self, response_value: str, voice_meta: Optional[Dict] = None,
                         input_method: str = "Button") -> dict:
        """Process a patient response and advance the FSM.

        Returns:
            dict with next question, options, power state, etc.
        """
        if self.current_row is None:
            return {"error": "No active session"}

        # Increment prompt instance counter
        self.prompt_instance_id += 1

        prev_state = self.current_row.state

        # Intercept REPEAT before apply_response (matching FSMv3.1_R2 behavior)
        # REPEAT re-displays the same question without recording a row or advancing counters
        normalized = str(response_value or "").strip().upper()
        if normalized in ("REPEAT", "__REPEAT__"):
            # JCC states: send handle (back to flip1), frontend does auto-flip to flip2 after 2s
            if prev_state in ("E", "F", "H", "I") and self.phoropter_auto_dispatch and self.phoropter_id:
                self._send_jcc("handle")  # → flip1
            self._log_conversation(
                "system",
                "Repeating the same step.",
                state=prev_state,
                step=self.current_row.step,
            )
            return self._build_response()  # auto_flip=True in response triggers frontend flip

        # Log the response
        self._log_conversation(
            "patient",
            response_value,
            state=prev_state,
            step=self.current_row.step,
        )

        # Apply response to FSM
        finalized = self.engine.apply_response(
            current=self.current_row,
            response_value=response_value,
            dv=self.derived_variables,
            ar_re=self.ar_re,
            ar_le=self.ar_le,
        )

        # Record the finalized row
        self._record_row(finalized, voice_meta=voice_meta, input_method=input_method)

        # ── STEP 1: Send RESPONSE commands for the PREVIOUS state ──
        # (JCC handle/increase/decrease, duochrome increase/decrease)
        # These must be sent BEFORE transitioning, while we're still in the old state.
        self._dispatch_response_commands(prev_state, response_value)

        # Track phase exit if state changed
        next_state = finalized.next_state
        if next_state != prev_state:
            self._track_phase_exit(prev_state)
            if prev_state not in self.phases_completed:
                self.phases_completed.append(prev_state)

        # Check terminal states
        if next_state in ("END", "ESCALATE"):
            self._track_phase_exit(prev_state)
            self.current_row = finalized
            self.current_row.state = next_state
            self._log_conversation("system", f"Test ended: {next_state}")
            return self._build_response()

        # Build next row
        next_row = self.engine._build_next_row(finalized, self.derived_variables)
        if next_row is None:
            self.current_row = finalized
            self._log_conversation("system", "FSM returned no next row — treating as END")
            return self._build_response(force_end=True)

        self.current_row = next_row

        # Track phase entry if new state
        if next_state != prev_state:
            self._track_phase_entry(next_state)

        # ── STEP 2: Send ENTRY/STEP commands for the NEW state ──
        # IMPORTANT: prev-state is NOT updated yet — so the broker sees the correct delta
        phase_changed = next_state != prev_state
        if phase_changed:
            self.dispatch_phoropter_commands(
                is_phase_entry=True,
                prev_response=response_value,
                prev_state=prev_state,
            )
        elif next_state in ("B", "D", "K", "P", "Q", "R"):
            # Coarse/bino/near: send power update on each step within the phase
            self.dispatch_phoropter_commands(
                is_phase_entry=False,
                prev_response=response_value,
                prev_state=prev_state,
            )

        # Update phoropter prev-state AFTER dispatching (so next command has correct prev)
        self._update_phoropter_prev_state()

        # Log the question
        self._log_conversation(
            "optometrist",
            self.current_row.question,
            state=self.current_row.state,
            step=self.current_row.step,
        )

        return self._build_response()

    def sync_power(self, power_data: dict) -> dict:
        """Sync manual power changes from frontend."""
        if self.current_row is None:
            return {"error": "No active session"}

        if "re_sph" in power_data:
            self.current_row.re_sph = float(power_data["re_sph"])
        if "re_cyl" in power_data:
            self.current_row.re_cyl = float(power_data["re_cyl"])
        if "re_axis" in power_data:
            self.current_row.re_axis = float(power_data["re_axis"])
        if "le_sph" in power_data:
            self.current_row.le_sph = float(power_data["le_sph"])
        if "le_cyl" in power_data:
            self.current_row.le_cyl = float(power_data["le_cyl"])
        if "le_axis" in power_data:
            self.current_row.le_axis = float(power_data["le_axis"])
        if "add_r" in power_data:
            self.current_row.add_r = float(power_data["add_r"])
        if "add_l" in power_data:
            self.current_row.add_l = float(power_data["add_l"])

        # Also update prev-state
        self._update_phoropter_prev_state()

        # Record manual adjustment
        self._record_row(self.current_row, interaction_type="Manual",
                         input_method="Manual_Adjustment")
        self._log_conversation("system", f"Manual power sync: {power_data}")

        return self._build_response()

    def get_status(self) -> dict:
        """Return full session status (for UI restore after refresh)."""
        return self._build_response()

    def get_derived_variables(self) -> dict:
        """Return all derived variables for debug display."""
        if self.derived_variables is None:
            return {}
        return asdict(self.derived_variables)

    def get_phoropter_commands(self) -> dict:
        """Build phoropter command summary for the API response (informational)."""
        if self.current_row is None or self.current_row.state in ("END", "ESCALATE"):
            return {}
        state = self.current_row.state
        return {
            "state": state,
            "aux_lens": STATE_AUX_LENS_MAP.get(state, "BINO"),
            "chart_type": STATE_CHART_MAP.get(state, "snellen"),
            "chart_param": self.current_row.chart_param,
        }

    # ── Private helpers ───────────────────────────────────────────────

    def _build_patient_input(self, data: dict) -> PatientInput:
        """Build PatientInput from intake form data."""
        lenso_re = None
        if data.get("lenso_re_sph") is not None:
            lenso_re = EyePrescription(
                sphere=data.get("lenso_re_sph"),
                cylinder=data.get("lenso_re_cyl"),
                axis=data.get("lenso_re_axis"),
            )
        lenso_le = None
        if data.get("lenso_le_sph") is not None:
            lenso_le = EyePrescription(
                sphere=data.get("lenso_le_sph"),
                cylinder=data.get("lenso_le_cyl"),
                axis=data.get("lenso_le_axis"),
            )

        return PatientInput(
            visit_id=self.session_id,
            age=data.get("age"),
            occupation=data.get("occupation", ""),
            screen_time_hours=data.get("screen_time_hours"),
            driving_hours=data.get("driving_hours"),
            primary_reason=data.get("primary_reason", "Routine check"),
            symptoms_text=data.get("symptoms_text", ""),
            satisfaction_with_current_rx=data.get("satisfaction", "No current Rx"),
            wear_type=data.get("wear_type", "None"),
            distance_target_preference=data.get("distance_target", "6/6_target"),
            priority=data.get("priority", "Standard"),
            near_priority_declared=data.get("near_priority", "Medium"),
            last_eye_test_months_ago=data.get("last_test_months"),
            rx_change_was_large=data.get("rx_change_large", False),
            fluctuating_vision_reported=data.get("fluctuating_vision", False),
            diabetes=data.get("diabetes", False),
            prior_eye_surgery=data.get("prior_surgery", "None"),
            keratoconus=data.get("keratoconus", False),
            amblyopia=data.get("amblyopia", False),
            infection=data.get("infection", False),
            optom_review_flag=data.get("optom_review", False),
            autorefractor_re=EyePrescription(
                sphere=data.get("ar_re_sph", 0.0),
                cylinder=data.get("ar_re_cyl", 0.0),
                axis=data.get("ar_re_axis", 180.0),
            ),
            autorefractor_le=EyePrescription(
                sphere=data.get("ar_le_sph", 0.0),
                cylinder=data.get("ar_le_cyl", 0.0),
                axis=data.get("ar_le_axis", 180.0),
            ),
            lenso_re=lenso_re,
            lenso_le=lenso_le,
            lenso_add_r=data.get("lenso_add_r"),
            lenso_add_l=data.get("lenso_add_l"),
        )

    def _build_response(self, force_end: bool = False) -> dict:
        """Build API response from current FSM state."""
        if self.current_row is None:
            return {"error": "No active session"}

        row = self.current_row
        state = row.state if not force_end else "END"

        # Get options from the row
        options = [
            row.opt_1, row.opt_2, row.opt_3,
            row.opt_4, row.opt_5, row.opt_6,
        ]
        options = [o for o in options if o not in ("", None)]

        is_terminal = state in ("END", "ESCALATE")

        response = {
            "session_id": self.session_id,
            "state": state,
            "phase_name": STATE_PHASE_DISPLAY.get(state, state),
            "phase_type": row.phase_type,
            "eye": STATE_EYE_MAP.get(state, row.eye),
            "step": row.step,
            "question": row.question if not is_terminal else "",
            "options": options if not is_terminal else [],
            "response_type": row.response_type,
            "chart_param": row.chart_param,
            "chart_type": STATE_CHART_MAP.get(state, ""),
            "is_terminal": is_terminal,
            "prescription": {
                "right": {
                    "sph": row.re_sph,
                    "cyl": row.re_cyl,
                    "axis": row.re_axis,
                    "add": row.add_r,
                },
                "left": {
                    "sph": row.le_sph,
                    "cyl": row.le_cyl,
                    "axis": row.le_axis,
                    "add": row.add_l,
                },
            },
            "aux_lens": STATE_AUX_LENS_MAP.get(state, "BINO"),
            "fog_active": getattr(row, "fog_active", False),
            "same_streak": row.same_streak,
            "phase_step_count": row.phase_step_count,
            "phoropter_commands": self.get_phoropter_commands() if not is_terminal else {},
            "auto_flip": state in ("E", "F", "H", "I") and not is_terminal,
            "flip_wait_seconds": 2,
            "flip_state": "flip1",  # Always starts at flip1; frontend sends handle after delay
        }

        return response

    def _record_row(self, row: FSMRuntimeRow, interaction_type: str = "QnA",
                    voice_meta: Optional[Dict] = None,
                    input_method: str = "Button") -> None:
        """Record a finalized FSM row to session history.

        Captures the full FSMRuntimeRow fields + voice metadata + input method.
        """
        # Full FSMRuntimeRow dump
        record = asdict(row)

        # Add session-level fields
        record["row_number"] = len(self.session_history) + 1
        record["timestamp"] = ist_now().isoformat()
        record["interaction_type"] = interaction_type
        record["input_method"] = input_method
        record["prompt_instance_id"] = self.prompt_instance_id
        record["session_language"] = self.session_language
        record["occluder_state"] = STATE_AUX_LENS_MAP.get(row.state, "BINO")
        record["chart_display"] = row.chart_type
        record["change_delta"] = (
            f"ds_re={row.ds_re} dc_re={row.dc_re} da_re={row.da_re} "
            f"ds_le={row.ds_le} dc_le={row.dc_le} da_le={row.da_le}"
        )
        record["phase_display_name"] = STATE_PHASE_DISPLAY.get(row.state, row.state)

        # Voice metadata (if provided)
        if voice_meta:
            # Save audio blob to disk if present
            audio_b64 = voice_meta.pop("audio_base64", None)
            if audio_b64:
                audio_file = self._save_audio_blob(
                    audio_b64, record["row_number"], record["timestamp"]
                )
                voice_meta["audio_file"] = audio_file
            # Serialize alternatives list to semicolon-separated string
            alts = voice_meta.get("alternatives")
            if isinstance(alts, list):
                voice_meta["alternatives"] = "; ".join(str(a) for a in alts)
            record.update(voice_meta)

        self.session_history.append(record)

    def _save_audio_blob(self, audio_base64: str, step: int, timestamp: str) -> str:
        """Decode base64 audio and save to disk. Returns the filename."""
        import base64
        audio_dir = Path("logs/sessions/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        # Clean timestamp for filename
        ts_clean = timestamp.replace(":", "").replace("-", "").replace("T", "_").split(".")[0]
        filename = f"{self.session_id}_step{step}_{ts_clean}.webm"
        filepath = audio_dir / filename
        try:
            raw = base64.b64decode(audio_base64)
            filepath.write_bytes(raw)
        except Exception as e:
            self._log_conversation("system", f"Audio save failed: {e}")
            return ""
        return filename

    def _update_phoropter_prev_state(self) -> None:
        """Update prev-state tracking from current row."""
        if self.current_row is None:
            return
        self._prev_re_sph = self.current_row.re_sph or 0.0
        self._prev_re_cyl = self.current_row.re_cyl or 0.0
        self._prev_re_axis = self.current_row.re_axis or 180.0
        self._prev_le_sph = self.current_row.le_sph or 0.0
        self._prev_le_cyl = self.current_row.le_cyl or 0.0
        self._prev_le_axis = self.current_row.le_axis or 180.0
        self._prev_add_r = self.current_row.add_r or 0.0
        self._prev_add_l = self.current_row.add_l or 0.0
        self._prev_aux_lens = STATE_AUX_LENS_MAP.get(self.current_row.state, "BINO")

    def _track_phase_entry(self, state: str) -> None:
        self.phase_start_times[state] = time.time()

    def _track_phase_exit(self, state: str) -> None:
        if state in self.phase_start_times:
            duration = time.time() - self.phase_start_times[state]
            self.duration_per_phase[state] = self.duration_per_phase.get(state, 0.0) + duration

    def _log_conversation(self, role: str, message: str, **extra) -> None:
        entry = {
            "timestamp": ist_now().isoformat(),
            "role": role,
            "message": message,
            **extra,
        }
        self.conversation_log.append(entry)

    def log_curl_command(self, method: str, url: str, body: Optional[dict] = None,
                        screenshot: Optional[str] = None) -> None:
        """Log a CURL command sent to the phoropter."""
        entry = {
            "timestamp": ist_now().isoformat(),
            "method": method,
            "url": url,
            "body": body,
        }
        if screenshot:
            entry["screenshot"] = screenshot
        self.curl_log.append(entry)

    # ── Phoropter auto-dispatch ─────────────────────────────────────

    def dispatch_phoropter_commands(self, is_phase_entry: bool = False,
                                    prev_response: str = "",
                                    prev_state: str = "") -> dict:
        """Send phoropter commands matching v1 interactive_session.py patterns.

        Called on:
        - Session init (is_phase_entry=True)
        - Each process_response (is_phase_entry=True if state changed, prev_response set)

        Command protocol per state (matching v1 exactly):
        B/D: chart + power(prev_state) + jcc_eye_mode on entry
        E/H: jcc("handle") per flip, jcc("increase"/"decrease") on choice
        F/I: jcc("power_axis_switch") on entry, then handle/increase/decrease
        G/J: chart on entry, jcc("decrease") for RED, jcc("increase") for GREEN
        K:   chart + power + jcc("BINO") on entry, power on each step
        P:   chart_near + power(with ADD) + jcc("R") on entry, power(with ADD) on step
        Q:   chart_near + jcc("L") on entry (NO power to avoid ADD double-click)
        R:   chart_near + jcc("BINO") on entry, single-ADD delta on step
        """
        if not self.phoropter_auto_dispatch or not self.phoropter_id:
            return {"skipped": True, "reason": "auto-dispatch disabled or no phoropter_id"}

        if self.current_row is None or self.current_row.state in ("END", "ESCALATE"):
            return {"skipped": True, "reason": "terminal state"}

        state = self.current_row.state
        results = {}

        # ── Coarse Sphere (B, D) ──
        if state in ("B", "D"):
            if is_phase_entry:
                results["chart"] = self._send_chart_for_state(state)
                results["power"] = self._send_power_with_prev()
                eye_mode = "R" if state == "B" else "L"
                results["jcc_eye"] = self._send_jcc(eye_mode)
            else:
                # Send chart on every step (chart advances on CLEAR responses)
                results["chart"] = self._send_chart_for_state(state)
                results["power"] = self._send_power_with_prev()

        # ── JCC Axis (E, H) — show JCC dot chart on entry ──
        elif state in ("E", "H"):
            if is_phase_entry:
                # v1 relies on TOPCON auto-showing JCC chart, but broker API needs explicit command
                results["chart"] = self._send_chart("Chart1", ["chart_19"])

        # ── JCC Power (F, I) — JCC chart + power_axis_switch on entry ──
        elif state in ("F", "I"):
            if is_phase_entry:
                results["chart"] = self._send_chart("Chart1", ["chart_19"])
                results["jcc_switch"] = self._send_jcc("power_axis_switch")
            # Response commands handled by _dispatch_response_commands

        # ── Duochrome (G, J) — chart on entry only ──
        elif state in ("G", "J"):
            if is_phase_entry:
                results["chart"] = self._send_chart("Chart1", ["chart_17"])
            # Response commands (RED→decrease, GREEN→increase) handled by _dispatch_response_commands

        # ── Binocular Balance (K) ──
        elif state == "K":
            if is_phase_entry:
                results["chart"] = self._send_chart("Chart1", ["chart_20"])
                results["power"] = self._send_power_with_prev()
                results["jcc_bino"] = self._send_jcc("BINO")
            else:
                results["power"] = self._send_power_with_prev()

        # ── Near Add RE (P) ──
        elif state == "P":
            if is_phase_entry:
                results["chart"] = self._send_chart("Chart5", ["chart_5"])
                results["power"] = self._send_power_with_prev(include_add=True)
                results["jcc_eye"] = self._send_jcc("R")
            else:
                results["power"] = self._send_power_with_prev(include_add=True)

        # ── Near Add LE (Q) ──
        elif state == "Q":
            if is_phase_entry:
                results["chart"] = self._send_chart("Chart5", ["chart_5"])
                results["jcc_eye"] = self._send_jcc("L")
                # NO power on entry — avoids ADD double-click (v1 pattern)
            else:
                results["power"] = self._send_power_with_prev(include_add=True)

        # ── Near Binocular (R) ──
        elif state == "R":
            if is_phase_entry:
                results["chart"] = self._send_chart("Chart5", ["chart_5"])
                results["jcc_bino"] = self._send_jcc("BINO")
                # NO power on entry (v1 pattern)
            else:
                results["power"] = self._send_add_bino_delta()

        # Capture screenshot after commands and attach to last curl log entry
        if results and self.auto_screenshot:
            screenshot = self._capture_screenshot()
            if screenshot and self.curl_log:
                self.curl_log[-1]["screenshot"] = screenshot
                results["screenshot"] = True

        return results

    def _capture_screenshot(self) -> Optional[str]:
        """Capture a screenshot from the phoropter. Returns base64 JPEG or None."""
        if not self.phoropter_id:
            return None
        url = f"{self.phoropter_base_url}/phoropter/{self.phoropter_id}/screenshot"
        try:
            resp = http_requests.post(url, timeout=10)
            if resp.ok:
                # Broker returns base64 string, sometimes wrapped in quotes
                text = resp.text.strip().strip('"')
                if text.startswith('/9j/') or text.startswith('iVBOR'):
                    return text
        except Exception as e:
            logger.debug(f"Screenshot failed: {e}")
        return None

    def _dispatch_response_commands(self, state: str, response_value: str) -> None:
        """Send phoropter commands for a patient RESPONSE in the given state.
        Called BEFORE state transition. Handles JCC flip/adjust and duochrome SPH.

        JCC REPEAT: sends handle + handle (flip back to pos1, then re-show pos2)
        matching v1's handle + auto_flip pattern.
        Non-JCC REPEAT: no CURL commands (already intercepted by process_response).
        """
        if not self.phoropter_auto_dispatch or not self.phoropter_id:
            return
        resp = response_value.upper()

        # JCC Axis (E, H): increase/decrease + handle
        # REPEAT handled in process_response before this is called
        if state in ("E", "H"):
            if resp in ("ONE", "BETTER_1"):
                self._send_jcc("increase")
                self._send_jcc("handle")
            elif resp in ("TWO", "BETTER_2"):
                self._send_jcc("decrease")
                self._send_jcc("handle")

        # JCC Power (F, I): increase/decrease + handle
        elif state in ("F", "I"):
            if resp in ("ONE", "BETTER_1"):
                self._send_jcc("increase")
                self._send_jcc("handle")
            elif resp in ("TWO", "BETTER_2"):
                self._send_jcc("decrease")
                self._send_jcc("handle")

        # Duochrome (G, J): decrease for RED, increase for GREEN
        # REPEAT/SAME: no CURL commands
        elif state in ("G", "J"):
            if resp in ("RED", "RED_CLEARER"):
                self._send_jcc("decrease")  # RAM: Red Add Minus
            elif resp in ("GREEN", "GREEN_CLEARER"):
                self._send_jcc("increase")  # GAP: Green Add Plus

        # All other states: no response-level CURL commands
        # B/D coarse, K binocular, P/Q/R near: power updates handled by dispatch_phoropter_commands

    def _phoropter_url(self, path: str = "/run-tests") -> str:
        return f"{self.phoropter_base_url}/phoropter/{self.phoropter_id}{path}"

    def _post_to_phoropter(self, url: str, body: dict) -> dict:
        """Send a POST request to the phoropter broker and log it."""
        self.log_curl_command("POST", url, body)
        try:
            resp = http_requests.post(url, json=body, timeout=15)
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            result = {"status": resp.status_code, "data": data}
            logger.info(f"Phoropter POST {url} → {resp.status_code}")
            return result
        except http_requests.exceptions.RequestException as e:
            logger.warning(f"Phoropter POST {url} failed: {e}")
            return {"status": 0, "error": str(e)}

    def _send_jcc(self, action: str) -> dict:
        """Send a JCC command: handle, increase, decrease, power_axis_switch, R, L, BINO."""
        url = self._phoropter_url()
        return self._post_to_phoropter(url, {"test_cases": [{"jcc": action}]})

    def _send_chart(self, tab: str, chart_items: list) -> dict:
        """Send a chart selection command."""
        url = self._phoropter_url()
        return self._post_to_phoropter(url, {
            "test_cases": [{"chart": {"tab": tab, "chart_items": chart_items}}]
        })

    def _send_chart_for_state(self, state: str) -> dict:
        """Send the correct chart for the current FSM state."""
        chart_type = STATE_CHART_MAP.get(state, "snellen")
        if chart_type in SPECIAL_CHART_ITEMS:
            info = SPECIAL_CHART_ITEMS[chart_type]
            return self._send_chart(info["tab"], info["chart_items"])
        chart_param = self.current_row.chart_param or "20_20_20"
        info = CHART_PARAM_TO_ITEMS.get(chart_param, {"tab": "Chart1", "chart_items": ["chart_15"]})
        return self._send_chart(info["tab"], info["chart_items"])

    def _send_power_with_prev(self, include_add: bool = False) -> dict:
        """Send run-tests with prev_state for delta calculation (matching v1 set_power_with_prev_state)."""
        row = self.current_row
        state = row.state
        aux_lens = STATE_AUX_LENS_MAP.get(state, "BINO")

        prev_right = {"sph": self._prev_re_sph, "cyl": self._prev_re_cyl, "axis": self._prev_re_axis}
        prev_left = {"sph": self._prev_le_sph, "cyl": self._prev_le_cyl, "axis": self._prev_le_axis}
        right_eye = {"sph": row.re_sph or 0.0, "cyl": row.re_cyl or 0.0, "axis": row.re_axis or 180.0}
        left_eye = {"sph": row.le_sph or 0.0, "cyl": row.le_cyl or 0.0, "axis": row.le_axis or 180.0}

        if include_add:
            if row.add_r is not None:
                right_eye["add"] = row.add_r
            if row.add_l is not None:
                left_eye["add"] = row.add_l
            if self._prev_add_r:
                prev_right["add"] = self._prev_add_r
            if self._prev_add_l:
                prev_left["add"] = self._prev_add_l

        payload = {
            "test_cases": [{
                "case_id": 1,
                "prev_aux_lens": self._prev_aux_lens,
                "prev_right_eye": prev_right,
                "prev_left_eye": prev_left,
                "aux_lens": aux_lens,
                "right_eye": right_eye,
                "left_eye": left_eye,
            }]
        }
        return self._post_to_phoropter(self._phoropter_url(), payload)

    def _send_add_bino_delta(self) -> dict:
        """Send binocular ADD with single-click (matching v1 _set_add_bino_delta_only).
        Only right_eye carries ADD; left_eye omits it to avoid double-click."""
        row = self.current_row
        prev_right = {"sph": self._prev_re_sph, "cyl": self._prev_re_cyl, "axis": self._prev_re_axis}
        prev_left = {"sph": self._prev_le_sph, "cyl": self._prev_le_cyl, "axis": self._prev_le_axis}
        right_eye = {"sph": row.re_sph or 0.0, "cyl": row.re_cyl or 0.0, "axis": row.re_axis or 180.0}
        left_eye = {"sph": row.le_sph or 0.0, "cyl": row.le_cyl or 0.0, "axis": row.le_axis or 180.0}

        # Only right_eye gets ADD (left_eye deliberately omits it)
        if row.add_r is not None:
            right_eye["add"] = row.add_r
        if self._prev_add_r:
            prev_right["add"] = self._prev_add_r

        payload = {
            "test_cases": [{
                "case_id": 1,
                "prev_right_eye": prev_right,
                "prev_left_eye": prev_left,
                "right_eye": right_eye,
                "left_eye": left_eye,
                "aux_lens": "BINO",
            }]
        }
        return self._post_to_phoropter(self._phoropter_url(), payload)
