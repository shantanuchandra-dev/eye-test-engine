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

import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
CHART_PARAM_TO_ITEMS = {
    "400": {"tab": "Chart1", "chart_items": ["chart_9"]},
    "200_150": {"tab": "Chart1", "chart_items": ["chart_10"]},
    "100_80": {"tab": "Chart1", "chart_items": ["chart_11"]},
    "70_60_50": {"tab": "Chart1", "chart_items": ["chart_12"]},
    "40_30_25": {"tab": "Chart1", "chart_items": ["chart_13"]},
    "20_15_10": {"tab": "Chart1", "chart_items": ["chart_14"]},
    "20_20_20": {"tab": "Chart1", "chart_items": ["chart_15"]},
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


class SessionOrchestrator:
    """Drives a single eye test session using the FSMv3.1 engine."""

    def __init__(self, calibration_path: str = "config/calibration.csv"):
        self.calibration = CalibrationLoader(calibration_path)
        self.engine = RefractionFSMEngine(self.calibration)
        self.dv_engine = DerivedVariablesEngine(self.calibration)

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
        self.session_start_time = datetime.now()

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

        # Initialize phoropter prev-state from starting prescription
        self._prev_re_sph = self.current_row.re_sph or 0.0
        self._prev_re_cyl = self.current_row.re_cyl or 0.0
        self._prev_re_axis = self.current_row.re_axis or 180.0
        self._prev_le_sph = self.current_row.le_sph or 0.0
        self._prev_le_cyl = self.current_row.le_cyl or 0.0
        self._prev_le_axis = self.current_row.le_axis or 180.0
        self._prev_aux_lens = STATE_AUX_LENS_MAP.get(self.current_row.state, "BINO")

        # Log conversation start
        self._log_conversation("system", f"Session started. State: {self.current_row.state}")

        return self._build_response()

    def process_response(self, response_value: str, voice_meta: Optional[Dict] = None) -> dict:
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
            self._log_conversation(
                "system",
                "Repeating the same step.",
                state=prev_state,
                step=self.current_row.step,
            )
            return self._build_response()

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
        self._record_row(finalized, voice_meta=voice_meta)

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

        # Update phoropter prev-state
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
        self._record_row(self.current_row, interaction_type="Manual")
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
        """Build phoropter command payload for current FSM state.

        Returns dict with:
        - chart: {tab, chart_items} for chart selection
        - run_tests: full payload for /run-tests endpoint
        - jcc_mode: optional JCC command (power_axis_switch, etc.)
        """
        if self.current_row is None:
            return {}

        state = self.current_row.state
        if state in ("END", "ESCALATE"):
            return {}

        commands = {}

        # 1. Chart command
        chart_type = STATE_CHART_MAP.get(state, "snellen")
        if chart_type in SPECIAL_CHART_ITEMS:
            commands["chart"] = SPECIAL_CHART_ITEMS[chart_type]
        else:
            chart_param = self.current_row.chart_param or "20_20_20"
            chart_info = CHART_PARAM_TO_ITEMS.get(chart_param)
            if chart_info:
                commands["chart"] = chart_info
            else:
                commands["chart"] = {"tab": "Chart1", "chart_items": ["chart_15"]}

        # 2. Run-tests payload (power + occluder with prev-state)
        aux_lens = STATE_AUX_LENS_MAP.get(state, "BINO")
        commands["run_tests"] = {
            "test_cases": [{
                "case_id": 1,
                "prev_aux_lens": self._prev_aux_lens,
                "prev_right_eye": {
                    "sph": self._prev_re_sph,
                    "cyl": self._prev_re_cyl,
                    "axis": self._prev_re_axis,
                },
                "prev_left_eye": {
                    "sph": self._prev_le_sph,
                    "cyl": self._prev_le_cyl,
                    "axis": self._prev_le_axis,
                },
                "aux_lens": aux_lens,
                "right_eye": {
                    "sph": self.current_row.re_sph or 0.0,
                    "cyl": self.current_row.re_cyl or 0.0,
                    "axis": self.current_row.re_axis or 180.0,
                },
                "left_eye": {
                    "sph": self.current_row.le_sph or 0.0,
                    "cyl": self.current_row.le_cyl or 0.0,
                    "axis": self.current_row.le_axis or 180.0,
                },
            }],
        }

        # 3. JCC mode for JCC states
        if state in ("E", "H"):
            commands["jcc_mode"] = "axis"
        elif state in ("F", "I"):
            commands["jcc_mode"] = "power"

        return commands

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
        }

        return response

    def _record_row(self, row: FSMRuntimeRow, interaction_type: str = "QnA",
                    voice_meta: Optional[Dict] = None) -> None:
        """Record a finalized FSM row to session history.

        Captures the full FSMRuntimeRow fields + voice metadata.
        """
        # Full FSMRuntimeRow dump
        record = asdict(row)

        # Add session-level fields
        record["row_number"] = len(self.session_history) + 1
        record["timestamp"] = datetime.now().isoformat()
        record["interaction_type"] = interaction_type
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
            record.update(voice_meta)

        self.session_history.append(record)

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
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
            **extra,
        }
        self.conversation_log.append(entry)

    def log_curl_command(self, method: str, url: str, body: Optional[dict] = None) -> None:
        """Log a CURL command sent to the phoropter."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "url": url,
            "body": body,
        }
        self.curl_log.append(entry)
