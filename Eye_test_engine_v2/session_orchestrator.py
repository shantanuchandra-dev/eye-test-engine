"""
Session orchestrator for Eye Test Engine v2.
Wraps FSMv2.2 RefractionFSMEngine with phoropter API integration.
"""
import importlib.util as _ilu
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path as _Path
from typing import Dict, List, Optional

from fsm.calibration import CalibrationLoader
from fsm.derived_variables_engine import DerivedVariablesEngine
from fsm.derived_variables import DerivedVariables
from fsm.patient import PatientInput
from fsm.prescription import EyePrescription
from fsm.refraction_engine import RefractionFSMEngine
from fsm.fsm_runtime import FSMRuntimeRow

# Load io/outputs.py directly to avoid clash with Python's built-in io module
_spec = _ilu.spec_from_file_location(
    "io_outputs",
    str(_Path(__file__).resolve().parent / "io" / "outputs.py"),
)
_outputs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_outputs)
SessionRow = _outputs.SessionRow

# Chart param to phoropter chart_items mapping
CHART_PARAM_MAP = {
    "400": {"tab": "Chart1", "chart_items": ["chart_9"]},        # E-chart 400
    "200_150": {"tab": "Chart1", "chart_items": ["chart_10"]},   # Snellen 200/150
    "100_80": {"tab": "Chart1", "chart_items": ["chart_11"]},    # Snellen 100/80
    "70_60_50": {"tab": "Chart1", "chart_items": ["chart_12"]},  # Snellen 70/60/50
    "40_30_25": {"tab": "Chart1", "chart_items": ["chart_13"]},  # Snellen 40/30/25
    "20_20_20": {"tab": "Chart1", "chart_items": ["chart_15"]},  # Snellen 20/20/20
}

# State to phoropter aux_lens mapping
STATE_OCCLUDER_MAP = {
    "A": "BINO",        # Distance baseline - both eyes
    "B": "AuxLensL",    # Coarse sphere RE - occlude left
    "D": "AuxLensR",    # Coarse sphere LE - occlude right
    "E": "AuxLensL",    # JCC Axis RE
    "F": "AuxLensL",    # JCC Power RE
    "G": "AuxLensL",    # Duochrome RE
    "H": "AuxLensR",    # JCC Axis LE
    "I": "AuxLensR",    # JCC Power LE
    "J": "AuxLensR",    # Duochrome LE
    "K": "BINO",        # Binocular Balance
    "P": "AuxLensL",    # Near Add RE
    "Q": "AuxLensR",    # Near Add LE
    "R": "BINO",        # Near Binocular
}

# State to chart type mapping
STATE_CHART_MAP = {
    "A": "snellen",       # Distance baseline
    "B": "snellen",       # Coarse sphere RE
    "D": "snellen",       # Coarse sphere LE
    "E": "jcc",           # JCC Axis RE
    "F": "jcc",           # JCC Power RE
    "G": "duochrome",     # Duochrome RE
    "H": "jcc",           # JCC Axis LE
    "I": "jcc",           # JCC Power LE
    "J": "duochrome",     # Duochrome LE
    "K": "bino",          # Binocular Balance
    "P": "near",          # Near Add RE
    "Q": "near",          # Near Add LE
    "R": "near",          # Near Binocular
}

# Phase names for UI display
STATE_NAMES = {
    "A": "Phase A: Distance Baseline",
    "B": "Phase B: Coarse Sphere RE",
    "D": "Phase D: Coarse Sphere LE",
    "E": "Phase E: JCC Axis RE",
    "F": "Phase F: JCC Power RE",
    "G": "Phase G: Duochrome RE",
    "H": "Phase H: JCC Axis LE",
    "I": "Phase I: JCC Power LE",
    "J": "Phase J: Duochrome LE",
    "K": "Phase K: Binocular Balance",
    "P": "Phase P: Near Add RE",
    "Q": "Phase Q: Near Add LE",
    "R": "Phase R: Near Binocular",
    "END": "Test Complete",
    "ESCALATE": "Escalation Required",
}


class SessionOrchestrator:
    """Orchestrates an eye test session using FSMv2.2 engine."""

    def __init__(self, base_url: str, phoropter_id: str, calibration_path: str):
        self.base_url = base_url
        self.phoropter_id = phoropter_id
        self.api_endpoint = f"{base_url}/phoropter/{phoropter_id}/run-tests"

        # Load calibration
        self.calibration = CalibrationLoader(calibration_path)
        self.dv_engine = DerivedVariablesEngine(self.calibration)
        self.fsm_engine = RefractionFSMEngine(self.calibration)

        # Session state
        self.patient_input: Optional[PatientInput] = None
        self.derived_vars: Optional[DerivedVariables] = None
        self.current_row: Optional[FSMRuntimeRow] = None
        self.visit_id: str = ""

        # History for logging
        self.session_history: List[SessionRow] = []
        self.session_start_time: Optional[datetime] = None
        self.session_end_time: Optional[datetime] = None
        self._row_counter = 0
        self._phases_visited: List[str] = []
        self._phase_start_times: Dict[str, datetime] = {}
        self._phase_jump_count = 0

        # Previous power state for phoropter delta commands
        self._prev_re = {"sph": 0.0, "cyl": 0.0, "axis": 180.0}
        self._prev_le = {"sph": 0.0, "cyl": 0.0, "axis": 180.0}
        self._prev_aux_lens = "BINO"
        self._prev_add_r = 0.0
        self._prev_add_l = 0.0
        self._jcc_mode = None  # Track current JCC mode: "axis" or "power"

    def initialize(self, patient_data: dict) -> dict:
        """Initialize session with patient data. Returns first stimulus."""
        self.session_start_time = datetime.now()
        self.visit_id = patient_data.get(
            "visit_id", f"session_{int(datetime.now().timestamp() * 1000)}"
        )

        # Build PatientInput
        self.patient_input = self._build_patient_input(patient_data)

        # Derive variables
        self.derived_vars = self.dv_engine.derive(self.patient_input)

        # Initialize FSM
        ar_re = self.patient_input.autorefractor_re
        ar_le = self.patient_input.autorefractor_le
        self.current_row = self.fsm_engine.initialize_row(
            visit_id=self.visit_id,
            dv=self.derived_vars,
            ar_re=ar_re,
            ar_le=ar_le,
        )

        # Track initial power state
        self._prev_re = {
            "sph": self.current_row.re_sph or 0.0,
            "cyl": self.current_row.re_cyl or 0.0,
            "axis": self.current_row.re_axis or 180.0,
        }
        self._prev_le = {
            "sph": self.current_row.le_sph or 0.0,
            "cyl": self.current_row.le_cyl or 0.0,
            "axis": self.current_row.le_axis or 180.0,
        }

        # Send initial phoropter commands
        self._send_phoropter_commands(self.current_row)
        self._track_phase_entry(self.current_row.state)

        return self._build_response()

    def process_response(self, response_value: str) -> dict:
        """Process patient response, advance FSM, send phoropter commands."""
        if self.current_row is None:
            return {"error": "Session not initialized"}

        # Apply response to FSM
        finalized = self.fsm_engine.apply_response(
            current=self.current_row,
            response_value=response_value,
            dv=self.derived_vars,
            ar_re=self.patient_input.autorefractor_re if self.patient_input else None,
            ar_le=self.patient_input.autorefractor_le if self.patient_input else None,
        )

        # Record to history
        self._record_row(finalized, response_value)

        # Check for end states
        if finalized.next_state in ("END", "ESCALATE"):
            self.current_row = finalized
            return self._build_response(terminal=True)

        # Build next row
        next_row = self.fsm_engine._build_next_row(finalized, self.derived_vars)
        if next_row is None:
            self.current_row = finalized
            return self._build_response(terminal=True)

        # Track state change
        if next_row.state != self.current_row.state:
            self._track_phase_entry(next_row.state)

        self.current_row = next_row

        # Send phoropter commands for new state
        self._send_phoropter_commands(self.current_row)

        return self._build_response()

    def get_derived_variables_display(self) -> dict:
        """Return derived variables and working variables for the debug panel."""
        result: Dict[str, dict] = {
            "derived_variables": {},
            "working_variables": {},
            "calibration": {},
        }

        if self.derived_vars:
            dv = self.derived_vars
            result["derived_variables"] = {
                "Age Bucket": dv.dv_age_bucket,
                "Distance Priority": dv.dv_distance_priority,
                "Near Priority": dv.dv_near_priority,
                "Symptom Risk": dv.dv_symptom_risk_level,
                "Medical Risk": dv.dv_medical_risk_level,
                "Stability": dv.dv_stability_level,
                "AR/Lenso Mismatch RE": dv.dv_ar_lenso_mismatch_level_RE,
                "AR/Lenso Mismatch LE": dv.dv_ar_lenso_mismatch_level_LE,
                "Start Source Policy": dv.dv_start_source_policy,
                "Start Rx RE": f"{dv.dv_start_rx_RE_sph}/{dv.dv_start_rx_RE_cyl}x{dv.dv_start_rx_RE_axis}",
                "Start Rx LE": f"{dv.dv_start_rx_LE_sph}/{dv.dv_start_rx_LE_cyl}x{dv.dv_start_rx_LE_axis}",
                "ADD Expected": dv.dv_add_expected,
                "Target Distance VA": dv.dv_target_distance_va,
                "Endpoint Bias": dv.dv_endpoint_bias_policy,
                "Step Size Policy": dv.dv_step_size_policy,
                "Max Delta from Start SPH": dv.dv_max_delta_from_start_sph,
                "Max Delta from AR SPH": dv.dv_max_delta_from_ar_sph,
                "Axis Tolerance": f"{dv.dv_axis_tolerance_deg}\u00b0",
                "CYL Tolerance": f"{dv.dv_cyl_tolerance_D}D",
                "Requires Optom Review": dv.dv_requires_optom_review,
                "Anomaly Watch": dv.dv_anomaly_watch,
                "Expected Convergence": dv.dv_expected_convergence_time,
                "Branching Guardrails": dv.dv_branching_guardrails,
                "Confidence Requirement": dv.dv_confidence_requirement,
                "Fogging Policy": dv.dv_fogging_policy,
                "Fogging Amount": f"{dv.dv_fogging_amount_D}D",
                "Fogging Clearance Mode": dv.dv_fogging_clearance_mode,
                "Axis Step Policy": dv.dv_axis_step_policy,
                "Duochrome Max Flips": dv.dv_duochrome_max_flips,
                "Near Test Required": dv.dv_near_test_required,
            }

        if self.current_row:
            row = self.current_row
            result["working_variables"] = {
                "Current State": row.state,
                "Phase Name": row.phase_name,
                "Step": row.step,
                "Phase Step Count": row.phase_step_count,
                "Same Streak": row.same_streak,
                "Duo Iteration": row.duo_iter,
                "Duo Flip Count": row.duo_flip,
                "Prev Axis Response": row.prev_axis_response or "(none)",
                "Axis Step": f"{row.axis_step}\u00b0",
                "SPH Step": f"{row.sph_step}D",
                "CYL Step": f"{row.cyl_step}D",
                "Chart Param": row.chart_param,
                "Target Chart Idx": row.target_chart_idx,
                "Current Chart Idx": row.chart_idx,
                "Last Delta RE": f"SPH:{row.ds_re:+.2f} CYL:{row.dc_re:+.2f} AXIS:{row.da_re:+.1f}",
                "Last Delta LE": f"SPH:{row.ds_le:+.2f} CYL:{row.dc_le:+.2f} AXIS:{row.da_le:+.1f}",
                "Last Delta ADD": f"R:{row.dadd_r:+.2f} L:{row.dadd_l:+.2f}",
                "RE Power": f"{row.re_sph}/{row.re_cyl}x{row.re_axis}",
                "LE Power": f"{row.le_sph}/{row.le_cyl}x{row.le_axis}",
                "ADD": f"R:{row.add_r} L:{row.add_l}",
                "Next State": row.next_state,
            }

        return result

    def sync_power(self, right: dict, left: dict) -> None:
        """Sync manual power changes from frontend.
        Updates both the FSM row and the prev-state tracker so that the
        next _send_phoropter_commands() computes deltas from the correct
        baseline (the manually-set values, not the stale pre-manual values).
        """
        if self.current_row is None:
            return
        if "sph" in right:
            self.current_row.re_sph = float(right["sph"])
        if "cyl" in right:
            self.current_row.re_cyl = float(right["cyl"])
        if "axis" in right:
            self.current_row.re_axis = float(right["axis"])
        if "add" in right:
            self.current_row.add_r = float(right["add"])
        if "sph" in left:
            self.current_row.le_sph = float(left["sph"])
        if "cyl" in left:
            self.current_row.le_cyl = float(left["cyl"])
        if "axis" in left:
            self.current_row.le_axis = float(left["axis"])
        if "add" in left:
            self.current_row.add_l = float(left["add"])

        # Update prev-state tracker to match the manual values.
        # Without this, the next _send_phoropter_commands() would compute
        # deltas from the old pre-manual baseline, causing incorrect clicks.
        self._prev_re = {
            "sph": self.current_row.re_sph or 0.0,
            "cyl": self.current_row.re_cyl or 0.0,
            "axis": self.current_row.re_axis or 180.0,
        }
        self._prev_le = {
            "sph": self.current_row.le_sph or 0.0,
            "cyl": self.current_row.le_cyl or 0.0,
            "axis": self.current_row.le_axis or 180.0,
        }
        self._prev_add_r = self.current_row.add_r or 0.0
        self._prev_add_l = self.current_row.add_l or 0.0

        # Record manual row
        self._record_manual_row(right, left)

    def get_duration_per_phase(self) -> Dict[str, float]:
        """Calculate time spent in each phase."""
        durations: Dict[str, float] = {}
        if not self.session_history:
            return durations
        for i, row in enumerate(self.session_history):
            state = row.state or "unknown"
            if i + 1 < len(self.session_history):
                try:
                    t0 = datetime.fromisoformat(row.timestamp)
                    t1 = datetime.fromisoformat(self.session_history[i + 1].timestamp)
                    durations[state] = durations.get(state, 0.0) + (t1 - t0).total_seconds()
                except (ValueError, TypeError):
                    pass
        return durations

    # ---- Private helpers ------------------------------------------------

    def _build_patient_input(self, data: dict) -> PatientInput:
        """Build PatientInput from frontend form data."""

        def _rx(d: Optional[dict]) -> Optional[EyePrescription]:
            if not d:
                return None
            sph = d.get("sph")
            cyl = d.get("cyl")
            axis = d.get("axis")
            if sph is None and cyl is None and axis is None:
                return None
            return EyePrescription(
                sphere=float(sph) if sph is not None else None,
                cylinder=float(cyl) if cyl is not None else None,
                axis=float(axis) if axis is not None else None,
            )

        ar = data.get("autorefractor", {})
        lenso = data.get("lensometry", {})

        return PatientInput(
            visit_id=self.visit_id,
            age=int(data["age"]) if data.get("age") else None,
            occupation=data.get("occupation", ""),
            screen_time_hours=float(data["screen_time_hours"]) if data.get("screen_time_hours") else None,
            driving_hours=float(data["driving_hours"]) if data.get("driving_hours") else None,
            primary_reason=data.get("primary_reason", ""),
            symptoms_text=data.get("symptoms_text", ""),
            satisfaction_with_current_rx=data.get("satisfaction", ""),
            wear_type=data.get("wear_type", ""),
            distance_target_preference=data.get("distance_target", ""),
            priority=data.get("priority", "Standard"),
            near_priority_declared=data.get("near_priority", ""),
            last_eye_test_months_ago=float(data["last_eye_test_months"]) if data.get("last_eye_test_months") else None,
            rx_change_was_large=bool(data.get("rx_change_large", False)),
            fluctuating_vision_reported=bool(data.get("fluctuating_vision", False)),
            diabetes=bool(data.get("diabetes", False)),
            prior_eye_surgery=data.get("prior_surgery", "None"),
            keratoconus=bool(data.get("keratoconus", False)),
            amblyopia=bool(data.get("amblyopia", False)),
            infection=bool(data.get("infection", False)),
            optom_review_flag=bool(data.get("optom_review", False)),
            autorefractor_re=_rx(ar.get("right")),
            autorefractor_le=_rx(ar.get("left")),
            lenso_re=_rx(lenso.get("right")),
            lenso_le=_rx(lenso.get("left")),
            lenso_add_r=float(lenso["add_r"]) if lenso.get("add_r") is not None else None,
            lenso_add_l=float(lenso["add_l"]) if lenso.get("add_l") is not None else None,
        )

    def _build_response(self, terminal: bool = False) -> dict:
        """Build response dict for the API."""
        row = self.current_row
        if row is None:
            return {"error": "No current state"}

        state = row.next_state if terminal else row.state
        phase_name = STATE_NAMES.get(state, state)

        # Collect options (non-empty opt_1..opt_6)
        options = []
        for attr in ["opt_1", "opt_2", "opt_3", "opt_4", "opt_5", "opt_6"]:
            val = getattr(row, attr, "")
            if val:
                options.append(val)

        response = {
            "state": state,
            "phase_name": phase_name,
            "phase_type": row.phase_type,
            "question": row.question,
            "options": options,
            "response_type": row.response_type,
            "eye": row.eye,
            "chart_param": row.chart_param,
            "chart_type": STATE_CHART_MAP.get(state, ""),
            "power": {
                "right": {
                    "sph": row.re_sph or 0.0,
                    "cyl": row.re_cyl or 0.0,
                    "axis": row.re_axis or 180.0,
                    "add": row.add_r or 0.0,
                },
                "left": {
                    "sph": row.le_sph or 0.0,
                    "cyl": row.le_cyl or 0.0,
                    "axis": row.le_axis or 180.0,
                    "add": row.add_l or 0.0,
                },
            },
            "step_info": {
                "step": row.step,
                "phase_step_count": row.phase_step_count,
                "same_streak": row.same_streak,
                "axis_step": row.axis_step,
                "sph_step": row.sph_step,
                "cyl_step": row.cyl_step,
            },
            "is_terminal": terminal,
            "total_rows": len(self.session_history),
        }

        if terminal:
            response["terminal_state"] = row.next_state

        return response

    def _record_row(self, finalized: FSMRuntimeRow, response_value: str) -> None:
        """Record a finalized FSM row to session history."""
        self._row_counter += 1
        sr = SessionRow(
            row_number=self._row_counter,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            interaction_type="QnA",
            r_sph=finalized.re_sph or 0.0,
            r_cyl=finalized.re_cyl or 0.0,
            r_axis=finalized.re_axis or 180.0,
            r_add=finalized.add_r or 0.0,
            l_sph=finalized.le_sph or 0.0,
            l_cyl=finalized.le_cyl or 0.0,
            l_axis=finalized.le_axis or 180.0,
            l_add=finalized.add_l or 0.0,
            occluder_state=STATE_OCCLUDER_MAP.get(finalized.state, ""),
            chart_param=finalized.chart_param,
            chart_display=finalized.chart_type,
            change_delta=self._compute_delta_str(finalized),
            phase_name=finalized.phase_name,
            state=finalized.state,
            question=finalized.question,
            response_value=response_value,
        )
        self.session_history.append(sr)

    def _record_manual_row(self, right: dict, left: dict) -> None:
        """Record a manual power change to session history."""
        self._row_counter += 1
        row = self.current_row
        sr = SessionRow(
            row_number=self._row_counter,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            interaction_type="Manual",
            r_sph=row.re_sph or 0.0,
            r_cyl=row.re_cyl or 0.0,
            r_axis=row.re_axis or 180.0,
            r_add=row.add_r or 0.0,
            l_sph=row.le_sph or 0.0,
            l_cyl=row.le_cyl or 0.0,
            l_axis=row.le_axis or 180.0,
            l_add=row.add_l or 0.0,
            occluder_state=STATE_OCCLUDER_MAP.get(row.state, ""),
            chart_param=row.chart_param,
            chart_display=row.chart_type,
            change_delta="Manual Adjust",
            phase_name=row.phase_name,
            state=row.state,
            question="",
            response_value="",
        )
        self.session_history.append(sr)

    def _compute_delta_str(self, row: FSMRuntimeRow) -> str:
        """Compute human-readable delta string."""
        parts = []
        if abs(row.ds_re) > 0.001:
            parts.append(f"RE_SPH:{row.ds_re:+.2f}")
        if abs(row.dc_re) > 0.001:
            parts.append(f"RE_CYL:{row.dc_re:+.2f}")
        if abs(row.da_re) > 0.1:
            parts.append(f"RE_AXIS:{row.da_re:+.0f}")
        if abs(row.ds_le) > 0.001:
            parts.append(f"LE_SPH:{row.ds_le:+.2f}")
        if abs(row.dc_le) > 0.001:
            parts.append(f"LE_CYL:{row.dc_le:+.2f}")
        if abs(row.da_le) > 0.1:
            parts.append(f"LE_AXIS:{row.da_le:+.0f}")
        if abs(row.dadd_r) > 0.001:
            parts.append(f"ADD_R:{row.dadd_r:+.2f}")
        if abs(row.dadd_l) > 0.001:
            parts.append(f"ADD_L:{row.dadd_l:+.2f}")
        return ", ".join(parts) if parts else "No change"

    def _track_phase_entry(self, state: str) -> None:
        """Record phase entry for duration tracking."""
        self._phase_start_times[state] = datetime.now()
        if state not in self._phases_visited:
            self._phases_visited.append(state)

    def _send_phoropter_commands(self, row: FSMRuntimeRow) -> None:
        """Send appropriate phoropter commands for the current FSM state."""
        state = row.state

        # Determine target power
        target_re = {
            "sph": row.re_sph or 0.0,
            "cyl": row.re_cyl or 0.0,
            "axis": row.re_axis or 180.0,
        }
        target_le = {
            "sph": row.le_sph or 0.0,
            "cyl": row.le_cyl or 0.0,
            "axis": row.le_axis or 180.0,
        }
        target_aux = STATE_OCCLUDER_MAP.get(state, "BINO")

        # Add ADD values for near phases
        target_add_r = row.add_r or 0.0
        target_add_l = row.add_l or 0.0

        # Set chart based on state
        chart_type = STATE_CHART_MAP.get(state, "")
        if chart_type == "snellen":
            chart_info = CHART_PARAM_MAP.get(str(row.chart_param))
            if chart_info:
                self._send_chart_command(chart_info["tab"], chart_info["chart_items"])
        elif chart_type == "jcc":
            self._send_chart_command("Chart1", ["chart_19"])
        elif chart_type == "duochrome":
            self._send_chart_command("Chart1", ["chart_17"])
        elif chart_type == "bino":
            self._send_chart_command("Chart1", ["chart_20"])
        elif chart_type == "near":
            self._send_chart_command("Chart5", ["chart_5"])

        # Send power + occluder with prev_state for accurate click calculation
        payload = {
            "test_cases": [{
                "case_id": 1,
                "prev_aux_lens": self._prev_aux_lens,
                "prev_right_eye": {**self._prev_re},
                "prev_left_eye": {**self._prev_le},
                "aux_lens": target_aux,
                "right_eye": {**target_re},
                "left_eye": {**target_le},
            }]
        }

        # Add ADD for near phases
        if state in ("P", "Q"):
            # Single-eye near ADD: include ADD for both eyes in the payload
            payload["test_cases"][0]["right_eye"]["add"] = target_add_r
            payload["test_cases"][0]["left_eye"]["add"] = target_add_l
            payload["test_cases"][0]["prev_right_eye"]["add"] = self._prev_add_r
            payload["test_cases"][0]["prev_left_eye"]["add"] = self._prev_add_l
        elif state == "R":
            # Binocular near ADD: only send right-eye ADD to avoid double-click.
            # The physical phoropter has one ADD dial that controls both eyes,
            # so including both R+L ADD causes two clicks for one intended change.
            payload["test_cases"][0]["right_eye"]["add"] = target_add_r
            payload["test_cases"][0]["prev_right_eye"]["add"] = self._prev_add_r
            # Intentionally omit left_eye.add and prev_left_eye.add

        self._post_to_phoropter(self.api_endpoint, payload)

        # Update prev state
        self._prev_re = target_re
        self._prev_le = target_le
        self._prev_aux_lens = target_aux
        self._prev_add_r = target_add_r
        self._prev_add_l = target_add_l

        # JCC mode management: power_axis_switch is a TOGGLE, not a SET.
        # The TOPCON phoropter defaults to AXIS mode when the JCC chart is
        # first displayed, so entering E/H from a non-JCC state requires
        # NO toggle. Only when transitioning E→F or H→I (axis→power) do
        # we need to send power_axis_switch.
        if state in ("E", "H"):
            # Axis mode needed. The phoropter defaults to axis when JCC
            # chart is shown, so only toggle if we were previously in
            # power mode (i.e. coming from F→E or I→H, which shouldn't
            # normally happen but handle defensively).
            if self._jcc_mode == "power":
                self._send_jcc_command("power_axis_switch")
            self._jcc_mode = "axis"
        elif state in ("F", "I"):
            # Power mode needed. Transition from E→F or H→I always requires
            # a toggle since we start in axis mode.
            if self._jcc_mode != "power":
                self._send_jcc_command("power_axis_switch")
            self._jcc_mode = "power"
        else:
            # Non-JCC state — reset tracking so next JCC entry starts fresh
            self._jcc_mode = None

    def _send_chart_command(self, tab: str, chart_items: list) -> None:
        """Send chart display command to phoropter."""
        payload = {"test_cases": [{"chart": {"tab": tab, "chart_items": chart_items}}]}
        self._post_to_phoropter(self.api_endpoint, payload)

    def _send_jcc_command(self, action: str) -> None:
        """Send JCC command to phoropter."""
        payload = {"test_cases": [{"jcc": action}]}
        self._post_to_phoropter(self.api_endpoint, payload)

    def _post_to_phoropter(self, url: str, payload: dict) -> None:
        """Send JSON POST to phoropter broker."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        print(f"[PHOROPTER] POST {url} payload={json.dumps(payload)}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8")
                print(f"[PHOROPTER] Response {resp.getcode()}: {body[:200]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            print(f"[PHOROPTER] HTTP Error {e.code}: {body[:200]}")
        except Exception as e:
            print(f"[PHOROPTER] Request failed: {e}")
