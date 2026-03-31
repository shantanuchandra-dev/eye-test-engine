from __future__ import annotations

from datetime import datetime
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
import re
import time
from typing import Optional

import pandas as pd

from fsm.config.calibration_loader import CalibrationLoader
from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.engines.refraction_fsm_engine import RefractionFSMEngine
from fsm.models.derived_variables import DerivedVariables
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription
from fsm.simulation.behavior_models import (
    AccommodativeResponder,
    HesitantResponder,
    IdealResponder,
    InconsistentResponder,
    NoisyResponder,
)
from fsm.simulation.common import (
    STATE_NAMES,
    SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG,
    SIM_DISTANCE_SUCCESS_CYL_TOL_D,
    SIM_DISTANCE_SUCCESS_SPH_TOL_D,
    execute_case,
)
from fsm.simulation.result_writer import create_run_folder, save_dataframe_csv, save_json, save_trace_csv
from fsm.simulation.virtual_patient import TruthRx

FLOAT_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

MIN_AGE = 3
MAX_AGE = 100
MAX_ABS_REQUIRED_SPH = 20.0
MAX_ABS_AR_SPH = 25.0
MAX_ABS_LENSO_SPH = 20.0
MAX_ABS_REQUIRED_CYL = 8.0
MAX_ABS_AR_CYL = 10.0
MAX_ABS_LENSO_CYL = 8.0
MAX_ADD = 4.0
REPLAY_OVERRIDE_VERSION = "real_csv_rule_based_v2__current_fsm_logic__no_dv_overrides"
DEFAULT_REAL_COHORT_BEHAVIOR_MODE = "patient_weighted"
DEFAULT_REAL_COHORT_REPLAYS_PER_PATIENT = 5
REAL_COHORT_BEHAVIOR_ASSIGNMENT_VERSION = "real_patient_behavior_weighted_v1"
REAL_COHORT_DETERMINISTIC_ASSIGNMENT_VERSION = "deterministic_virtual_patient_v1"
REAL_COHORT_FIXED_BEHAVIOR_ASSIGNMENT_VERSION_PREFIX = "fixed_real_patient_behavior"

BEHAVIOR_MODEL_CLASSES = {
    "ideal": IdealResponder,
    "accommodative": AccommodativeResponder,
    "noisy": NoisyResponder,
    "hesitant": HesitantResponder,
    "inconsistent": InconsistentResponder,
}
SUPPORTED_REAL_COHORT_BEHAVIOR_MODES = (
    "deterministic",
    "patient_weighted",
    "ideal",
    "accommodative",
    "noisy",
    "hesitant",
    "inconsistent",
)

BEHAVIOR_REPORT_ORDER = [
    "deterministic",
    "ideal",
    "accommodative",
    "noisy",
    "hesitant",
    "inconsistent",
]

BASE_BEHAVIOR_WEIGHTS = {
    "ideal": 0.45,
    "accommodative": 0.20,
    "noisy": 0.15,
    "hesitant": 0.15,
    "inconsistent": 0.05,
}


def format_duration_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    if total_seconds < 60:
        return f"{total_seconds:.2f}s"

    rounded_seconds = int(round(total_seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


@dataclass
class RealPatientCase:
    case_id: str
    qms_id: str
    created_at: str
    gender: str
    source_row_index: int
    patient: PatientInput
    dv: DerivedVariables
    truth: TruthRx
    truth_add_valid: bool
    history_inference_version: str


def axis_error(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    d = abs(float(a) - float(b)) % 180
    return min(d, 180 - d)


def _is_cardinal_axis(axis: Optional[float]) -> bool:
    if axis is None:
        return False
    return abs(float(axis) % 180.0) < 1e-9


def _abs_gap(value: Optional[float], truth: float) -> float:
    if value is None:
        return 0.0
    return abs(float(value) - float(truth))


def _parse_float(value) -> Optional[float]:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    compact = text.replace(" ", "")
    if compact.lower() in {"nan", "none", "null"}:
        return None

    if not FLOAT_PATTERN.match(compact):
        return None

    return float(compact)


def _parse_int(value) -> Optional[int]:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _round_quarter(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) / 0.25) * 0.25


def _normalize_axis(axis: Optional[float], cyl: Optional[float]) -> Optional[float]:
    if axis is None:
        if cyl is not None and abs(cyl) < 1e-9:
            return 0.0
        return None

    wrapped = round(float(axis)) % 180
    if wrapped < 0:
        wrapped += 180

    if wrapped == 0 and cyl is not None and abs(cyl) >= 1e-9:
        return 180.0
    return float(wrapped)


def _normalize_triplet(sph_raw, cyl_raw, axis_raw) -> tuple[Optional[float], Optional[float], Optional[float]]:
    sph = _round_quarter(_parse_float(sph_raw))
    cyl = _round_quarter(_parse_float(cyl_raw))
    axis = _normalize_axis(_parse_float(axis_raw), cyl)

    if cyl is not None and cyl > 0:
        if sph is not None:
            sph = _round_quarter(sph + cyl)
        cyl = _round_quarter(-cyl)
        axis = _normalize_axis((axis or 0.0) + 90.0, cyl)

    if cyl is not None and abs(cyl) < 1e-9:
        cyl = 0.0
        axis = 0.0 if axis is None else axis

    return sph, cyl, axis


def _normalize_add(value) -> Optional[float]:
    parsed = _round_quarter(_parse_float(value))
    if parsed is None:
        return None
    if parsed < 0 or parsed > MAX_ADD:
        return None
    return parsed


def _build_eye_rx(
    sph: Optional[float],
    cyl: Optional[float],
    axis: Optional[float],
) -> Optional[EyePrescription]:
    if sph is None and cyl is None and axis is None:
        return None
    return EyePrescription(sphere=sph, cylinder=cyl, axis=axis)


def _valid_required_eye(
    rx: Optional[EyePrescription],
    *,
    max_abs_sph: float,
    max_abs_cyl: float,
) -> bool:
    if rx is None or not rx.has_full_rx():
        return False
    if abs(float(rx.sphere)) > max_abs_sph:
        return False
    if float(rx.cylinder) > 1e-9 or abs(float(rx.cylinder)) > max_abs_cyl:
        return False
    if float(rx.axis) < 0 or float(rx.axis) > 180:
        return False
    return True


def _sanitize_optional_eye(
    sph: Optional[float],
    cyl: Optional[float],
    axis: Optional[float],
    *,
    max_abs_sph: float,
    max_abs_cyl: float,
) -> Optional[EyePrescription]:
    rx = _build_eye_rx(sph, cyl, axis)
    if rx is None:
        return None
    if not _valid_required_eye(rx, max_abs_sph=max_abs_sph, max_abs_cyl=max_abs_cyl):
        return None
    return rx


def _has_material_distance_correction(truth: TruthRx) -> bool:
    return any(
        abs(value) >= 0.75
        for value in (
            truth.re_sph,
            truth.le_sph,
            truth.re_cyl,
            truth.le_cyl,
        )
    )


def _has_presbyopic_add(age: int, add_r: float, add_l: float, lenso_add_r: Optional[float], lenso_add_l: Optional[float]) -> bool:
    return (
        max(add_r, add_l, lenso_add_r or 0.0, lenso_add_l or 0.0) >= 0.75
        or age >= 50
    )


def _rx_is_close(
    left: Optional[EyePrescription],
    right: Optional[EyePrescription],
    *,
    truth_left: EyePrescription,
    truth_right: EyePrescription,
    sph_tol: float = SIM_DISTANCE_SUCCESS_SPH_TOL_D,
    cyl_tol: float = SIM_DISTANCE_SUCCESS_CYL_TOL_D,
    axis_tol: float = SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG,
) -> bool:
    if left is None or right is None or not left.has_full_rx() or not right.has_full_rx():
        return False

    return (
        abs(float(right.sphere) - float(truth_right.sphere)) <= sph_tol
        and abs(float(right.cylinder) - float(truth_right.cylinder)) <= cyl_tol
        and axis_error(right.axis, truth_right.axis) <= axis_tol
        and abs(float(left.sphere) - float(truth_left.sphere)) <= sph_tol
        and abs(float(left.cylinder) - float(truth_left.cylinder)) <= cyl_tol
        and axis_error(left.axis, truth_left.axis) <= axis_tol
    )


def synthesize_patient_input(
    *,
    qms_id: str,
    age: int,
    ar_re: EyePrescription,
    ar_le: EyePrescription,
    lenso_re: Optional[EyePrescription],
    lenso_le: Optional[EyePrescription],
    lenso_add_r: Optional[float],
    lenso_add_l: Optional[float],
    truth_re: EyePrescription,
    truth_le: EyePrescription,
    truth_add_r: float,
    truth_add_l: float,
) -> PatientInput:
    truth = TruthRx(
        re_sph=float(truth_re.sphere),
        re_cyl=float(truth_re.cylinder),
        re_axis=float(truth_re.axis),
        le_sph=float(truth_le.sphere),
        le_cyl=float(truth_le.cylinder),
        le_axis=float(truth_le.axis),
        add_r=float(truth_add_r),
        add_l=float(truth_add_l),
    )

    has_distance_correction = _has_material_distance_correction(truth)
    has_presbyopic_add = _has_presbyopic_add(
        age,
        truth_add_r,
        truth_add_l,
        lenso_add_r,
        lenso_add_l,
    )
    lenso_close_to_truth = _rx_is_close(
        lenso_le,
        lenso_re,
        truth_left=truth_le,
        truth_right=truth_re,
    )

    if age < 18:
        screen_time_hours = 4.0
    elif age <= 35:
        screen_time_hours = 8.0
    elif age <= 50:
        screen_time_hours = 6.0 if has_presbyopic_add else 7.0
    else:
        screen_time_hours = 4.0

    if has_distance_correction and 23 <= age <= 60 and not has_presbyopic_add:
        driving_hours = 2.0
    elif 23 <= age <= 65:
        driving_hours = 1.0
    else:
        driving_hours = 0.25

    if has_presbyopic_add and not has_distance_correction:
        primary_reason = "Blurred near"
    elif has_distance_correction:
        primary_reason = "Blurred distance"
    else:
        primary_reason = "Routine check"

    symptoms = []
    if primary_reason == "Blurred distance":
        symptoms.append("Blurred distance")
    if has_presbyopic_add:
        symptoms.append("Blurred near")
    if screen_time_hours >= 6.0 and age >= 18:
        symptoms.append("Eye strain")
    symptoms_text = ", ".join(dict.fromkeys(symptoms))

    if lenso_re is None and lenso_le is None:
        satisfaction = "Not satisfied" if (has_distance_correction or has_presbyopic_add) else "Satisfied"
    else:
        satisfaction = "Satisfied" if lenso_close_to_truth else "Not satisfied"

    if has_presbyopic_add:
        wear_type = "Progressive"
    else:
        wear_type = "Single Vision"

    if has_presbyopic_add and age >= 45:
        priority = "Comfort-first"
    elif has_distance_correction:
        priority = "Distance-first"
    else:
        priority = "Balanced"

    if has_presbyopic_add and age >= 42:
        near_priority = "High"
    elif screen_time_hours >= 7.0:
        near_priority = "Medium"
    else:
        near_priority = "Low"

    add_gap = max(
        abs((lenso_add_r or 0.0) - truth_add_r),
        abs((lenso_add_l or 0.0) - truth_add_l),
    )
    rx_change_was_large = (
        not lenso_close_to_truth
        and (
            add_gap > 0.75
            or abs(float(ar_re.sphere) - float(truth_re.sphere)) > 1.0
            or abs(float(ar_le.sphere) - float(truth_le.sphere)) > 1.0
            or abs(float(ar_re.cylinder) - float(truth_re.cylinder)) > 0.75
            or abs(float(ar_le.cylinder) - float(truth_le.cylinder)) > 0.75
        )
    )

    if lenso_re is None and lenso_le is None:
        last_eye_test_months_ago = 24.0
    elif satisfaction == "Satisfied":
        last_eye_test_months_ago = 12.0
    elif rx_change_was_large:
        last_eye_test_months_ago = 30.0
    else:
        last_eye_test_months_ago = 18.0

    return PatientInput(
        visit_id=qms_id,
        age=age,
        # Live intake no longer captures occupation, so cohort replay should
        # not synthesize it into DV branching.
        occupation="",
        screen_time_hours=screen_time_hours,
        driving_hours=driving_hours,
        primary_reason=primary_reason,
        symptoms_text=symptoms_text,
        satisfaction_with_current_rx=satisfaction,
        wear_type=wear_type,
        distance_target_preference="",
        priority=priority,
        near_priority_declared=near_priority,
        last_eye_test_months_ago=last_eye_test_months_ago,
        rx_change_was_large=rx_change_was_large,
        fluctuating_vision_reported=False,
        diabetes=False,
        prior_eye_surgery="None",
        keratoconus=False,
        amblyopia=False,
        infection=False,
        optom_review_flag=False,
        autorefractor_re=ar_re,
        autorefractor_le=ar_le,
        lenso_re=lenso_re,
        lenso_le=lenso_le,
        lenso_add_r=lenso_add_r,
        lenso_add_l=lenso_add_l,
        truth_re=truth_re,
        truth_le=truth_le,
    )


def apply_replay_dv_overrides(dv: DerivedVariables) -> DerivedVariables:
    """Return the current ETE_v2 DV set without replay-only behavioral overrides."""
    return replace(dv)


def _normalize_behavior_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {
        behavior_id: max(0.01, float(weight))
        for behavior_id, weight in weights.items()
    }
    total = sum(clipped.values())
    return {
        behavior_id: float(weight / total)
        for behavior_id, weight in clipped.items()
    }


def derive_patient_conditioned_behavior_weights(case: RealPatientCase) -> dict[str, float]:
    """
    Build replay-only response-behavior weights from known patient context.

    This intentionally uses age, measured inputs, derived variables, and the
    recorded final Rx. Gender is retained in the case metadata for reporting
    but is not used as a direct behavior prior.
    """
    weights = dict(BASE_BEHAVIOR_WEIGHTS)

    age = int(case.patient.age or 0)

    if age < 18:
        weights["accommodative"] += 0.20
        weights["hesitant"] += 0.05
        weights["ideal"] -= 0.15
    elif age <= 25:
        weights["accommodative"] += 0.10
        weights["ideal"] -= 0.06
    elif age >= 45:
        weights["ideal"] += 0.08
        weights["accommodative"] -= 0.08

    if case.dv.dv_fogging_required and age < 35:
        weights["accommodative"] += 0.08

    if case.dv.dv_stability_level != "Stable":
        weights["hesitant"] += 0.06
        weights["inconsistent"] += 0.08
        weights["ideal"] -= 0.08

    if case.dv.dv_symptom_risk_level in {"Moderate", "High"}:
        weights["hesitant"] += 0.05
        weights["noisy"] += 0.03

    if case.dv.dv_medical_risk_level in {"Moderate", "High"}:
        weights["inconsistent"] += 0.05
        weights["ideal"] -= 0.04

    mismatch_levels = {
        case.dv.dv_ar_lenso_mismatch_level_RE,
        case.dv.dv_ar_lenso_mismatch_level_LE,
    }
    if "Large" in mismatch_levels:
        weights["inconsistent"] += 0.10
        weights["noisy"] += 0.06
        weights["ideal"] -= 0.10
    elif "Medium" in mismatch_levels:
        weights["hesitant"] += 0.05
        weights["noisy"] += 0.04
        weights["ideal"] -= 0.05

    ar_sph_gap = max(
        _abs_gap(case.patient.autorefractor_re.sphere if case.patient.autorefractor_re else None, case.truth.re_sph),
        _abs_gap(case.patient.autorefractor_le.sphere if case.patient.autorefractor_le else None, case.truth.le_sph),
    )
    ar_cyl_gap = max(
        _abs_gap(case.patient.autorefractor_re.cylinder if case.patient.autorefractor_re else None, case.truth.re_cyl),
        _abs_gap(case.patient.autorefractor_le.cylinder if case.patient.autorefractor_le else None, case.truth.le_cyl),
    )
    ar_axis_gap = max(
        axis_error(case.patient.autorefractor_re.axis if case.patient.autorefractor_re else None, case.truth.re_axis),
        axis_error(case.patient.autorefractor_le.axis if case.patient.autorefractor_le else None, case.truth.le_axis),
    )
    start_sph_gap = max(
        _abs_gap(case.dv.dv_start_rx_RE_sph, case.truth.re_sph),
        _abs_gap(case.dv.dv_start_rx_LE_sph, case.truth.le_sph),
    )
    start_axis_gap = max(
        axis_error(case.dv.dv_start_rx_RE_axis, case.truth.re_axis),
        axis_error(case.dv.dv_start_rx_LE_axis, case.truth.le_axis),
    )
    max_truth_cyl = max(abs(float(case.truth.re_cyl)), abs(float(case.truth.le_cyl)))

    if max(ar_sph_gap, ar_cyl_gap, start_sph_gap) >= 1.0:
        weights["hesitant"] += 0.08
        weights["inconsistent"] += 0.05
        weights["ideal"] -= 0.08
    elif max(ar_sph_gap, ar_cyl_gap, start_sph_gap) >= 0.5:
        weights["hesitant"] += 0.04
        weights["noisy"] += 0.04

    if max(ar_axis_gap, start_axis_gap) >= 25:
        weights["hesitant"] += 0.12
        weights["inconsistent"] += 0.06
        weights["ideal"] -= 0.08
    elif max(ar_axis_gap, start_axis_gap) >= 10:
        weights["hesitant"] += 0.05

    if max_truth_cyl >= 2.0:
        weights["hesitant"] += 0.06

    if case.dv.dv_start_source_policy == "Start_AR":
        weights["noisy"] += 0.03
        if _is_cardinal_axis(case.dv.dv_start_rx_RE_axis) or _is_cardinal_axis(case.dv.dv_start_rx_LE_axis):
            weights["hesitant"] += 0.10
            weights["inconsistent"] += 0.04

    lenso_close_to_truth = _rx_is_close(
        case.patient.lenso_le,
        case.patient.lenso_re,
        truth_left=EyePrescription(case.truth.le_sph, case.truth.le_cyl, case.truth.le_axis),
        truth_right=EyePrescription(case.truth.re_sph, case.truth.re_cyl, case.truth.re_axis),
    )

    easy_case = (
        25 <= age < 45
        and case.dv.dv_stability_level == "Stable"
        and case.dv.dv_symptom_risk_level == "None"
        and case.dv.dv_medical_risk_level == "None"
        and "Large" not in mismatch_levels
        and ar_sph_gap <= 0.5
        and ar_axis_gap <= 10
        and max_truth_cyl < 1.5
    )

    if easy_case or lenso_close_to_truth:
        weights["ideal"] += 0.12
        weights["noisy"] -= 0.04
        weights["hesitant"] -= 0.04

    return _normalize_behavior_weights(weights)


def _behavior_weight_columns(weight_map: dict[str, float]) -> dict[str, float]:
    return {
        f"behavior_weight_{behavior_id}": float(weight_map.get(behavior_id, 0.0))
        for behavior_id in BASE_BEHAVIOR_WEIGHTS
    }


def _build_trace_dataframe(trace_rows: list[dict], case: RealPatientCase, result: dict) -> pd.DataFrame:
    df = pd.DataFrame(trace_rows)
    if df.empty:
        return df

    df["case_id"] = case.case_id
    df["qms_id"] = case.qms_id
    df["created_at"] = case.created_at
    df["gender"] = case.gender
    df["history_inference_version"] = case.history_inference_version

    for key, value in asdict(case.dv).items():
        df[key] = value

    df["truth_re_sph"] = case.truth.re_sph
    df["truth_re_cyl"] = case.truth.re_cyl
    df["truth_re_axis"] = case.truth.re_axis
    df["truth_le_sph"] = case.truth.le_sph
    df["truth_le_cyl"] = case.truth.le_cyl
    df["truth_le_axis"] = case.truth.le_axis
    df["truth_add_r"] = case.truth.add_r
    df["truth_add_l"] = case.truth.add_l

    for key in (
        "behavior_mode",
        "behavior_assignment_version",
        "behavior_id",
        "replay_index",
        "steps",
        "outcome",
        "termination_state",
        "failure_state",
        "distance_success_within_tolerance",
        "add_success_within_tolerance",
        "final_re_sph",
        "final_re_cyl",
        "final_re_axis",
        "final_le_sph",
        "final_le_cyl",
        "final_le_axis",
        "final_add_r",
        "final_add_l",
        "re_sph_err",
        "re_cyl_err",
        "re_axis_err",
        "le_sph_err",
        "le_cyl_err",
        "le_axis_err",
        "add_r_err",
        "add_l_err",
    ):
        df[key] = result.get(key)

    return df


def _dominant_label(values: pd.Series, default: str = "") -> str:
    if len(values) == 0:
        return default
    counts = values.value_counts()
    if len(counts) == 0:
        return default
    return str(counts.index[0])


def _aggregate_case_trials(trial_rows: list[dict]) -> dict:
    if not trial_rows:
        raise RuntimeError("Cannot aggregate an empty replay list")

    trial_df = pd.DataFrame(trial_rows)
    completed_df = trial_df[trial_df["outcome"] == "SUCCESS"]
    failure_df = trial_df[trial_df["outcome"] != "SUCCESS"]

    distance_metrics = compute_distance_accuracy_metrics(completed_df)
    add_metrics = compute_add_accuracy_metrics(completed_df)

    row = dict(trial_rows[0])
    row.pop("replay_index", None)
    row.pop("behavior_id", None)

    total_replays = int(len(trial_df))
    completed_replays = int((trial_df["outcome"] == "SUCCESS").sum())
    escalated_replays = int((trial_df["outcome"] == "ESCALATE").sum())
    timed_out_replays = int((trial_df["outcome"] == "TIMEOUT").sum())

    outcome_counts = trial_df["outcome"].value_counts()
    dominant_outcome = str(outcome_counts.index[0]) if len(outcome_counts) else "TIMEOUT"

    row.update(
        {
            "behavior_mode": str(trial_rows[0]["behavior_mode"]),
            "behavior_assignment_version": str(trial_rows[0]["behavior_assignment_version"]),
            "replays_per_patient": total_replays,
            "completed_replays": completed_replays,
            "escalated_replays": escalated_replays,
            "timed_out_replays": timed_out_replays,
            "completion_probability": float(completed_replays / total_replays),
            "escalation_probability": float(escalated_replays / total_replays),
            "timeout_probability": float(timed_out_replays / total_replays),
            "distance_success_probability": float(trial_df["distance_success_within_tolerance"].mean()),
            "distance_success_within_tolerance": float(trial_df["distance_success_within_tolerance"].mean()),
            "add_success_probability": (
                float(trial_df["add_success_within_tolerance"].dropna().mean())
                if trial_df["add_success_within_tolerance"].dropna().shape[0]
                else None
            ),
            "accuracy_among_completed": float(distance_metrics["accuracy_among_completed"]),
            "RE_sphere_within_0.25": float(distance_metrics["RE_sphere_within_0.25"]),
            "RE_cyl_within_0.25": float(distance_metrics["RE_cyl_within_0.25"]),
            "RE_axis_within_10deg": float(distance_metrics["RE_axis_within_10deg"]),
            "LE_sphere_within_0.25": float(distance_metrics["LE_sphere_within_0.25"]),
            "LE_cyl_within_0.25": float(distance_metrics["LE_cyl_within_0.25"]),
            "LE_axis_within_10deg": float(distance_metrics["LE_axis_within_10deg"]),
            "add_accuracy_among_completed_valid_add": float(add_metrics["add_accuracy_among_completed_valid_add"]),
            "outcome": dominant_outcome,
            "termination_state": (
                "END"
                if dominant_outcome == "SUCCESS"
                else ("TIMEOUT" if dominant_outcome == "TIMEOUT" else "ESCALATE")
            ),
            "failure_state": _dominant_label(failure_df["failure_state"], default=""),
            "dominant_behavior_id": _dominant_label(trial_df["behavior_id"], default="deterministic"),
            "selected_prescribed_rx_source": _dominant_label(trial_df["selected_prescribed_rx_source"], default=""),
            "accepted_achieved_probability": float((trial_df["patient_accepted_achieved_over_current_rx"] == "Yes").mean()),
            "accepted_pgp_probability": float((trial_df["patient_accepted_achieved_over_current_rx"] == "No").mean()),
            "steps": float(trial_df["steps"].mean()),
            "steps_min": int(trial_df["steps"].min()),
            "steps_max": int(trial_df["steps"].max()),
        }
    )

    for behavior_id in BEHAVIOR_REPORT_ORDER:
        count = int((trial_df["behavior_id"] == behavior_id).sum())
        row[f"behavior_{behavior_id}_replays"] = count
        row[f"behavior_{behavior_id}_probability"] = float(count / total_replays)

    numeric_mean_columns = [
        "final_re_sph",
        "final_re_cyl",
        "final_re_axis",
        "final_le_sph",
        "final_le_cyl",
        "final_le_axis",
        "final_add_r",
        "final_add_l",
        "re_sph_err",
        "re_cyl_err",
        "re_axis_err",
        "le_sph_err",
        "le_cyl_err",
        "le_axis_err",
        "add_r_err",
        "add_l_err",
    ]
    for column in numeric_mean_columns:
        if column in trial_df.columns and trial_df[column].notna().any():
            row[column] = float(trial_df[column].mean())
        else:
            row[column] = None

    return row


def load_real_patient_cases(
    csv_path: str | Path,
    calibration: CalibrationLoader,
    limit: Optional[int] = None,
) -> tuple[list[RealPatientCase], pd.DataFrame]:
    df = pd.read_csv(csv_path, low_memory=False)
    dv_engine = DerivedVariablesEngine(calibration)

    cases: list[RealPatientCase] = []
    invalid_rows: list[dict] = []

    for row_index, row in df.iterrows():
        reasons: list[str] = []

        age = _parse_int(row.get("step2_age"))
        if age is None or age < MIN_AGE or age > MAX_AGE:
            reasons.append("invalid_age")

        ar_le_vals = _normalize_triplet(row.get("ar_left_sph"), row.get("ar_left_cyl"), row.get("ar_left_axis"))
        ar_re_vals = _normalize_triplet(row.get("ar_right_sph"), row.get("ar_right_cyl"), row.get("ar_right_axis"))
        lenso_le_vals = _normalize_triplet(row.get("lenso_left_sph"), row.get("lenso_left_cyl"), row.get("lenso_left_axis"))
        lenso_re_vals = _normalize_triplet(row.get("lenso_right_sph"), row.get("lenso_right_cyl"), row.get("lenso_right_axis"))
        truth_re_vals = _normalize_triplet(row.get("step12_sph_right"), row.get("step12_cyl_right"), row.get("step12_ax_right"))
        truth_le_vals = _normalize_triplet(row.get("step12_sph_left"), row.get("step12_cyl_left"), row.get("step12_ax_left"))

        ar_le = _build_eye_rx(*ar_le_vals)
        ar_re = _build_eye_rx(*ar_re_vals)
        truth_re = _build_eye_rx(*truth_re_vals)
        truth_le = _build_eye_rx(*truth_le_vals)

        if not _valid_required_eye(ar_re, max_abs_sph=MAX_ABS_AR_SPH, max_abs_cyl=MAX_ABS_AR_CYL):
            reasons.append("invalid_ar_right")
        if not _valid_required_eye(ar_le, max_abs_sph=MAX_ABS_AR_SPH, max_abs_cyl=MAX_ABS_AR_CYL):
            reasons.append("invalid_ar_left")
        if not _valid_required_eye(truth_re, max_abs_sph=MAX_ABS_REQUIRED_SPH, max_abs_cyl=MAX_ABS_REQUIRED_CYL):
            reasons.append("invalid_step12_right")
        if not _valid_required_eye(truth_le, max_abs_sph=MAX_ABS_REQUIRED_SPH, max_abs_cyl=MAX_ABS_REQUIRED_CYL):
            reasons.append("invalid_step12_left")

        if reasons:
            invalid_rows.append(
                {
                    "source_row_index": row_index,
                    "qms_id": str(row.get("qms_id", "")).strip(),
                    "created_at": str(row.get("created_at", "")).strip(),
                    "gender": str(row.get("gender", "")).strip(),
                    "reasons": "|".join(reasons),
                }
            )
            continue

        lenso_le = _sanitize_optional_eye(
            *lenso_le_vals,
            max_abs_sph=MAX_ABS_LENSO_SPH,
            max_abs_cyl=MAX_ABS_LENSO_CYL,
        )
        lenso_re = _sanitize_optional_eye(
            *lenso_re_vals,
            max_abs_sph=MAX_ABS_LENSO_SPH,
            max_abs_cyl=MAX_ABS_LENSO_CYL,
        )

        truth_add_r_parsed = _normalize_add(row.get("step12_na_right"))
        truth_add_l_parsed = _normalize_add(row.get("step12_na_left"))
        lenso_add_r = _normalize_add(row.get("lenso_right_ap"))
        lenso_add_l = _normalize_add(row.get("lenso_left_ap"))

        truth_add_valid = truth_add_r_parsed is not None and truth_add_l_parsed is not None
        truth_add_r = truth_add_r_parsed or 0.0
        truth_add_l = truth_add_l_parsed or 0.0

        patient = synthesize_patient_input(
            qms_id=str(row["qms_id"]).strip(),
            age=int(age),
            ar_re=ar_re,
            ar_le=ar_le,
            lenso_re=lenso_re,
            lenso_le=lenso_le,
            lenso_add_r=lenso_add_r,
            lenso_add_l=lenso_add_l,
            truth_re=truth_re,
            truth_le=truth_le,
            truth_add_r=truth_add_r,
            truth_add_l=truth_add_l,
        )

        case_id = f"REAL_{str(row['qms_id']).strip()}"
        patient.visit_id = case_id
        dv = apply_replay_dv_overrides(dv_engine.derive(patient))

        case = RealPatientCase(
            case_id=case_id,
            qms_id=str(row["qms_id"]).strip(),
            created_at=str(row.get("created_at", "")).strip(),
            gender=str(row.get("gender", "")).strip().lower(),
            source_row_index=row_index,
            patient=patient,
            dv=dv,
            truth=TruthRx(
                re_sph=float(truth_re.sphere),
                re_cyl=float(truth_re.cylinder),
                re_axis=float(truth_re.axis),
                le_sph=float(truth_le.sphere),
                le_cyl=float(truth_le.cylinder),
                le_axis=float(truth_le.axis),
                add_r=float(truth_add_r),
                add_l=float(truth_add_l),
            ),
            truth_add_valid=truth_add_valid,
            history_inference_version=REPLAY_OVERRIDE_VERSION,
        )

        cases.append(case)

        if limit is not None and len(cases) >= limit:
            break

    invalid_df = pd.DataFrame(
        invalid_rows,
        columns=["source_row_index", "qms_id", "created_at", "gender", "reasons"],
    )
    return cases, invalid_df


def select_real_patient_cases(
    cases: list[RealPatientCase],
    *,
    qms_id: Optional[str] = None,
    near_test_required: Optional[bool] = None,
    limit: Optional[int] = None,
) -> list[RealPatientCase]:
    selected = cases

    if qms_id is not None:
        target_qms_id = str(qms_id).strip()
        selected = [case for case in selected if case.qms_id == target_qms_id]

    if near_test_required is not None:
        selected = [
            case
            for case in selected
            if bool(case.dv.dv_near_test_required) == bool(near_test_required)
        ]

    if limit is not None:
        selected = selected[:limit]

    return selected


class RealPatientCohortRunner:
    def __init__(
        self,
        calibration: CalibrationLoader,
        *,
        behavior_mode: str = DEFAULT_REAL_COHORT_BEHAVIOR_MODE,
        replays_per_patient: int = DEFAULT_REAL_COHORT_REPLAYS_PER_PATIENT,
        seed_base: int = 20260318,
    ):
        self.calibration = calibration
        self.engine = RefractionFSMEngine(calibration)
        self.behavior_mode = str(behavior_mode).strip()
        self.replays_per_patient = max(1, int(replays_per_patient))
        self.seed_base = int(seed_base)

        if self.behavior_mode not in SUPPORTED_REAL_COHORT_BEHAVIOR_MODES:
            raise ValueError(f"Unsupported behavior_mode: {self.behavior_mode}")

    def _behavior_assignment_version(self) -> str:
        if self.behavior_mode == "deterministic":
            return REAL_COHORT_DETERMINISTIC_ASSIGNMENT_VERSION
        if self.behavior_mode in BEHAVIOR_MODEL_CLASSES:
            return f"{REAL_COHORT_FIXED_BEHAVIOR_ASSIGNMENT_VERSION_PREFIX}_{self.behavior_mode}_v1"
        return REAL_COHORT_BEHAVIOR_ASSIGNMENT_VERSION

    def _replay_seed(self, case: RealPatientCase, replay_index: int) -> int:
        return (
            self.seed_base
            + (case.source_row_index + 1) * 1009
            + replay_index * 37
        )

    def _select_behavior_for_replay(
        self,
        case: RealPatientCase,
        replay_index: int,
    ) -> tuple[Optional[object], str, dict[str, float]]:
        if self.behavior_mode == "deterministic":
            weight_map = {
                "ideal": 1.0,
                "accommodative": 0.0,
                "noisy": 0.0,
                "hesitant": 0.0,
                "inconsistent": 0.0,
            }
            return None, "deterministic", weight_map

        replay_seed = self._replay_seed(case, replay_index)

        if self.behavior_mode in BEHAVIOR_MODEL_CLASSES:
            behavior_id = self.behavior_mode
            weight_map = {
                key: (1.0 if key == behavior_id else 0.0)
                for key in BEHAVIOR_MODEL_CLASSES
            }
            behavior_cls = BEHAVIOR_MODEL_CLASSES[behavior_id]
            behavior_model = behavior_cls(seed=replay_seed + 17, weight=1.0)
            return behavior_model, behavior_id, weight_map

        weight_map = derive_patient_conditioned_behavior_weights(case)

        chooser = random.Random(replay_seed)
        behavior_id = chooser.choices(
            list(weight_map.keys()),
            weights=[weight_map[key] for key in weight_map],
            k=1,
        )[0]

        behavior_cls = BEHAVIOR_MODEL_CLASSES[behavior_id]
        behavior_model = behavior_cls(seed=replay_seed + 17, weight=weight_map[behavior_id])
        return behavior_model, behavior_id, weight_map

    def _run_trial(
        self,
        case: RealPatientCase,
        *,
        max_steps: int = 200,
        replay_index: int = 0,
        collect_trace: bool = False,
    ) -> tuple[dict, Optional[pd.DataFrame]]:
        behavior_model, behavior_id, behavior_weights = self._select_behavior_for_replay(
            case,
            replay_index,
        )
        trace_meta = {
            "test_id": case.case_id,
            "case_id": case.case_id,
            "qms_id": case.qms_id,
            "behavior_mode": self.behavior_mode,
            "behavior_id": behavior_id,
            "replay_index": replay_index,
        }
        result, trace_df = execute_case(
            engine=self.engine,
            case=case,
            behavior_model=behavior_model,
            max_steps=max_steps,
            collect_trace=collect_trace,
            trace_metadata=trace_meta,
        )

        patient_fields = asdict(case.patient)
        dv_fields = asdict(case.dv)

        result.update({
            "case_id": case.case_id,
            "qms_id": case.qms_id,
            "source_row_index": case.source_row_index,
            "created_at": case.created_at,
            "gender": case.gender,
            "history_inference_version": case.history_inference_version,
            "behavior_mode": self.behavior_mode,
            "behavior_assignment_version": self._behavior_assignment_version(),
            "behavior_id": behavior_id,
            "replay_index": replay_index,
            "truth_add_valid": case.truth_add_valid,
            "truth_re_sph": case.truth.re_sph,
            "truth_re_cyl": case.truth.re_cyl,
            "truth_re_axis": case.truth.re_axis,
            "truth_le_sph": case.truth.le_sph,
            "truth_le_cyl": case.truth.le_cyl,
            "truth_le_axis": case.truth.le_axis,
            "truth_add_r": case.truth.add_r,
            "truth_add_l": case.truth.add_l,
        })

        input_projection = {
            f"input_{k}": v
            for k, v in patient_fields.items()
            if k not in {"autorefractor_re", "autorefractor_le", "lenso_re", "lenso_le", "truth_re", "truth_le"}
        }
        result.update(input_projection)
        result.update(dv_fields)

        result.update(
            {
                "ar_re_sph": case.patient.autorefractor_re.sphere if case.patient.autorefractor_re else None,
                "ar_re_cyl": case.patient.autorefractor_re.cylinder if case.patient.autorefractor_re else None,
                "ar_re_axis": case.patient.autorefractor_re.axis if case.patient.autorefractor_re else None,
                "ar_le_sph": case.patient.autorefractor_le.sphere if case.patient.autorefractor_le else None,
                "ar_le_cyl": case.patient.autorefractor_le.cylinder if case.patient.autorefractor_le else None,
                "ar_le_axis": case.patient.autorefractor_le.axis if case.patient.autorefractor_le else None,
                "lenso_re_sph": case.patient.lenso_re.sphere if case.patient.lenso_re else None,
                "lenso_re_cyl": case.patient.lenso_re.cylinder if case.patient.lenso_re else None,
                "lenso_re_axis": case.patient.lenso_re.axis if case.patient.lenso_re else None,
                "lenso_le_sph": case.patient.lenso_le.sphere if case.patient.lenso_le else None,
                "lenso_le_cyl": case.patient.lenso_le.cylinder if case.patient.lenso_le else None,
                "lenso_le_axis": case.patient.lenso_le.axis if case.patient.lenso_le else None,
            }
        )
        result.update(_behavior_weight_columns(behavior_weights))

        if collect_trace:
            trace_df = _build_trace_dataframe(
                [] if trace_df is None else trace_df.to_dict(orient="records"),
                case,
                result,
            )
        return result, trace_df

    def run_one_trial(
        self,
        case: RealPatientCase,
        *,
        max_steps: int = 200,
        replay_index: int = 0,
    ) -> dict:
        result, _ = self._run_trial(
            case,
            max_steps=max_steps,
            replay_index=replay_index,
            collect_trace=False,
        )
        return result

    def run_one_trial_with_trace(
        self,
        case: RealPatientCase,
        *,
        max_steps: int = 200,
        replay_index: int = 0,
    ) -> tuple[dict, pd.DataFrame]:
        result, trace_df = self._run_trial(
            case,
            max_steps=max_steps,
            replay_index=replay_index,
            collect_trace=True,
        )
        return result, trace_df if trace_df is not None else pd.DataFrame()

    def run_case_trials(self, case: RealPatientCase, max_steps: int = 200) -> list[dict]:
        return [
            self.run_one_trial(case, max_steps=max_steps, replay_index=replay_index)
            for replay_index in range(self.replays_per_patient)
        ]

    def run_one(self, case: RealPatientCase, max_steps: int = 200) -> dict:
        return _aggregate_case_trials(self.run_case_trials(case, max_steps=max_steps))

    def run_cases(self, cases: list[RealPatientCase], max_steps: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
        aggregate_rows = []
        trial_rows = []
        for index, case in enumerate(cases, start=1):
            case_trials = self.run_case_trials(case, max_steps=max_steps)
            trial_rows.extend(case_trials)
            aggregate_rows.append(_aggregate_case_trials(case_trials))
            if index % 500 == 0:
                print(
                    f"Completed {index} cohort cases "
                    f"({index * self.replays_per_patient} replay simulations)"
                )
        return pd.DataFrame(aggregate_rows), pd.DataFrame(trial_rows)


def compute_distance_accuracy_metrics(completed_df: pd.DataFrame) -> dict:
    if len(completed_df) == 0:
        return {
            "accuracy_among_completed": 0.0,
            "RE_sphere_within_0.25": 0.0,
            "RE_cyl_within_0.25": 0.0,
            "RE_axis_within_10deg": 0.0,
            "LE_sphere_within_0.25": 0.0,
            "LE_cyl_within_0.25": 0.0,
            "LE_axis_within_10deg": 0.0,
            "mean_re_sph_err": 0.0,
            "mean_re_cyl_err": 0.0,
            "mean_re_axis_err": 0.0,
            "mean_le_sph_err": 0.0,
            "mean_le_cyl_err": 0.0,
            "mean_le_axis_err": 0.0,
        }

    re_sphere_025 = (completed_df["re_sph_err"] <= SIM_DISTANCE_SUCCESS_SPH_TOL_D).mean()
    re_cyl_025 = (completed_df["re_cyl_err"] <= SIM_DISTANCE_SUCCESS_CYL_TOL_D).mean()
    re_axis_10 = (completed_df["re_axis_err"] <= SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG).mean()
    le_sphere_025 = (completed_df["le_sph_err"] <= SIM_DISTANCE_SUCCESS_SPH_TOL_D).mean()
    le_cyl_025 = (completed_df["le_cyl_err"] <= SIM_DISTANCE_SUCCESS_CYL_TOL_D).mean()
    le_axis_10 = (completed_df["le_axis_err"] <= SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG).mean()

    cumulative_avg_accuracy = (
        re_sphere_025
        + re_cyl_025
        + re_axis_10
        + le_sphere_025
        + le_cyl_025
        + le_axis_10
    ) / 6.0

    return {
        "accuracy_among_completed": cumulative_avg_accuracy,
        "RE_sphere_within_0.25": re_sphere_025,
        "RE_cyl_within_0.25": re_cyl_025,
        "RE_axis_within_10deg": re_axis_10,
        "LE_sphere_within_0.25": le_sphere_025,
        "LE_cyl_within_0.25": le_cyl_025,
        "LE_axis_within_10deg": le_axis_10,
        "mean_re_sph_err": completed_df["re_sph_err"].mean(),
        "mean_re_cyl_err": completed_df["re_cyl_err"].mean(),
        "mean_re_axis_err": completed_df["re_axis_err"].mean(),
        "mean_le_sph_err": completed_df["le_sph_err"].mean(),
        "mean_le_cyl_err": completed_df["le_cyl_err"].mean(),
        "mean_le_axis_err": completed_df["le_axis_err"].mean(),
    }


def compute_add_accuracy_metrics(completed_df: pd.DataFrame) -> dict:
    add_df = completed_df[completed_df["truth_add_valid"] == True]
    if len(add_df) == 0:
        return {
            "valid_add_truth_cases": 0,
            "add_accuracy_among_completed_valid_add": 0.0,
            "add_r_within_0.25": 0.0,
            "add_l_within_0.25": 0.0,
        }

    add_r_025 = (add_df["add_r_err"] <= 0.25).mean()
    add_l_025 = (add_df["add_l_err"] <= 0.25).mean()

    return {
        "valid_add_truth_cases": int(len(add_df)),
        "add_accuracy_among_completed_valid_add": float((add_r_025 + add_l_025) / 2.0),
        "add_r_within_0.25": float(add_r_025),
        "add_l_within_0.25": float(add_l_025),
    }


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if len(df) == 0 or group_col not in df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "total_cases",
                "completion_rate",
                "escalation_rate",
                "timeout_rate",
                "distance_accuracy_among_completed",
                "distance_success_rate_all_cases",
                "avg_steps",
            ]
        )

    rows = []

    for group_name, group_df in df.groupby(group_col):
        if "completion_probability" in group_df.columns:
            completion_rate = float(group_df["completion_probability"].mean())
            escalation_rate = float(group_df["escalation_probability"].mean())
            timeout_rate = float(group_df["timeout_probability"].mean())
            distance_accuracy = float(group_df["accuracy_among_completed"].mean())
            distance_success_rate = float(group_df["distance_success_probability"].mean())
        else:
            completed = group_df[group_df["outcome"] == "SUCCESS"]
            distance_metrics = compute_distance_accuracy_metrics(completed)
            completion_rate = float((group_df["outcome"] == "SUCCESS").mean())
            escalation_rate = float((group_df["outcome"] == "ESCALATE").mean())
            timeout_rate = float((group_df["outcome"] == "TIMEOUT").mean())
            distance_accuracy = float(distance_metrics["accuracy_among_completed"])
            distance_success_rate = float(group_df["distance_success_within_tolerance"].mean())

        rows.append(
            {
                group_col: group_name,
                "total_cases": int(len(group_df)),
                "completion_rate": round(completion_rate, 4),
                "escalation_rate": round(escalation_rate, 4),
                "timeout_rate": round(timeout_rate, 4),
                "distance_accuracy_among_completed": round(distance_accuracy, 4),
                "distance_success_rate_all_cases": round(distance_success_rate, 4),
                "avg_steps": round(group_df["steps"].mean(), 2),
            }
        )

    return pd.DataFrame(rows).sort_values("distance_accuracy_among_completed")


def summarize_states(df: pd.DataFrame) -> pd.DataFrame:
    failures = df[df["outcome"] != "SUCCESS"]
    if len(failures) == 0:
        return pd.DataFrame(
            columns=[
                "state_code",
                "state_name",
                "escalations",
                "timeouts",
                "non_completions",
                "pct_of_non_completions",
            ]
        )

    rows = []
    total_failures = len(failures)

    for state, state_df in failures.groupby("failure_state"):
        rows.append(
            {
                "state_code": state,
                "state_name": STATE_NAMES.get(state, state),
                "escalations": int((state_df["outcome"] == "ESCALATE").sum()),
                "timeouts": int((state_df["outcome"] == "TIMEOUT").sum()),
                "non_completions": int(len(state_df)),
                "pct_of_non_completions": round(len(state_df) / total_failures, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("non_completions", ascending=False)


def summarize_behaviors(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or "behavior_id" not in df.columns:
        return pd.DataFrame(
            columns=[
                "behavior_id",
                "total_replays",
                "pct_of_replays",
                "completion_rate",
                "escalation_rate",
                "timeout_rate",
                "distance_success_rate_all_cases",
                "avg_steps",
            ]
        )

    rows = []
    total_replays = len(df)

    for behavior_id, behavior_df in df.groupby("behavior_id"):
        rows.append(
            {
                "behavior_id": behavior_id,
                "total_replays": int(len(behavior_df)),
                "pct_of_replays": round(len(behavior_df) / total_replays, 4),
                "completion_rate": round((behavior_df["outcome"] == "SUCCESS").mean(), 4),
                "escalation_rate": round((behavior_df["outcome"] == "ESCALATE").mean(), 4),
                "timeout_rate": round((behavior_df["outcome"] == "TIMEOUT").mean(), 4),
                "distance_success_rate_all_cases": round(behavior_df["distance_success_within_tolerance"].mean(), 4),
                "avg_steps": round(behavior_df["steps"].mean(), 2),
            }
        )

    return pd.DataFrame(rows).sort_values("total_replays", ascending=False)


def run_real_patient_cohort(
    *,
    csv_path: str | Path,
    calibration_path: str | Path = "config/calibration.csv",
    results_root: str = "results",
    limit: Optional[int] = None,
    max_steps: int = 200,
    behavior_mode: str = DEFAULT_REAL_COHORT_BEHAVIOR_MODE,
    replays_per_patient: int = DEFAULT_REAL_COHORT_REPLAYS_PER_PATIENT,
    seed_base: int = 20260318,
    qms_id: Optional[str] = None,
    require_near_test: bool = False,
    save_trace: bool = False,
) -> tuple[Path, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    started_at = datetime.now().isoformat(timespec="seconds")
    started_at_perf = time.perf_counter()

    calibration = CalibrationLoader(str(calibration_path))
    load_limit = None if (qms_id is not None or require_near_test) else limit
    cases, invalid_df = load_real_patient_cases(csv_path, calibration, limit=load_limit)
    cases = select_real_patient_cases(
        cases,
        qms_id=qms_id,
        near_test_required=(True if require_near_test else None),
        limit=limit,
    )

    if qms_id is not None and len(cases) == 0:
        raise ValueError(f"No valid real-patient cohort case found for qms_id={qms_id}")
    if require_near_test and len(cases) == 0:
        raise ValueError("No valid real-patient cohort cases found with dv_near_test_required=True")
    if save_trace and (len(cases) != 1 or int(replays_per_patient) != 1):
        raise ValueError("save_trace requires exactly one selected patient case and replays_per_patient=1")

    runner = RealPatientCohortRunner(
        calibration,
        behavior_mode=behavior_mode,
        replays_per_patient=replays_per_patient,
        seed_base=seed_base,
    )
    results_folder, run_id = create_run_folder(results_root, "real_patient_cohort")

    trace_path: Optional[Path] = None
    if save_trace:
        trial_result, trace_df = runner.run_one_trial_with_trace(
            cases[0],
            max_steps=max_steps,
            replay_index=0,
        )
        trial_df = pd.DataFrame([trial_result])
        case_df = pd.DataFrame([_aggregate_case_trials([trial_result])])
        trace_path = save_trace_csv(trace_df.to_dict(orient="records"), results_folder, "trace.csv")
    else:
        case_df, trial_df = runner.run_cases(cases, max_steps=max_steps)
    completed = trial_df[trial_df["outcome"] == "SUCCESS"]

    distance_metrics = compute_distance_accuracy_metrics(completed)
    add_metrics = compute_add_accuracy_metrics(completed)
    gender_table = summarize_group(case_df, "gender")
    age_bucket_table = summarize_group(case_df, "dv_age_bucket")
    state_table = summarize_states(trial_df)
    behavior_table = summarize_behaviors(trial_df)

    save_dataframe_csv(case_df, results_folder, "cohort_replay_results.csv")
    save_dataframe_csv(trial_df, results_folder, "cohort_replay_trials.csv")
    save_dataframe_csv(invalid_df, results_folder, "excluded_records.csv")
    save_dataframe_csv(pd.DataFrame([distance_metrics | add_metrics]), results_folder, "rx_accuracy_metrics.csv")
    save_dataframe_csv(gender_table, results_folder, "leaderboard_gender.csv")
    save_dataframe_csv(age_bucket_table, results_folder, "leaderboard_age_bucket.csv")
    save_dataframe_csv(state_table, results_folder, "fsm_noncompletion_distribution.csv")
    save_dataframe_csv(behavior_table, results_folder, "behavior_distribution.csv")

    duration_seconds = time.perf_counter() - started_at_perf
    ended_at = datetime.now().isoformat(timespec="seconds")

    summary = {
        "run_id": run_id,
        "simulation_type": "real_patient_cohort",
        "csv_path": str(csv_path),
        "selected_qms_id": str(qms_id).strip() if qms_id is not None else None,
        "require_near_test": bool(require_near_test),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": float(round(duration_seconds, 3)),
        "duration_display": format_duration_seconds(duration_seconds),
        "behavior_mode": behavior_mode,
        "replays_per_patient": int(replays_per_patient),
        "total_replay_simulations": int(len(trial_df)),
        "valid_cases": int(len(case_df)),
        "excluded_cases": int(len(invalid_df)),
        "completion_rate": float(round(case_df["completion_probability"].mean(), 4)) if len(case_df) else 0.0,
        "escalation_rate": float(round(case_df["escalation_probability"].mean(), 4)) if len(case_df) else 0.0,
        "timeout_rate": float(round(case_df["timeout_probability"].mean(), 4)) if len(case_df) else 0.0,
        "distance_accuracy_among_completed": float(round(distance_metrics["accuracy_among_completed"], 4)),
        "distance_success_rate_all_cases": float(round(case_df["distance_success_probability"].mean(), 4)) if len(case_df) else 0.0,
        "average_steps": float(round(case_df["steps"].mean(), 4)) if len(case_df) else 0.0,
        "valid_add_truth_cases_among_completed": int(add_metrics["valid_add_truth_cases"]),
        "add_accuracy_among_completed_valid_add": float(round(add_metrics["add_accuracy_among_completed_valid_add"], 4)),
        "history_inference_version": REPLAY_OVERRIDE_VERSION,
        "behavior_assignment_version": runner._behavior_assignment_version(),
        "trace_file": str(trace_path) if trace_path is not None else None,
    }
    save_json(summary, results_folder, "summary.json")

    return results_folder, run_id, case_df, invalid_df, trial_df, summary
