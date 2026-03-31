from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import pandas as pd

from fsm.engines.refraction_fsm_engine import RefractionFSMEngine
from fsm.models.fsm_runtime import FSMRuntimeRow
from fsm.models.patient import PatientInput
from fsm.simulation.virtual_patient import TruthRx, VirtualPatient

SIM_DISTANCE_SUCCESS_SPH_TOL_D = 0.25
SIM_DISTANCE_SUCCESS_CYL_TOL_D = 0.25
SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG = 10.0


STATE_NAMES = {
    "B": "Coarse Sphere RE",
    "C": "Distance VA Confirm RE",
    "D": "Coarse Sphere LE",
    "E": "JCC Axis RE",
    "F": "JCC Power RE",
    "G": "Duochrome RE",
    "H": "JCC Axis LE",
    "I": "JCC Power LE",
    "J": "Duochrome LE",
    "K": "Binocular Balance",
    "L": "Distance VA Confirm LE",
    "P": "Near Add RE",
    "Q": "Near Add LE",
    "R": "Near Binocular",
    "S": "Final Compare First Option Achieved Rx",
    "T": "Final Compare Second Option PGP",
    "U": "Final Compare Decision",
}


def axis_error(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    d = abs(float(a) - float(b)) % 180
    return min(d, 180 - d)


def case_patient_input(case) -> Optional[PatientInput]:
    return getattr(case, "patient_input", None) or getattr(case, "patient", None)


def case_truth(case) -> TruthRx:
    if hasattr(case, "truth") and isinstance(case.truth, TruthRx):
        return case.truth

    truth_rx = getattr(case, "truth_rx", None)
    if isinstance(truth_rx, dict):
        return TruthRx(
            re_sph=float(truth_rx["re_sph"]),
            re_cyl=float(truth_rx["re_cyl"]),
            re_axis=float(truth_rx["re_axis"]),
            le_sph=float(truth_rx["le_sph"]),
            le_cyl=float(truth_rx["le_cyl"]),
            le_axis=float(truth_rx["le_axis"]),
            add_r=float(truth_rx.get("add_r", 0.0) or 0.0),
            add_l=float(truth_rx.get("add_l", 0.0) or 0.0),
        )

    raise TypeError("Simulation case does not expose truth or truth_rx")


def case_ar_re(case) -> Optional[EyePrescription]:
    patient = case_patient_input(case)
    return getattr(case, "ar_re", None) or (patient.autorefractor_re if patient else None)


def case_ar_le(case) -> Optional[EyePrescription]:
    patient = case_patient_input(case)
    return getattr(case, "ar_le", None) or (patient.autorefractor_le if patient else None)


def case_lenso_re(case) -> Optional[EyePrescription]:
    patient = case_patient_input(case)
    return getattr(case, "lenso_re", None) or (patient.lenso_re if patient else None)


def case_lenso_le(case) -> Optional[EyePrescription]:
    patient = case_patient_input(case)
    return getattr(case, "lenso_le", None) or (patient.lenso_le if patient else None)


def seed_final_compare_context(row: FSMRuntimeRow, patient_input: Optional[PatientInput]) -> None:
    row.final_compare_enabled = False
    row.final_compare_option_source = "Achieved"
    row.final_compare_round = 0
    row.final_compare_choice_round_1 = ""
    row.final_compare_choice_round_2 = ""
    row.patient_accepted_achieved_over_current_rx = ""

    if not patient_input:
        return

    lenso_re = patient_input.lenso_re
    lenso_le = patient_input.lenso_le
    enabled = bool(
        lenso_re
        and lenso_le
        and lenso_re.has_full_rx()
        and lenso_le.has_full_rx()
    )
    row.final_compare_enabled = enabled
    if not enabled:
        return

    row.final_compare_current_re_sph = lenso_re.sphere if lenso_re else None
    row.final_compare_current_re_cyl = lenso_re.cylinder if lenso_re else None
    row.final_compare_current_re_axis = lenso_re.axis if lenso_re else None
    row.final_compare_current_le_sph = lenso_le.sphere if lenso_le else None
    row.final_compare_current_le_cyl = lenso_le.cylinder if lenso_le else None
    row.final_compare_current_le_axis = lenso_le.axis if lenso_le else None
    row.final_compare_current_add_r = patient_input.lenso_add_r
    row.final_compare_current_add_l = patient_input.lenso_add_l


def _rx_payload(
    re_sph: Optional[float],
    re_cyl: Optional[float],
    re_axis: Optional[float],
    add_r: Optional[float],
    le_sph: Optional[float],
    le_cyl: Optional[float],
    le_axis: Optional[float],
    add_l: Optional[float],
) -> dict:
    return {
        "right": {"sph": re_sph, "cyl": re_cyl, "axis": re_axis, "add": add_r},
        "left": {"sph": le_sph, "cyl": le_cyl, "axis": le_axis, "add": add_l},
    }


def _rx_has_any_values(payload: dict) -> bool:
    for eye in ("right", "left"):
        eye_payload = payload.get(eye, {})
        if any(eye_payload.get(key) is not None for key in ("sph", "cyl", "axis", "add")):
            return True
    return False


def resolved_final_compare_payloads(row: FSMRuntimeRow) -> dict:
    achieved = _rx_payload(
        row.final_compare_achieved_re_sph,
        row.final_compare_achieved_re_cyl,
        row.final_compare_achieved_re_axis,
        row.final_compare_achieved_add_r,
        row.final_compare_achieved_le_sph,
        row.final_compare_achieved_le_cyl,
        row.final_compare_achieved_le_axis,
        row.final_compare_achieved_add_l,
    )
    current = _rx_payload(
        row.final_compare_current_re_sph,
        row.final_compare_current_re_cyl,
        row.final_compare_current_re_axis,
        row.final_compare_current_add_r,
        row.final_compare_current_le_sph,
        row.final_compare_current_le_cyl,
        row.final_compare_current_le_axis,
        row.final_compare_current_add_l,
    )
    fallback = _rx_payload(
        row.re_sph,
        row.re_cyl,
        row.re_axis,
        row.add_r,
        row.le_sph,
        row.le_cyl,
        row.le_axis,
        row.add_l,
    )

    acceptance_flag = str(row.patient_accepted_achieved_over_current_rx or "").strip()
    achieved_available = _rx_has_any_values(achieved)
    current_available = _rx_has_any_values(current)

    if acceptance_flag == "Yes" and achieved_available:
        prescribed = achieved
        selected_source = "Achieved"
    elif acceptance_flag == "No" and current_available:
        prescribed = current
        selected_source = "PGP"
    elif achieved_available:
        prescribed = achieved
        selected_source = ""
    elif current_available:
        prescribed = current
        selected_source = ""
    else:
        prescribed = fallback
        selected_source = ""

    return {
        "achieved": achieved,
        "current": current,
        "prescribed": prescribed,
        "selected_source": selected_source,
    }


def flatten_rx_payload(prefix: str, payload: dict) -> dict:
    right = payload.get("right", {})
    left = payload.get("left", {})
    return {
        f"{prefix}_re_sph": right.get("sph"),
        f"{prefix}_re_cyl": right.get("cyl"),
        f"{prefix}_re_axis": right.get("axis"),
        f"{prefix}_le_sph": left.get("sph"),
        f"{prefix}_le_cyl": left.get("cyl"),
        f"{prefix}_le_axis": left.get("axis"),
        f"{prefix}_add_r": right.get("add"),
        f"{prefix}_add_l": left.get("add"),
    }


def prescription_distance_score(payload: dict, truth: TruthRx) -> float:
    right = payload.get("right", {})
    left = payload.get("left", {})
    return (
        abs(float(right.get("sph") or 0.0) - truth.re_sph)
        + abs(float(right.get("cyl") or 0.0) - truth.re_cyl)
        + axis_error(right.get("axis"), truth.re_axis) / 20.0
        + abs(float(left.get("sph") or 0.0) - truth.le_sph)
        + abs(float(left.get("cyl") or 0.0) - truth.le_cyl)
        + axis_error(left.get("axis"), truth.le_axis) / 20.0
        + abs(float(right.get("add") or 0.0) - truth.add_r) * 0.5
        + abs(float(left.get("add") or 0.0) - truth.add_l) * 0.5
    )


def prescription_error_metrics(payload: dict, truth: TruthRx, truth_add_valid: bool) -> dict:
    right = payload.get("right", {})
    left = payload.get("left", {})

    re_sph_err = abs(float(right.get("sph") or 0.0) - truth.re_sph)
    re_cyl_err = abs(float(right.get("cyl") or 0.0) - truth.re_cyl)
    re_axis_err = axis_error(right.get("axis"), truth.re_axis)

    le_sph_err = abs(float(left.get("sph") or 0.0) - truth.le_sph)
    le_cyl_err = abs(float(left.get("cyl") or 0.0) - truth.le_cyl)
    le_axis_err = axis_error(left.get("axis"), truth.le_axis)

    add_r_err = abs(float(right.get("add") or 0.0) - truth.add_r) if truth_add_valid else None
    add_l_err = abs(float(left.get("add") or 0.0) - truth.add_l) if truth_add_valid else None

    distance_success = (
        re_sph_err <= SIM_DISTANCE_SUCCESS_SPH_TOL_D
        and re_cyl_err <= SIM_DISTANCE_SUCCESS_CYL_TOL_D
        and re_axis_err <= SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG
        and le_sph_err <= SIM_DISTANCE_SUCCESS_SPH_TOL_D
        and le_cyl_err <= SIM_DISTANCE_SUCCESS_CYL_TOL_D
        and le_axis_err <= SIM_DISTANCE_SUCCESS_AXIS_TOL_DEG
    )
    add_success = (
        (add_r_err is not None and add_r_err <= 0.25)
        and (add_l_err is not None and add_l_err <= 0.25)
    ) if truth_add_valid else None

    return {
        "re_sph_err": re_sph_err,
        "re_cyl_err": re_cyl_err,
        "re_axis_err": re_axis_err,
        "le_sph_err": le_sph_err,
        "le_cyl_err": le_cyl_err,
        "le_axis_err": le_axis_err,
        "add_r_err": add_r_err,
        "add_l_err": add_l_err,
        "distance_success_within_tolerance": distance_success,
        "add_success_within_tolerance": add_success,
    }


def execute_case(
    *,
    engine: RefractionFSMEngine,
    case,
    behavior_model=None,
    max_steps: int = 200,
    collect_trace: bool = False,
    trace_metadata: Optional[dict] = None,
) -> tuple[dict, Optional[pd.DataFrame]]:
    patient = case_patient_input(case)
    truth = case_truth(case)

    current = engine.initialize_row(
        visit_id=getattr(case, "case_id", getattr(patient, "visit_id", "SIM_CASE")),
        dv=case.dv,
        ar_re=case_ar_re(case),
        ar_le=case_ar_le(case),
    )
    seed_final_compare_context(current, patient)

    deterministic_patient = VirtualPatient(truth)
    steps = 0
    last_row = None
    trace_rows: list[dict] = []

    while steps < max_steps:
        if behavior_model is None:
            response = deterministic_patient.respond(current)
        else:
            response = behavior_model.respond(row=current, truth=truth, case_context=case)

        finalized = engine.apply_response(
            current=current,
            response_value=response,
            dv=case.dv,
            ar_re=case_ar_re(case),
            ar_le=case_ar_le(case),
        )
        last_row = finalized
        steps += 1

        if collect_trace:
            trace_row = asdict(finalized)
            if trace_metadata:
                trace_row.update(trace_metadata)
            trace_rows.append(trace_row)

        if finalized.next_state in ("END", "ESCALATE"):
            break

        next_row = engine._build_next_row(finalized, case.dv)
        if next_row is None:
            break
        current = next_row

    if last_row is None:
        raise RuntimeError("Simulation produced no FSM rows")

    outcome = "SUCCESS" if last_row.next_state == "END" else "ESCALATE"
    if steps >= max_steps and outcome != "SUCCESS":
        outcome = "TIMEOUT"

    payloads = resolved_final_compare_payloads(last_row)
    prescribed_metrics = prescription_error_metrics(
        payloads["prescribed"],
        truth,
        bool(getattr(case, "truth_add_valid", True)),
    )
    achieved_metrics = prescription_error_metrics(
        payloads["achieved"] if _rx_has_any_values(payloads["achieved"]) else payloads["prescribed"],
        truth,
        bool(getattr(case, "truth_add_valid", True)),
    )

    result = {
        "steps": steps,
        "outcome": outcome,
        "termination_state": last_row.next_state,
        "failure_state": last_row.state,
        "final_compare_enabled": bool(last_row.final_compare_enabled),
        "final_compare_round": int(last_row.final_compare_round or 0),
        "final_compare_choice_round_1": last_row.final_compare_choice_round_1,
        "final_compare_choice_round_2": last_row.final_compare_choice_round_2,
        "patient_accepted_achieved_over_current_rx": last_row.patient_accepted_achieved_over_current_rx,
        "selected_prescribed_rx_source": payloads["selected_source"],
        "working_end_re_sph": last_row.re_sph,
        "working_end_re_cyl": last_row.re_cyl,
        "working_end_re_axis": last_row.re_axis,
        "working_end_le_sph": last_row.le_sph,
        "working_end_le_cyl": last_row.le_cyl,
        "working_end_le_axis": last_row.le_axis,
        "working_end_add_r": last_row.add_r,
        "working_end_add_l": last_row.add_l,
        "distance_va_re_chart": last_row.distance_va_re_chart,
        "distance_va_le_chart": last_row.distance_va_le_chart,
        "distance_va_re_line": last_row.distance_va_re_line,
        "distance_va_le_line": last_row.distance_va_le_line,
    }
    result.update(flatten_rx_payload("achieved", payloads["achieved"]))
    result.update(flatten_rx_payload("pgp", payloads["current"]))
    result.update(flatten_rx_payload("final", payloads["prescribed"]))
    result.update(prescribed_metrics)
    result.update(
        {
            "achieved_re_sph_err": achieved_metrics["re_sph_err"],
            "achieved_re_cyl_err": achieved_metrics["re_cyl_err"],
            "achieved_re_axis_err": achieved_metrics["re_axis_err"],
            "achieved_le_sph_err": achieved_metrics["le_sph_err"],
            "achieved_le_cyl_err": achieved_metrics["le_cyl_err"],
            "achieved_le_axis_err": achieved_metrics["le_axis_err"],
            "achieved_add_r_err": achieved_metrics["add_r_err"],
            "achieved_add_l_err": achieved_metrics["add_l_err"],
            "achieved_distance_success_within_tolerance": achieved_metrics["distance_success_within_tolerance"],
            "achieved_add_success_within_tolerance": achieved_metrics["add_success_within_tolerance"],
        }
    )

    trace_df = pd.DataFrame(trace_rows) if collect_trace else None
    return result, trace_df
