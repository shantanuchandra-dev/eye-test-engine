"""
Output handling: Write session CSV logs, metadata JSON, and combined logs.
Adapted for ETE_v2 — uses FSMRuntimeRow instead of RowContext.
"""
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Session CSV schema (23 columns)
SESSION_CSV_FIELDS = [
    "Row_Number",
    "Timestamp",
    "Manual_or_QnA",
    "Input_Method",
    "R_SPH",
    "R_CYL",
    "R_AXIS",
    "R_ADD",
    "L_SPH",
    "L_CYL",
    "L_AXIS",
    "L_ADD",
    "Occluder_State",
    "Chart_Number",
    "Chart_Display",
    "Change_Delta",
    "Current_Phase",
    "Phase_ID",
    "Optometrist_Question",
    "Patient_Answer_Intent",
    "Voice_Transcript",
    "Voice_Alternatives",
    "Voice_Confidence",
]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _row_to_dict(row: dict) -> dict:
    """Convert a SessionRow dict into a flat dict matching SESSION_CSV_FIELDS."""
    return {
        "Row_Number": row.get("row_number", ""),
        "Timestamp": row.get("timestamp", ""),
        "Manual_or_QnA": row.get("interaction_type", "QnA"),
        "Input_Method": row.get("input_method", ""),
        "R_SPH": row.get("re_sph", ""),
        "R_CYL": row.get("re_cyl", ""),
        "R_AXIS": row.get("re_axis", ""),
        "R_ADD": row.get("add_r", ""),
        "L_SPH": row.get("le_sph", ""),
        "L_CYL": row.get("le_cyl", ""),
        "L_AXIS": row.get("le_axis", ""),
        "L_ADD": row.get("add_l", ""),
        "Occluder_State": row.get("occluder_state", ""),
        "Chart_Number": row.get("chart_param", ""),
        "Chart_Display": row.get("chart_display", ""),
        "Change_Delta": row.get("change_delta", ""),
        "Current_Phase": row.get("phase_name", ""),
        "Phase_ID": row.get("state", ""),
        "Optometrist_Question": row.get("question", ""),
        "Patient_Answer_Intent": row.get("response_value", ""),
        "Voice_Transcript": row.get("transcript", ""),
        "Voice_Alternatives": row.get("alternatives", ""),
        "Voice_Confidence": row.get("match_confidence", ""),
    }


def session_csv_string(rows: List[dict]) -> str:
    """Return per-session CSV content as a string. For remote upload."""
    if not rows:
        return ""
    import io as _io
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SESSION_CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(_row_to_dict(row))
    return buf.getvalue()


def write_session_csv(rows: List[dict], output_path: Path) -> None:
    if not rows:
        return
    _ensure_dir(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SESSION_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_dict(row))


VOICE_UTTERANCE_FIELDS = [
    "Timestamp",
    "Session_ID",
    "Step",
    "State",
    "Phase_Name",
    "Transcript",
    "Alternatives",
    "Intent_Matched",
    "Canonical_Label",
    "Confidence",
    "Match_Method",
    "Input_Method",
    "Language",
    "Stimulus_Letters",
    "Audio_File",
    "Accepted",
]


def write_voice_utterances_csv(
    session_history: List[dict],
    failed_voice_attempts: List[dict],
    session_id: str,
    output_path: Path,
) -> None:
    """Write a per-session voice utterance CSV for training data.

    Includes both successful voice matches (from session_history) and
    failed attempts, sorted by timestamp for chronological ordering.
    """
    utterances = []

    # Successful voice interactions from session history
    for row in session_history:
        im = row.get("input_method", "")
        if not im.startswith("Voice"):
            continue
        utterances.append({
            "Timestamp": row.get("timestamp", ""),
            "Session_ID": session_id,
            "Step": row.get("row_number", ""),
            "State": row.get("state", ""),
            "Phase_Name": row.get("phase_name", ""),
            "Transcript": row.get("transcript", ""),
            "Alternatives": row.get("alternatives", ""),
            "Intent_Matched": row.get("response_value", ""),
            "Canonical_Label": row.get("canonical_label", ""),
            "Confidence": row.get("match_confidence", ""),
            "Match_Method": row.get("match_method", ""),
            "Input_Method": im,
            "Language": row.get("session_language", ""),
            "Stimulus_Letters": row.get("stimulus_letters", ""),
            "Audio_File": row.get("audio_file", ""),
            "Accepted": "true",
        })

    # Failed voice attempts
    for att in (failed_voice_attempts or []):
        alts = att.get("alternatives", [])
        if isinstance(alts, list):
            alts = "; ".join(str(a) for a in alts)
        utterances.append({
            "Timestamp": att.get("timestamp", ""),
            "Session_ID": session_id,
            "Step": att.get("step", ""),
            "State": att.get("state", ""),
            "Phase_Name": att.get("phase_name", ""),
            "Transcript": att.get("transcript", ""),
            "Alternatives": alts,
            "Intent_Matched": "",
            "Canonical_Label": att.get("canonical_label", ""),
            "Confidence": att.get("match_confidence", ""),
            "Match_Method": att.get("match_method", ""),
            "Input_Method": f"Voice_{att.get('backend', 'Browser')}".replace("voice_", ""),
            "Language": att.get("language", ""),
            "Stimulus_Letters": att.get("stimulus_letters", ""),
            "Audio_File": "",
            "Accepted": "false",
        })

    if not utterances:
        return

    # Sort by timestamp for chronological order
    utterances.sort(key=lambda u: u.get("Timestamp", ""))

    _ensure_dir(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VOICE_UTTERANCE_FIELDS)
        writer.writeheader()
        writer.writerows(utterances)


def write_session_metadata(metadata: dict, output_path: Path) -> None:
    _ensure_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def append_to_combined_log(rows: List[dict], session_id: str,
                           combined_path: Path) -> None:
    if not rows:
        return
    _ensure_dir(combined_path)
    fields = ["Session_ID"] + SESSION_CSV_FIELDS
    file_exists = combined_path.exists() and combined_path.stat().st_size > 0
    with combined_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            row_dict = {"Session_ID": session_id, **_row_to_dict(row)}
            writer.writerow(row_dict)


def combined_log_rows_csv_string(
    rows: List[dict], session_id: str, *, include_header: bool
) -> str:
    """CSV for combined_log: Session_ID + session rows. Used for remote merge."""
    if not rows:
        return ""
    import io as _io

    fields = ["Session_ID"] + SESSION_CSV_FIELDS
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    if include_header:
        writer.writeheader()
    for row in rows:
        row_dict = {"Session_ID": session_id, **_row_to_dict(row)}
        writer.writerow(row_dict)
    return buf.getvalue()


COMBINED_METADATA_FIELDS = [
    "Session_ID",
    "Phoropter_ID",
    "Operator_Name",
    "Customer_Name",
    "Customer_Phone",
    "Customer_Age",
    "Customer_Gender",
    "Start_Time",
    "End_Time",
    "Duration_Seconds",
    "Completion_Status",
    "Total_Interactions",
    "Manual_Count",
    "QnA_Count",
    "Phase_Jump_Count",
    "Unable_To_Read_Count",
    "AR_R_SPH", "AR_R_CYL", "AR_R_AXIS",
    "AR_L_SPH", "AR_L_CYL", "AR_L_AXIS",
    "Lenso_R_SPH", "Lenso_R_CYL", "Lenso_R_AXIS",
    "Lenso_L_SPH", "Lenso_L_CYL", "Lenso_L_AXIS",
    "Achieved_R_SPH", "Achieved_R_CYL", "Achieved_R_AXIS", "Achieved_R_ADD",
    "Achieved_L_SPH", "Achieved_L_CYL", "Achieved_L_AXIS", "Achieved_L_ADD",
    "Current_R_SPH", "Current_R_CYL", "Current_R_AXIS", "Current_R_ADD",
    "Current_L_SPH", "Current_L_CYL", "Current_L_AXIS", "Current_L_ADD",
    "Final_R_SPH", "Final_R_CYL", "Final_R_AXIS", "Final_R_ADD",
    "Final_L_SPH", "Final_L_CYL", "Final_L_AXIS", "Final_L_ADD",
    "Final_R_Distance_VA", "Final_L_Distance_VA",
    "Final_Rx_Compare_Ran", "Final_Rx_Compare_Current_Source",
    "Final_Rx_Compare_Round_1", "Final_Rx_Compare_Round_2",
    "Accepted_Achieved_Over_Current_Rx", "Final_Rx_Selected_Source",
    "Phases_Completed",
    # ── Patient Input fields ──
    "PI_Age", "PI_Occupation", "PI_Screen_Time_Hours", "PI_Driving_Hours",
    "PI_Primary_Reason", "PI_Symptoms", "PI_Satisfaction",
    "PI_Wear_Type", "PI_Distance_Target", "PI_Priority", "PI_Near_Priority",
    "PI_Last_Test_Months", "PI_Rx_Change_Large", "PI_Fluctuating_Vision",
    "PI_Diabetes", "PI_Prior_Surgery", "PI_Keratoconus", "PI_Amblyopia",
    "PI_Infection", "PI_Optom_Review",
    # ── Derived Variables fields ──
    "DV_Age_Bucket", "DV_Distance_Priority", "DV_Near_Priority",
    "DV_Symptom_Risk", "DV_Medical_Risk", "DV_Stability",
    "DV_Mismatch_RE", "DV_Mismatch_LE", "DV_Start_Source_Policy",
    "DV_Start_RE_SPH", "DV_Start_RE_CYL", "DV_Start_RE_AXIS",
    "DV_Start_LE_SPH", "DV_Start_LE_CYL", "DV_Start_LE_AXIS",
    "DV_Add_Expected", "DV_Target_VA", "DV_Endpoint_Bias", "DV_Step_Size_Policy",
    "DV_Max_Delta_Start_SPH", "DV_Max_Delta_AR_SPH",
    "DV_Axis_Tolerance", "DV_Axis_Tolerance_RE", "DV_Axis_Tolerance_LE",
    "DV_Axis_Tolerance_CYL_RE", "DV_Axis_Tolerance_CYL_LE", "DV_Cyl_Tolerance",
    "DV_Requires_Optom_Review", "DV_Anomaly_Watch",
    "DV_Convergence_Time", "DV_Branching_Guardrails", "DV_Confidence_Req",
    "DV_Fogging_Policy", "DV_Fogging_Amount", "DV_Fogging_Clearance", "DV_Fogging_Confirm",
    "DV_Axis_Step_Policy", "DV_Duochrome_Max_Flips", "DV_Near_Test_Required",
    "DV_Accommodation_Level", "DV_Fogging_Required", "DV_Fogging_Stop_At_Target",
    "DV_JCC_Axis_Same_Req", "DV_JCC_Axis_Max_Flips",
    "DV_Near_Start_Add_R", "DV_Near_Start_Add_L",
    "DV_Near_Binoc_Step", "DV_Near_Binoc_Max_Plus", "DV_Near_Binoc_Max_Minus",
]


def _safe_get(d, *keys, default=""):
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current if current is not None else default


def build_combined_metadata_flat(metadata: dict) -> dict:
    """Single combined_metadata.csv row as a flat dict."""
    qm = metadata.get("quality_metrics", {})
    final = metadata.get("final_prescription", {})
    achieved = metadata.get("achieved_prescription", {})
    current_rx = metadata.get("pgp_rx", metadata.get("current_rx", {}))
    final_compare = metadata.get("final_rx_comparison", {})
    ar = metadata.get("ar", {})
    lenso = metadata.get("lensometry", {})
    pi = metadata.get("patient_input", {})
    dv = metadata.get("derived_variables", {})
    return {
        "Session_ID": metadata.get("session_id", ""),
        "Phoropter_ID": metadata.get("phoropter_id", ""),
        "Operator_Name": metadata.get("operator_name", ""),
        "Customer_Name": metadata.get("customer_name", ""),
        "Customer_Phone": metadata.get("customer_phone", ""),
        "Customer_Age": metadata.get("customer_age", ""),
        "Customer_Gender": metadata.get("customer_gender", ""),
        "Start_Time": metadata.get("session_start_time", ""),
        "End_Time": metadata.get("session_end_time", ""),
        "Duration_Seconds": metadata.get("session_duration_seconds", ""),
        "Completion_Status": metadata.get("test_completion_status", ""),
        "Total_Interactions": metadata.get("total_interactions", ""),
        "Manual_Count": qm.get("manual_adjustment_count", 0),
        "QnA_Count": qm.get("qna_interaction_count", 0),
        "Phase_Jump_Count": qm.get("phase_jump_count", 0),
        "Unable_To_Read_Count": qm.get("unable_to_read_count", 0),
        "AR_R_SPH": _safe_get(ar, "right", "sph"),
        "AR_R_CYL": _safe_get(ar, "right", "cyl"),
        "AR_R_AXIS": _safe_get(ar, "right", "axis"),
        "AR_L_SPH": _safe_get(ar, "left", "sph"),
        "AR_L_CYL": _safe_get(ar, "left", "cyl"),
        "AR_L_AXIS": _safe_get(ar, "left", "axis"),
        "Lenso_R_SPH": _safe_get(lenso, "right", "sph"),
        "Lenso_R_CYL": _safe_get(lenso, "right", "cyl"),
        "Lenso_R_AXIS": _safe_get(lenso, "right", "axis"),
        "Lenso_L_SPH": _safe_get(lenso, "left", "sph"),
        "Lenso_L_CYL": _safe_get(lenso, "left", "cyl"),
        "Lenso_L_AXIS": _safe_get(lenso, "left", "axis"),
        "Achieved_R_SPH": _safe_get(achieved, "right", "sph"),
        "Achieved_R_CYL": _safe_get(achieved, "right", "cyl"),
        "Achieved_R_AXIS": _safe_get(achieved, "right", "axis"),
        "Achieved_R_ADD": _safe_get(achieved, "right", "add"),
        "Achieved_L_SPH": _safe_get(achieved, "left", "sph"),
        "Achieved_L_CYL": _safe_get(achieved, "left", "cyl"),
        "Achieved_L_AXIS": _safe_get(achieved, "left", "axis"),
        "Achieved_L_ADD": _safe_get(achieved, "left", "add"),
        "Current_R_SPH": _safe_get(current_rx, "right", "sph"),
        "Current_R_CYL": _safe_get(current_rx, "right", "cyl"),
        "Current_R_AXIS": _safe_get(current_rx, "right", "axis"),
        "Current_R_ADD": _safe_get(current_rx, "right", "add"),
        "Current_L_SPH": _safe_get(current_rx, "left", "sph"),
        "Current_L_CYL": _safe_get(current_rx, "left", "cyl"),
        "Current_L_AXIS": _safe_get(current_rx, "left", "axis"),
        "Current_L_ADD": _safe_get(current_rx, "left", "add"),
        "Final_R_SPH": _safe_get(final, "right", "sph"),
        "Final_R_CYL": _safe_get(final, "right", "cyl"),
        "Final_R_AXIS": _safe_get(final, "right", "axis"),
        "Final_R_ADD": _safe_get(final, "right", "add"),
        "Final_L_SPH": _safe_get(final, "left", "sph"),
        "Final_L_CYL": _safe_get(final, "left", "cyl"),
        "Final_L_AXIS": _safe_get(final, "left", "axis"),
        "Final_L_ADD": _safe_get(final, "left", "add"),
        "Final_R_Distance_VA": _safe_get(metadata, "final_distance_va", "right", "line"),
        "Final_L_Distance_VA": _safe_get(metadata, "final_distance_va", "left", "line"),
        "Final_Rx_Compare_Ran": final_compare.get("ran", ""),
        "Final_Rx_Compare_Current_Source": final_compare.get("current_source", ""),
        "Final_Rx_Compare_Round_1": final_compare.get("round_1_choice", ""),
        "Final_Rx_Compare_Round_2": final_compare.get("round_2_choice", ""),
        "Accepted_Achieved_Over_Current_Rx": final_compare.get("accepted_achieved_over_current_rx", ""),
        "Final_Rx_Selected_Source": final_compare.get("selected_prescribed_rx_source", ""),
        "Phases_Completed": "; ".join(metadata.get("phases_completed", [])),
        # ── Patient Input ──
        "PI_Age": pi.get("age", ""),
        "PI_Occupation": pi.get("occupation", ""),
        "PI_Screen_Time_Hours": pi.get("screen_time_hours", ""),
        "PI_Driving_Hours": pi.get("driving_hours", ""),
        "PI_Primary_Reason": pi.get("primary_reason", ""),
        "PI_Symptoms": pi.get("symptoms_text", ""),
        "PI_Satisfaction": pi.get("satisfaction_with_current_rx", ""),
        "PI_Wear_Type": pi.get("wear_type", ""),
        "PI_Distance_Target": pi.get("distance_target_preference", ""),
        "PI_Priority": pi.get("priority", ""),
        "PI_Near_Priority": pi.get("near_priority_declared", ""),
        "PI_Last_Test_Months": pi.get("last_eye_test_months_ago", ""),
        "PI_Rx_Change_Large": pi.get("rx_change_was_large", ""),
        "PI_Fluctuating_Vision": pi.get("fluctuating_vision_reported", ""),
        "PI_Diabetes": pi.get("diabetes", ""),
        "PI_Prior_Surgery": pi.get("prior_eye_surgery", ""),
        "PI_Keratoconus": pi.get("keratoconus", ""),
        "PI_Amblyopia": pi.get("amblyopia", ""),
        "PI_Infection": pi.get("infection", ""),
        "PI_Optom_Review": pi.get("optom_review_flag", ""),
        # ── Derived Variables ──
        "DV_Age_Bucket": dv.get("dv_age_bucket", ""),
        "DV_Distance_Priority": dv.get("dv_distance_priority", ""),
        "DV_Near_Priority": dv.get("dv_near_priority", ""),
        "DV_Symptom_Risk": dv.get("dv_symptom_risk_level", ""),
        "DV_Medical_Risk": dv.get("dv_medical_risk_level", ""),
        "DV_Stability": dv.get("dv_stability_level", ""),
        "DV_Mismatch_RE": dv.get("dv_ar_lenso_mismatch_level_RE", ""),
        "DV_Mismatch_LE": dv.get("dv_ar_lenso_mismatch_level_LE", ""),
        "DV_Start_Source_Policy": dv.get("dv_start_source_policy", ""),
        "DV_Start_RE_SPH": dv.get("dv_start_rx_RE_sph", ""),
        "DV_Start_RE_CYL": dv.get("dv_start_rx_RE_cyl", ""),
        "DV_Start_RE_AXIS": dv.get("dv_start_rx_RE_axis", ""),
        "DV_Start_LE_SPH": dv.get("dv_start_rx_LE_sph", ""),
        "DV_Start_LE_CYL": dv.get("dv_start_rx_LE_cyl", ""),
        "DV_Start_LE_AXIS": dv.get("dv_start_rx_LE_axis", ""),
        "DV_Add_Expected": dv.get("dv_add_expected", ""),
        "DV_Target_VA": dv.get("dv_target_distance_va", ""),
        "DV_Endpoint_Bias": dv.get("dv_endpoint_bias_policy", ""),
        "DV_Step_Size_Policy": dv.get("dv_step_size_policy", ""),
        "DV_Max_Delta_Start_SPH": dv.get("dv_max_delta_from_start_sph", ""),
        "DV_Max_Delta_AR_SPH": dv.get("dv_max_delta_from_ar_sph", ""),
        "DV_Axis_Tolerance": dv.get("dv_axis_tolerance_deg", ""),
        "DV_Axis_Tolerance_RE": dv.get("dv_axis_tolerance_deg_RE", ""),
        "DV_Axis_Tolerance_LE": dv.get("dv_axis_tolerance_deg_LE", ""),
        "DV_Axis_Tolerance_CYL_RE": dv.get("dv_axis_cyl_magnitude_for_tolerance_RE", ""),
        "DV_Axis_Tolerance_CYL_LE": dv.get("dv_axis_cyl_magnitude_for_tolerance_LE", ""),
        "DV_Cyl_Tolerance": dv.get("dv_cyl_tolerance_D", ""),
        "DV_Requires_Optom_Review": dv.get("dv_requires_optom_review", ""),
        "DV_Anomaly_Watch": dv.get("dv_anomaly_watch", ""),
        "DV_Convergence_Time": dv.get("dv_expected_convergence_time", ""),
        "DV_Branching_Guardrails": dv.get("dv_branching_guardrails", ""),
        "DV_Confidence_Req": dv.get("dv_confidence_requirement", ""),
        "DV_Fogging_Policy": dv.get("dv_fogging_policy", ""),
        "DV_Fogging_Amount": dv.get("dv_fogging_amount_D", ""),
        "DV_Fogging_Clearance": dv.get("dv_fogging_clearance_mode", ""),
        "DV_Fogging_Confirm": dv.get("dv_fogging_required_confirmation", ""),
        "DV_Axis_Step_Policy": dv.get("dv_axis_step_policy", ""),
        "DV_Duochrome_Max_Flips": dv.get("dv_duochrome_max_flips", ""),
        "DV_Near_Test_Required": dv.get("dv_near_test_required", ""),
        "DV_Accommodation_Level": dv.get("dv_accommodation_level", ""),
        "DV_Fogging_Required": dv.get("dv_fogging_required", ""),
        "DV_Fogging_Stop_At_Target": dv.get("dv_fogging_stop_at_target_va", ""),
        "DV_JCC_Axis_Same_Req": dv.get("dv_jcc_axis_same_required", ""),
        "DV_JCC_Axis_Max_Flips": dv.get("dv_jcc_axis_max_flips", ""),
        "DV_Near_Start_Add_R": dv.get("dv_near_start_add_r", ""),
        "DV_Near_Start_Add_L": dv.get("dv_near_start_add_l", ""),
        "DV_Near_Binoc_Step": dv.get("dv_near_binoc_step_D", ""),
        "DV_Near_Binoc_Max_Plus": dv.get("dv_near_binoc_max_plus_steps", ""),
        "DV_Near_Binoc_Max_Minus": dv.get("dv_near_binoc_max_minus_steps", ""),
    }


def combined_metadata_row_csv_string(metadata: dict, *, include_header: bool) -> str:
    """One row (and optional header) for combined_metadata.csv."""
    import io as _io

    flat = build_combined_metadata_flat(metadata)
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COMBINED_METADATA_FIELDS)
    if include_header:
        writer.writeheader()
    writer.writerow(flat)
    return buf.getvalue()


def append_to_combined_metadata(metadata: dict, combined_path: Path) -> None:
    _ensure_dir(combined_path)
    file_exists = combined_path.exists() and combined_path.stat().st_size > 0
    flat = build_combined_metadata_flat(metadata)
    with combined_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)


def _serialize_patient_input(patient_input) -> dict:
    """Convert PatientInput dataclass to a JSON-safe dict."""
    if patient_input is None:
        return {}
    d = asdict(patient_input)
    # EyePrescription sub-objects are already dicts via asdict
    return d


def _serialize_derived_variables(dv) -> dict:
    """Convert DerivedVariables dataclass to a JSON-safe dict."""
    if dv is None:
        return {}
    return asdict(dv)


def _eye_payload_from_prescription(rx) -> dict:
    if rx is None:
        return {}
    return {
        "sph": getattr(rx, "sphere", None),
        "cyl": getattr(rx, "cylinder", None),
        "axis": getattr(rx, "axis", None),
    }


def _rx_payload(
    re_sph,
    re_cyl,
    re_axis,
    add_r,
    le_sph,
    le_cyl,
    le_axis,
    add_l,
) -> dict:
    return {
        "right": {"sph": re_sph, "cyl": re_cyl, "axis": re_axis, "add": add_r},
        "left": {"sph": le_sph, "cyl": le_cyl, "axis": le_axis, "add": add_l},
    }


def _payload_has_values(payload: dict) -> bool:
    for eye in ("right", "left"):
        eye_payload = payload.get(eye, {})
        if any(eye_payload.get(key) is not None for key in ("sph", "cyl", "axis", "add")):
            return True
    return False


def _derive_objective_payloads(patient_input) -> tuple[dict, dict]:
    if patient_input is None:
        return {}, {}
    ar = {
        "right": _eye_payload_from_prescription(getattr(patient_input, "autorefractor_re", None)),
        "left": _eye_payload_from_prescription(getattr(patient_input, "autorefractor_le", None)),
    }
    lenso = {
        "right": {
            **_eye_payload_from_prescription(getattr(patient_input, "lenso_re", None)),
            "add": getattr(patient_input, "lenso_add_r", None),
        },
        "left": {
            **_eye_payload_from_prescription(getattr(patient_input, "lenso_le", None)),
            "add": getattr(patient_input, "lenso_add_l", None),
        },
    }
    return ar, lenso


def build_session_metadata(
    session_id: str,
    phoropter_id: str,
    session_start_time: datetime,
    session_end_time: datetime,
    completion_status: str,
    rows: List[dict],
    ar: Optional[dict] = None,
    lensometry: Optional[dict] = None,
    phase_jump_count: int = 0,
    unable_to_read_count: int = 0,
    phases_completed: Optional[List[str]] = None,
    phases_skipped: Optional[List[str]] = None,
    duration_per_phase: Optional[Dict[str, float]] = None,
    operator_name: str = "",
    customer_name: str = "",
    customer_phone: str = "",
    customer_age: str = "",
    customer_gender: str = "",
    qualitative_feedback: str = "",
    patient_input=None,
    derived_variables=None,
    calibration_snapshot: Optional[list] = None,
) -> dict:
    duration = (session_end_time - session_start_time).total_seconds()
    manual_count = sum(1 for r in rows if r.get("interaction_type") == "Manual")
    qna_count = sum(1 for r in rows if r.get("interaction_type") == "QnA")

    derived_ar, derived_lenso = _derive_objective_payloads(patient_input)
    ar = ar or derived_ar
    lensometry = lensometry or derived_lenso

    final_rx = {}
    achieved_rx = {}
    current_rx = {}
    final_distance_va = {}
    final_rx_comparison = {}
    if rows:
        last = rows[-1]
        achieved_rx = _rx_payload(
            last.get("final_compare_achieved_re_sph", last.get("re_sph")),
            last.get("final_compare_achieved_re_cyl", last.get("re_cyl")),
            last.get("final_compare_achieved_re_axis", last.get("re_axis")),
            last.get("final_compare_achieved_add_r", last.get("add_r")),
            last.get("final_compare_achieved_le_sph", last.get("le_sph")),
            last.get("final_compare_achieved_le_cyl", last.get("le_cyl")),
            last.get("final_compare_achieved_le_axis", last.get("le_axis")),
            last.get("final_compare_achieved_add_l", last.get("add_l")),
        )
        current_rx = _rx_payload(
            last.get("final_compare_current_re_sph", _safe_get(lensometry, "right", "sph", default=None)),
            last.get("final_compare_current_re_cyl", _safe_get(lensometry, "right", "cyl", default=None)),
            last.get("final_compare_current_re_axis", _safe_get(lensometry, "right", "axis", default=None)),
            last.get("final_compare_current_add_r", _safe_get(lensometry, "right", "add", default=None)),
            last.get("final_compare_current_le_sph", _safe_get(lensometry, "left", "sph", default=None)),
            last.get("final_compare_current_le_cyl", _safe_get(lensometry, "left", "cyl", default=None)),
            last.get("final_compare_current_le_axis", _safe_get(lensometry, "left", "axis", default=None)),
            last.get("final_compare_current_add_l", _safe_get(lensometry, "left", "add", default=None)),
        )
        comparison_ran = bool(last.get("final_compare_enabled", False))
        accepted_achieved = last.get("patient_accepted_achieved_over_current_rx", "") == "Yes"
        if comparison_ran:
            if accepted_achieved and _payload_has_values(achieved_rx):
                final_rx = achieved_rx
                selected_source = "Achieved"
            elif _payload_has_values(current_rx):
                final_rx = current_rx
                selected_source = "PGP"
            elif _payload_has_values(achieved_rx):
                final_rx = achieved_rx
                selected_source = "Achieved"
            else:
                final_rx = _rx_payload(
                    last.get("re_sph"), last.get("re_cyl"), last.get("re_axis"), last.get("add_r"),
                    last.get("le_sph"), last.get("le_cyl"), last.get("le_axis"), last.get("add_l"),
                )
                selected_source = ""
        else:
            final_rx = _rx_payload(
                last.get("re_sph"), last.get("re_cyl"), last.get("re_axis"), last.get("add_r"),
                last.get("le_sph"), last.get("le_cyl"), last.get("le_axis"), last.get("add_l"),
            )
            selected_source = ""
        final_distance_va = {
            "right": {
                "chart": last.get("distance_va_re_chart", ""),
                "line": last.get("distance_va_re_line", ""),
            },
            "left": {
                "chart": last.get("distance_va_le_chart", ""),
                "line": last.get("distance_va_le_line", ""),
            },
        }
        final_rx_comparison = {
            "ran": comparison_ran,
            "current_source": "PGP" if comparison_ran else "",
            "option_1_source": "Achieved" if comparison_ran else "",
            "option_2_source": "PGP" if comparison_ran else "",
            "round_1_choice": last.get("final_compare_choice_round_1", ""),
            "round_2_choice": last.get("final_compare_choice_round_2", ""),
            "accepted_achieved_over_current_rx": last.get("patient_accepted_achieved_over_current_rx", ""),
            "selected_prescribed_rx_source": selected_source,
        }

    return {
        "session_id": session_id,
        "phoropter_id": phoropter_id,
        "operator_name": operator_name,
        "customer_name": customer_name or "",
        "customer_phone": customer_phone or "",
        "customer_age": customer_age or "",
        "customer_gender": customer_gender or "",
        "session_start_time": session_start_time.isoformat(),
        "session_end_time": session_end_time.isoformat(),
        "session_duration_seconds": round(duration, 1),
        "test_completion_status": completion_status,
        "total_interactions": len(rows),
        "ar": ar or {},
        "lensometry": lensometry or {},
        "final_prescription": final_rx,
        "achieved_prescription": achieved_rx,
        "pgp_rx": current_rx,
        "current_rx": current_rx,
        "final_distance_va": final_distance_va,
        "final_rx_comparison": final_rx_comparison,
        "qualitative_feedback": qualitative_feedback or "",
        "phases_completed": phases_completed or [],
        "phases_skipped": phases_skipped or [],
        "quality_metrics": {
            "manual_adjustment_count": manual_count,
            "qna_interaction_count": qna_count,
            "phase_jump_count": phase_jump_count,
            "unable_to_read_count": unable_to_read_count,
            "duration_per_phase": duration_per_phase or {},
        },
        "calibration": calibration_snapshot or [],
        "patient_input": _serialize_patient_input(patient_input),
        "derived_variables": _serialize_derived_variables(derived_variables),
    }
