"""
Output handling: Write session CSV logs, metadata JSON, and combined logs.
Adapted for ETE_v2 — uses FSMRuntimeRow instead of RowContext.
"""
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# 19-column schema matching v1
SESSION_CSV_FIELDS = [
    "Row_Number",
    "Timestamp",
    "Manual_or_QnA",
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
]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _row_to_dict(row: dict) -> dict:
    """Convert a SessionRow dict into a flat dict matching SESSION_CSV_FIELDS."""
    return {
        "Row_Number": row.get("row_number", ""),
        "Timestamp": row.get("timestamp", ""),
        "Manual_or_QnA": row.get("interaction_type", "QnA"),
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
    "Final_R_SPH", "Final_R_CYL", "Final_R_AXIS", "Final_R_ADD",
    "Final_L_SPH", "Final_L_CYL", "Final_L_AXIS", "Final_L_ADD",
    "Phases_Completed",
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
    ar = metadata.get("ar", {})
    lenso = metadata.get("lensometry", {})
    return {
        "Session_ID": metadata.get("session_id", ""),
        "Phoropter_ID": metadata.get("phoropter_id", ""),
        "Operator_Name": metadata.get("operator_name", ""),
        "Customer_Name": metadata.get("customer_name", ""),
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
        "Final_R_SPH": _safe_get(final, "right", "sph"),
        "Final_R_CYL": _safe_get(final, "right", "cyl"),
        "Final_R_AXIS": _safe_get(final, "right", "axis"),
        "Final_R_ADD": _safe_get(final, "right", "add"),
        "Final_L_SPH": _safe_get(final, "left", "sph"),
        "Final_L_CYL": _safe_get(final, "left", "cyl"),
        "Final_L_AXIS": _safe_get(final, "left", "axis"),
        "Final_L_ADD": _safe_get(final, "left", "add"),
        "Phases_Completed": "; ".join(metadata.get("phases_completed", [])),
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
    customer_age: str = "",
    customer_gender: str = "",
    qualitative_feedback: str = "",
) -> dict:
    duration = (session_end_time - session_start_time).total_seconds()
    manual_count = sum(1 for r in rows if r.get("interaction_type") == "Manual")
    qna_count = sum(1 for r in rows if r.get("interaction_type") == "QnA")

    final_rx = {}
    if rows:
        last = rows[-1]
        final_rx = {
            "right": {"sph": last.get("re_sph"), "cyl": last.get("re_cyl"),
                       "axis": last.get("re_axis"), "add": last.get("add_r")},
            "left": {"sph": last.get("le_sph"), "cyl": last.get("le_cyl"),
                      "axis": last.get("le_axis"), "add": last.get("add_l")},
        }

    return {
        "session_id": session_id,
        "phoropter_id": phoropter_id,
        "operator_name": operator_name,
        "customer_name": customer_name or "",
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
    }
