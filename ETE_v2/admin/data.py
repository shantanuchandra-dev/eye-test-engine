"""
Admin dashboard data processing — merges CSV metadata with Manual Rx from Supabase,
computes deviation & accuracy metrics.
"""
from __future__ import annotations

import csv
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ete_io.dashboard_data import load_metadata_rows, _parse_date, _to_float
from ete_io.ist_time import ist_now


# ── Default accuracy thresholds (Indian optometry standards) ──

DEFAULT_THRESHOLDS = {
    "sph_tolerance": 0.25,      # ±0.25 D
    "cyl_tolerance": 0.25,      # ±0.25 D
    "axis_tolerance": 5,        # ±5 degrees
    "axis_cyl_min": 0.75,       # Axis only matters when |cyl| >= this
    "add_tolerance": 0.25,      # ±0.25 D
}

# ── In-memory cache — keyed by date range string, e.g. "2026-04-15|2026-04-15" ──
_metadata_cache: Dict[str, Any] = {}  # {range_key: {"rows": [...], "ts": float}}
_CACHE_TTL = 60  # seconds


def _get_supabase_storage():
    """Returns (storage_handle, bucket_name) or (None, None)."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "").strip()
    bucket = os.environ.get("SUPABASE_BUCKET", "Eye_Test_logs").strip()
    if not url or not key:
        return None, None
    try:
        from supabase import create_client
        client = create_client(url, key)
        return client.storage.from_(bucket), bucket
    except Exception:
        return None, None


def _json_metadata_to_flat(meta: dict) -> Dict[str, Any]:
    """Convert a *_metadata.json dict to the flat CSV-row format the dashboard expects.

    Reuses the same mapping as ete_io.outputs.build_combined_metadata_flat but
    is self-contained so admin/ has no coupling to the upload code.
    """
    def _g(d, *keys):
        for k in keys:
            if d is None or not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    qm = meta.get("quality_metrics") or {}
    final = meta.get("final_prescription") or {}
    achieved = meta.get("achieved_prescription") or {}
    ar = meta.get("ar") or {}
    lenso = meta.get("lensometry") or {}

    return {
        "Session_ID": meta.get("session_id", ""),
        "Phoropter_ID": meta.get("phoropter_id", ""),
        "Operator_Name": meta.get("operator_name", ""),
        "Customer_Name": meta.get("customer_name", ""),
        "Customer_Phone": meta.get("customer_phone", ""),
        "Customer_Age": meta.get("customer_age", ""),
        "Customer_Gender": meta.get("customer_gender", ""),
        "Start_Time": meta.get("session_start_time", ""),
        "End_Time": meta.get("session_end_time", ""),
        "Duration_Seconds": meta.get("session_duration_seconds", ""),
        "Completion_Status": meta.get("test_completion_status", ""),
        "Total_Interactions": meta.get("total_interactions", ""),
        "Manual_Count": qm.get("manual_adjustment_count", 0),
        "QnA_Count": qm.get("qna_interaction_count", 0),
        "Phase_Jump_Count": qm.get("phase_jump_count", 0),
        "Unable_To_Read_Count": qm.get("unable_to_read_count", 0),
        "AR_R_SPH": _g(ar, "right", "sph"),
        "AR_R_CYL": _g(ar, "right", "cyl"),
        "AR_R_AXIS": _g(ar, "right", "axis"),
        "AR_L_SPH": _g(ar, "left", "sph"),
        "AR_L_CYL": _g(ar, "left", "cyl"),
        "AR_L_AXIS": _g(ar, "left", "axis"),
        "Achieved_R_SPH": _g(achieved, "right", "sph"),
        "Achieved_R_CYL": _g(achieved, "right", "cyl"),
        "Achieved_R_AXIS": _g(achieved, "right", "axis"),
        "Achieved_R_ADD": _g(achieved, "right", "add"),
        "Achieved_L_SPH": _g(achieved, "left", "sph"),
        "Achieved_L_CYL": _g(achieved, "left", "cyl"),
        "Achieved_L_AXIS": _g(achieved, "left", "axis"),
        "Achieved_L_ADD": _g(achieved, "left", "add"),
        "Final_R_SPH": _g(final, "right", "sph"),
        "Final_R_CYL": _g(final, "right", "cyl"),
        "Final_R_AXIS": _g(final, "right", "axis"),
        "Final_R_ADD": _g(final, "right", "add"),
        "Final_L_SPH": _g(final, "left", "sph"),
        "Final_L_CYL": _g(final, "left", "cyl"),
        "Final_L_AXIS": _g(final, "left", "axis"),
        "Final_L_ADD": _g(final, "left", "add"),
        "Phases_Completed": "; ".join(meta.get("phases_completed") or []),
    }


def _list_all_bucket_files(storage) -> List[dict]:
    """Paginate through all files in the bucket root."""
    all_files: List[dict] = []
    offset = 0
    limit = 200
    while True:
        batch = storage.list("", {"limit": limit, "offset": offset,
                                   "sortBy": {"column": "created_at", "order": "desc"}})
        if not batch:
            break
        all_files.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return all_files


def _parse_utc_date(ts: str) -> Optional[date]:
    """Parse a UTC ISO timestamp like '2026-04-14T22:47:13.960Z' to an IST date."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # Convert UTC to IST (UTC+5:30) since sessions happen in India
        ist = dt + timedelta(hours=5, minutes=30)
        return ist.date()
    except Exception:
        return None


def load_metadata_from_supabase(
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Load session metadata JSONs from the bucket, filtered by date range.

    Only downloads files whose created_at falls within [from_date, to_date+1day]
    (to account for UTC vs IST offset). Results are cached per date-range key.
    """
    import time as _time

    cache_key = f"{from_date or 'none'}|{to_date or 'none'}"
    now = _time.time()
    cached = _metadata_cache.get(cache_key)
    if not force_refresh and cached and (now - cached["ts"]) < _CACHE_TTL:
        return cached["rows"]

    storage, _bucket = _get_supabase_storage()
    if storage is None:
        return []

    try:
        import json as _json

        all_files = _list_all_bucket_files(storage)

        # Filter to *_metadata.json files, then by date if given
        meta_files = []
        for f in all_files:
            if not f["name"].endswith("_metadata.json"):
                continue
            if from_date or to_date:
                file_date = _parse_utc_date(f.get("created_at", ""))
                if file_date is None:
                    continue
                # Include a 1-day buffer on each side for UTC/IST boundary
                if from_date and file_date < from_date - timedelta(days=1):
                    continue
                if to_date and file_date > to_date + timedelta(days=1):
                    continue
            meta_files.append(f["name"])

        def _download_one(fname):
            try:
                raw = storage.download(fname)
                if raw is None:
                    return None
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                meta = _json.loads(text)
                flat = _json_metadata_to_flat(meta)
                return flat if flat.get("Session_ID") else None
            except Exception:
                return None

        rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_download_one, f): f for f in meta_files}
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    rows.append(result)

        _metadata_cache[cache_key] = {"rows": rows, "ts": now}
        return rows
    except Exception as e:
        print(f"[admin] Failed to load metadata from Supabase Storage: {e}")
        return cached["rows"] if cached else []


def _deduplicate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only one row per Session_ID (last occurrence wins)."""
    seen: Dict[str, int] = {}
    for i, r in enumerate(rows):
        sid = r.get("Session_ID", "")
        if sid:
            seen[sid] = i
    return [rows[i] for i in sorted(seen.values())]


def load_all_metadata(
    local_path: Path,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Load metadata from Supabase (date-filtered), fall back to local CSV."""
    rows = load_metadata_from_supabase(from_date, to_date, force_refresh)
    if not rows:
        rows = load_metadata_rows(local_path)
    return _deduplicate(rows)


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "" or val == "—":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _ai_field(row: dict, eye: str, param: str) -> Optional[float]:
    """Read AI power from CSV, trying Achieved_* first then Final_*."""
    val = row.get(f"Achieved_{eye}_{param}")
    if val is None or val == "":
        val = row.get(f"Final_{eye}_{param}")
    return _safe_float(val)


def _axis_diff(a1: Optional[float], a2: Optional[float]) -> Optional[float]:
    """Circular axis difference (0–180 degree scale)."""
    if a1 is None or a2 is None:
        return None
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)


def compute_eye_deviation(
    manual: dict,
    ai_sph: Optional[float],
    ai_cyl: Optional[float],
    ai_axis: Optional[float],
    prefix_m: str,
    thresholds: dict,
) -> dict:
    """Compute deviation for one eye (R or L).

    Returns dict with sph_dev, cyl_dev, axis_dev, within_tolerance (bool).
    """
    m_sph = _safe_float(manual.get(f"{prefix_m}_sph"))
    m_cyl = _safe_float(manual.get(f"{prefix_m}_cyl"))
    m_axis = _safe_float(manual.get(f"{prefix_m}_axis"))

    a_sph = ai_sph
    a_cyl = ai_cyl
    a_axis = ai_axis

    sph_dev = abs(m_sph - a_sph) if m_sph is not None and a_sph is not None else None
    cyl_dev = abs(m_cyl - a_cyl) if m_cyl is not None and a_cyl is not None else None
    axis_dev = _axis_diff(m_axis, a_axis)

    # Determine if within tolerance
    within = True
    if sph_dev is not None and sph_dev > thresholds["sph_tolerance"]:
        within = False
    if cyl_dev is not None and cyl_dev > thresholds["cyl_tolerance"]:
        within = False
    # Axis only checked if cylinder is significant
    cyl_magnitude = max(abs(m_cyl or 0), abs(a_cyl or 0))
    if axis_dev is not None and cyl_magnitude >= thresholds["axis_cyl_min"]:
        if axis_dev > thresholds["axis_tolerance"]:
            within = False

    return {
        "sph_dev": round(sph_dev, 2) if sph_dev is not None else None,
        "cyl_dev": round(cyl_dev, 2) if cyl_dev is not None else None,
        "axis_dev": round(axis_dev, 1) if axis_dev is not None else None,
        "within_tolerance": within,
    }


def compute_session_deviation(
    manual_rx: dict,
    ai_powers: dict,
    thresholds: dict,
) -> dict:
    """Compute deviation for both eyes of a session.

    ai_powers should have keys: ai_r_sph, ai_r_cyl, ai_r_axis, ai_l_sph, etc.
    """
    r_dev = compute_eye_deviation(
        manual_rx, ai_powers.get("ai_r_sph"), ai_powers.get("ai_r_cyl"),
        ai_powers.get("ai_r_axis"), "r", thresholds,
    )
    l_dev = compute_eye_deviation(
        manual_rx, ai_powers.get("ai_l_sph"), ai_powers.get("ai_l_cyl"),
        ai_powers.get("ai_l_axis"), "l", thresholds,
    )

    both_within = r_dev["within_tolerance"] and l_dev["within_tolerance"]

    return {
        "right": r_dev,
        "left": l_dev,
        "accurate": both_within,
    }


def get_date_range(period: str) -> tuple[Optional[date], Optional[date]]:
    """Convert period string to (from_date, to_date)."""
    today = ist_now().date()
    if period == "today":
        return today, today
    elif period == "week":
        return today - timedelta(days=today.weekday()), today
    elif period == "month":
        return today.replace(day=1), today
    elif period == "all":
        return None, None
    return None, None


def filter_sessions(
    rows: list[dict],
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    search: Optional[str] = None,
    phoropters: Optional[list[str]] = None,
) -> list[dict]:
    """Filter metadata rows by date range, search term, and phoropter IDs."""
    out = []
    search_lower = search.strip().lower() if search else None
    phoropter_set = set(phoropters) if phoropters else None

    for r in rows:
        start_d = _parse_date(r.get("Start_Time") or "")
        if from_date and (start_d is None or start_d < from_date):
            continue
        if to_date and (start_d is None or start_d > to_date):
            continue
        if phoropter_set and (r.get("Phoropter_ID") or "").strip() not in phoropter_set:
            continue
        if search_lower:
            searchable = " ".join([
                r.get("Session_ID", ""),
                r.get("Customer_Name", ""),
                r.get("Customer_Phone", ""),
                r.get("Phoropter_ID", ""),
                r.get("Operator_Name", ""),
            ]).lower()
            if search_lower not in searchable:
                continue
        out.append(r)
    return out


def merge_sessions_with_rx(
    rows: list[dict],
    manual_rx_map: dict[str, dict],
    thresholds: dict,
) -> list[dict]:
    """Merge CSV rows with manual Rx data and compute deviations."""
    merged = []
    for r in rows:
        sid = r.get("Session_ID", "")
        mrx = manual_rx_map.get(sid)
        entry = {
            "session_id": sid,
            "customer_name": r.get("Customer_Name", ""),
            "customer_phone": r.get("Customer_Phone", ""),
            "phoropter_id": r.get("Phoropter_ID", ""),
            "operator_name": r.get("Operator_Name", ""),
            "start_time": r.get("Start_Time", ""),
            "duration_seconds": _to_float(r.get("Duration_Seconds")),
            "completion_status": r.get("Completion_Status", ""),
            # AI power (try Achieved_* first, fall back to Final_*)
            "ai_r_sph": _ai_field(r, "R", "SPH"),
            "ai_r_cyl": _ai_field(r, "R", "CYL"),
            "ai_r_axis": _ai_field(r, "R", "AXIS"),
            "ai_r_add": _ai_field(r, "R", "ADD"),
            "ai_l_sph": _ai_field(r, "L", "SPH"),
            "ai_l_cyl": _ai_field(r, "L", "CYL"),
            "ai_l_axis": _ai_field(r, "L", "AXIS"),
            "ai_l_add": _ai_field(r, "L", "ADD"),
            # Manual Rx
            "manual_rx": None,
            "has_manual_rx": False,
            "deviation": None,
            "accurate": None,
        }

        if mrx:
            entry["manual_rx"] = {
                "r_sph": _safe_float(mrx.get("r_sph")),
                "r_cyl": _safe_float(mrx.get("r_cyl")),
                "r_axis": _safe_float(mrx.get("r_axis")),
                "r_add": _safe_float(mrx.get("r_add")),
                "l_sph": _safe_float(mrx.get("l_sph")),
                "l_cyl": _safe_float(mrx.get("l_cyl")),
                "l_axis": _safe_float(mrx.get("l_axis")),
                "l_add": _safe_float(mrx.get("l_add")),
                "entered_by": mrx.get("entered_by", ""),
            }
            entry["has_manual_rx"] = True
            ai_powers = {k: entry[k] for k in (
                "ai_r_sph", "ai_r_cyl", "ai_r_axis",
                "ai_l_sph", "ai_l_cyl", "ai_l_axis",
            )}
            entry["deviation"] = compute_session_deviation(mrx, ai_powers, thresholds)
            entry["accurate"] = entry["deviation"]["accurate"]

        merged.append(entry)

    return merged


def compute_stats(merged: list[dict]) -> dict:
    """Compute aggregate stats from merged session list."""
    total = len(merged)
    rx_done = sum(1 for m in merged if m["has_manual_rx"])
    rx_pending = total - rx_done

    # Accuracy among sessions that have Manual Rx
    with_rx = [m for m in merged if m["has_manual_rx"] and m["accurate"] is not None]
    accurate_count = sum(1 for m in with_rx if m["accurate"])
    avg_accuracy = (accurate_count / len(with_rx) * 100) if with_rx else None
    high_deviation = sum(1 for m in with_rx if not m["accurate"])

    return {
        "total_tests": total,
        "manual_rx_done": rx_done,
        "pending_rx": rx_pending,
        "avg_accuracy": round(avg_accuracy, 1) if avg_accuracy is not None else None,
        "high_deviation": high_deviation,
    }
