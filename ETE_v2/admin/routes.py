"""Flask Blueprint for the admin dashboard — session analytics, Manual Rx, accuracy."""
from __future__ import annotations

import csv
import io
import json
import os
import secrets
import time
from datetime import date
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_from_directory

from admin import db as admin_db
from admin.data import (
    DEFAULT_THRESHOLDS,
    compute_stats,
    filter_sessions,
    get_date_range,
    load_all_metadata,
    merge_sessions_with_rx,
)

ADMIN_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
ADMIN_PASSWORD = os.environ.get("BOOKING_ADMIN_PASSWORD")

admin_bp = Blueprint("admin_dashboard", __name__)

# ── Admin session tokens (shared pattern with booking admin) ──
_ADMIN_TOKENS: dict[str, float] = {}
_ADMIN_TOKEN_TTL = 24 * 60 * 60


def _issue_admin_token() -> str:
    token = secrets.token_urlsafe(32)
    _ADMIN_TOKENS[token] = time.time() + _ADMIN_TOKEN_TTL
    now = time.time()
    for t in [t for t, exp in _ADMIN_TOKENS.items() if exp < now]:
        _ADMIN_TOKENS.pop(t, None)
    return token


def _is_valid_admin_token(token: str) -> bool:
    if not token:
        return False
    expiry = _ADMIN_TOKENS.get(token)
    if expiry is None:
        return False
    if expiry < time.time():
        _ADMIN_TOKENS.pop(token, None)
        return False
    return True


def _json() -> dict:
    return request.get_json(force=True, silent=True) or {}


def _require_admin():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if _is_valid_admin_token(token):
            return None
        return jsonify({"error": "Session expired. Please log in again."}), 401
    return jsonify({"error": "Unauthorized"}), 401


def _get_metadata_path() -> Path:
    from api_server import COMBINED_METADATA_PATH
    return COMBINED_METADATA_PATH


def _get_thresholds() -> dict:
    """Load thresholds from config file or return defaults."""
    config_path = Path(__file__).resolve().parent / "thresholds.json"
    if config_path.exists():
        try:
            return {**DEFAULT_THRESHOLDS, **json.loads(config_path.read_text())}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_THRESHOLDS)


def _save_thresholds(thresholds: dict):
    config_path = Path(__file__).resolve().parent / "thresholds.json"
    config_path.write_text(json.dumps(thresholds, indent=2))


# ═══════════════════════════════════════════════════════════════
# Page routes
# ═══════════════════════════════════════════════════════════════

@admin_bp.route("/admin/dashboard")
def serve_admin_dashboard():
    return send_from_directory(ADMIN_FRONTEND_DIR, "admin-dashboard.html")


# ═══════════════════════════════════════════════════════════════
# Auth
# ═══════════════════════════════════════════════════════════════

@admin_bp.route("/api/admin/dashboard/verify", methods=["POST"])
def admin_dashboard_verify():
    data = _json()
    if data.get("password") != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid password"}), 403
    token = _issue_admin_token()
    return jsonify({"ok": True, "token": token, "expires_in": _ADMIN_TOKEN_TTL})


@admin_bp.route("/api/admin/dashboard/logout", methods=["POST"])
def admin_dashboard_logout():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        _ADMIN_TOKENS.pop(auth_header[7:].strip(), None)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════
# Sessions & Stats
# ═══════════════════════════════════════════════════════════════

@admin_bp.route("/api/admin/dashboard/sessions")
def admin_sessions():
    err = _require_admin()
    if err:
        return err

    period = request.args.get("period", "today")
    from_str = request.args.get("from")
    to_str = request.args.get("to")
    search = request.args.get("search", "").strip()
    rx_filter = request.args.get("rx_filter", "all")  # all | filled | empty
    phoropters_str = request.args.get("phoropters", "").strip()
    phoropters = [p.strip() for p in phoropters_str.split(",") if p.strip()] or None
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))

    # Date range
    if from_str and to_str:
        from_date = date.fromisoformat(from_str)
        to_date = date.fromisoformat(to_str)
    else:
        from_date, to_date = get_date_range(period)

    # Load session rows — only fetch files in the requested date range
    force_refresh = request.args.get("refresh") == "1"
    rows = load_all_metadata(_get_metadata_path(), from_date, to_date, force_refresh)

    # Extract distinct phoropter IDs (before filtering) for the multi-select
    all_phoropters = sorted(set(
        r.get("Phoropter_ID", "").strip()
        for r in rows if r.get("Phoropter_ID", "").strip()
    ))

    filtered = filter_sessions(rows, from_date, to_date, search or None, phoropters)

    # Reverse so most recent first
    filtered.reverse()

    # Get all session IDs for this filtered set
    session_ids = [r.get("Session_ID", "") for r in filtered if r.get("Session_ID")]

    # Fetch manual Rx from Supabase
    manual_rx_list = admin_db.list_manual_rx(session_ids if session_ids else None)
    manual_rx_map = {rx["session_id"]: rx for rx in manual_rx_list}

    thresholds = _get_thresholds()
    merged = merge_sessions_with_rx(filtered, manual_rx_map, thresholds)

    # Apply rx_filter
    if rx_filter == "filled":
        merged = [m for m in merged if m["has_manual_rx"]]
    elif rx_filter == "empty":
        merged = [m for m in merged if not m["has_manual_rx"]]

    # Compute stats on the full filtered set (before pagination)
    stats = compute_stats(merged)

    # Paginate
    total = len(merged)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    page_items = merged[start:start + per_page]

    return jsonify({
        "stats": stats,
        "sessions": page_items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
        "thresholds": thresholds,
        "phoropters": all_phoropters,
    })


# ═══════════════════════════════════════════════════════════════
# Manual Rx CRUD
# ═══════════════════════════════════════════════════════════════

@admin_bp.route("/api/admin/dashboard/manual-rx/<session_id>", methods=["GET"])
def admin_get_manual_rx(session_id):
    err = _require_admin()
    if err:
        return err
    rx = admin_db.get_manual_rx(session_id)
    if not rx:
        return jsonify({"manual_rx": None})
    return jsonify({"manual_rx": rx})


@admin_bp.route("/api/admin/dashboard/manual-rx/<session_id>", methods=["PUT"])
def admin_upsert_manual_rx(session_id):
    err = _require_admin()
    if err:
        return err
    data = _json()
    rx = admin_db.upsert_manual_rx(session_id, data)
    return jsonify({"ok": True, "manual_rx": rx})


@admin_bp.route("/api/admin/dashboard/manual-rx/<session_id>", methods=["DELETE"])
def admin_delete_manual_rx(session_id):
    err = _require_admin()
    if err:
        return err
    admin_db.delete_manual_rx(session_id)
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════
# Thresholds (settings)
# ═══════════════════════════════════════════════════════════════

@admin_bp.route("/api/admin/dashboard/thresholds", methods=["GET"])
def admin_get_thresholds():
    err = _require_admin()
    if err:
        return err
    return jsonify(_get_thresholds())


@admin_bp.route("/api/admin/dashboard/thresholds", methods=["PUT"])
def admin_set_thresholds():
    err = _require_admin()
    if err:
        return err
    data = _json()
    current = _get_thresholds()
    for key in DEFAULT_THRESHOLDS:
        if key in data:
            try:
                current[key] = float(data[key])
            except (TypeError, ValueError):
                pass
    _save_thresholds(current)
    return jsonify(current)


# ═══════════════════════════════════════════════════════════════
# CSV Export
# ═══════════════════════════════════════════════════════════════

@admin_bp.route("/api/admin/dashboard/export")
def admin_export_csv():
    err = _require_admin()
    if err:
        return err

    period = request.args.get("period", "all")
    from_str = request.args.get("from")
    to_str = request.args.get("to")
    search = request.args.get("search", "").strip()
    rx_filter = request.args.get("rx_filter", "all")
    phoropters_str = request.args.get("phoropters", "").strip()
    phoropters = [p.strip() for p in phoropters_str.split(",") if p.strip()] or None

    if from_str and to_str:
        from_date = date.fromisoformat(from_str)
        to_date = date.fromisoformat(to_str)
    else:
        from_date, to_date = get_date_range(period)

    rows = load_all_metadata(_get_metadata_path(), from_date, to_date)
    filtered = filter_sessions(rows, from_date, to_date, search or None, phoropters)
    filtered.reverse()

    session_ids = [r.get("Session_ID", "") for r in filtered if r.get("Session_ID")]
    manual_rx_list = admin_db.list_manual_rx(session_ids if session_ids else None)
    manual_rx_map = {rx["session_id"]: rx for rx in manual_rx_list}

    thresholds = _get_thresholds()
    merged = merge_sessions_with_rx(filtered, manual_rx_map, thresholds)

    if rx_filter == "filled":
        merged = [m for m in merged if m["has_manual_rx"]]
    elif rx_filter == "empty":
        merged = [m for m in merged if not m["has_manual_rx"]]

    cols = [
        "session_id", "customer_name", "customer_phone", "phoropter_id",
        "start_time", "duration_seconds", "completion_status",
        "ai_r_sph", "ai_r_cyl", "ai_r_axis", "ai_r_add",
        "ai_l_sph", "ai_l_cyl", "ai_l_axis", "ai_l_add",
        "manual_r_sph", "manual_r_cyl", "manual_r_axis", "manual_r_add",
        "manual_l_sph", "manual_l_cyl", "manual_l_axis", "manual_l_add",
        "accurate",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for m in merged:
        flat = {k: m.get(k, "") for k in cols}
        if m.get("manual_rx"):
            rx = m["manual_rx"]
            flat["manual_r_sph"] = rx.get("r_sph", "")
            flat["manual_r_cyl"] = rx.get("r_cyl", "")
            flat["manual_r_axis"] = rx.get("r_axis", "")
            flat["manual_r_add"] = rx.get("r_add", "")
            flat["manual_l_sph"] = rx.get("l_sph", "")
            flat["manual_l_cyl"] = rx.get("l_cyl", "")
            flat["manual_l_axis"] = rx.get("l_axis", "")
            flat["manual_l_add"] = rx.get("l_add", "")
        writer.writerow(flat)

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=admin_dashboard_export.csv"},
    )
