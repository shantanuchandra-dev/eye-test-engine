"""Supabase queries for the admin dashboard — Manual Rx CRUD."""
from __future__ import annotations

import os
from typing import Optional


def _get_client():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
    from supabase import create_client
    return create_client(url, key)


# ── Manual Rx ─────────────────────────────────────────────────

def get_manual_rx(session_id: str) -> Optional[dict]:
    client = _get_client()
    resp = client.table("manual_rx").select("*").eq("session_id", session_id).execute()
    return resp.data[0] if resp.data else None


def list_manual_rx(session_ids: list[str] | None = None) -> list[dict]:
    client = _get_client()
    q = client.table("manual_rx").select("*")
    if session_ids:
        q = q.in_("session_id", session_ids)
    resp = q.execute()
    return resp.data


def upsert_manual_rx(session_id: str, data: dict) -> dict:
    client = _get_client()
    row = {
        "session_id": session_id,
        "r_sph": data.get("r_sph"),
        "r_cyl": data.get("r_cyl"),
        "r_axis": data.get("r_axis"),
        "r_add": data.get("r_add"),
        "l_sph": data.get("l_sph"),
        "l_cyl": data.get("l_cyl"),
        "l_axis": data.get("l_axis"),
        "l_add": data.get("l_add"),
        "entered_by": data.get("entered_by", ""),
    }
    resp = client.table("manual_rx").upsert(row, on_conflict="session_id").execute()
    return resp.data[0] if resp.data else row


def delete_manual_rx(session_id: str) -> bool:
    client = _get_client()
    resp = client.table("manual_rx").delete().eq("session_id", session_id).execute()
    return bool(resp.data)
