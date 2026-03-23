"""
Upload eye test session CSV and metadata to Supabase Storage.
Ported from v1 as-is.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def upload_session(session_id: str, csv_content: str, metadata: dict) -> Optional[str]:
    """
    Upload session CSV and metadata to the configured Supabase Storage bucket.

    Returns:
        None on success, or an error message string on failure.
    """
    backend = (os.environ.get("REMOTE_STORAGE") or "").strip().lower()
    if backend != "supabase":
        return None

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "").strip()
    bucket = os.environ.get("SUPABASE_BUCKET", "eye-test-sessions").strip()

    if not url or not key:
        return "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) are required for Supabase storage"

    try:
        from supabase import create_client
    except ImportError:
        return "Install supabase: pip install supabase"

    try:
        client = create_client(url, key)
        storage = client.storage.from_(bucket)

        csv_path = f"{session_id}.csv"
        storage.upload(
            csv_path,
            csv_content.encode("utf-8"),
            {"content-type": "text/csv", "upsert": "true"},
        )

        meta_path = f"{session_id}_metadata.json"
        meta_bytes = json.dumps(metadata, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        storage.upload(
            meta_path,
            meta_bytes,
            {"content-type": "application/json", "upsert": "true"},
        )

        return None
    except Exception as e:
        err_msg = str(e)
        if not err_msg and hasattr(e, "message"):
            err_msg = getattr(e, "message", "Unknown error")
        return err_msg or "Unknown Supabase error"
