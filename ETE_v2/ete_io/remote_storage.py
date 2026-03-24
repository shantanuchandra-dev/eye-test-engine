"""
Upload eye test session CSV and metadata to Supabase Storage.
Ported from v1 as-is.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple


def remote_supabase_remote_only() -> bool:
    """True when REMOTE_STORAGE=supabase: session logs are not written to local disk."""
    return (os.environ.get("REMOTE_STORAGE") or "").strip().lower() == "supabase"


def _supabase_upload_enabled() -> tuple[bool, bool]:
    """
    Returns (should_upload, explicit_supabase).

    Supabase uploads run only when REMOTE_STORAGE=supabase (opt-in).
    Having SUPABASE_* in the environment alone does not upload anything.

    Set REMOTE_STORAGE to none|local|off|false|0 (or leave empty) to keep logs local only.
    """
    remote = (os.environ.get("REMOTE_STORAGE") or "").strip().lower()
    if remote in ("none", "local", "off", "false", "0"):
        return False, False
    explicit = remote == "supabase"
    return explicit, explicit


def _download_storage_text(storage, object_name: str) -> Optional[str]:
    try:
        data = storage.download(object_name)
        if data is None:
            return None
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8")
        return str(data)
    except Exception:
        return None


def upload_session(
    session_id: str,
    csv_content: str,
    metadata: dict,
    failed_voice_csv_content: Optional[str] = None,
    combined_log_csv_content: Optional[str] = None,
    *,
    combined_log_merge: Optional[Tuple[str, List[dict]]] = None,
    combined_metadata_merge: Optional[dict] = None,
) -> Optional[str]:
    """
    Upload session CSV, metadata JSON, optional failed-voice CSV, and optional combined
    log / metadata CSVs to Supabase Storage.

    When local combined_log.csv exists, pass combined_log_csv_content. When logs are
    remote-only, pass combined_log_merge=(session_id, rows) and combined_metadata_merge
    to merge with existing Storage objects.

    Returns:
        None on success or when upload is skipped, or an error message on failure.
    """
    should_upload, explicit_supabase = _supabase_upload_enabled()
    if not should_upload:
        return None

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "").strip()
    bucket = os.environ.get("SUPABASE_BUCKET", "eye-test-sessions").strip()

    if not url or not key:
        if explicit_supabase:
            return "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) are required for Supabase storage"
        return None

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

        if failed_voice_csv_content:
            fva_name = f"{session_id}_failed_voice_attempts.csv"
            storage.upload(
                fva_name,
                failed_voice_csv_content.encode("utf-8"),
                {"content-type": "text/csv", "upsert": "true"},
            )

        if combined_log_merge:
            merge_sid, merge_rows = combined_log_merge
            if merge_rows:
                from ete_io.outputs import combined_log_rows_csv_string

                existing = _download_storage_text(storage, "combined_log.csv")
                has_existing = bool(existing and existing.strip())
                new_csv = combined_log_rows_csv_string(
                    merge_rows, merge_sid, include_header=not has_existing
                )
                merged = (
                    (existing.rstrip("\r\n") + "\n" + new_csv) if has_existing else new_csv
                )
                storage.upload(
                    "combined_log.csv",
                    merged.encode("utf-8"),
                    {"content-type": "text/csv", "upsert": "true"},
                )
        elif combined_log_csv_content is not None:
            storage.upload(
                "combined_log.csv",
                combined_log_csv_content.encode("utf-8"),
                {"content-type": "text/csv", "upsert": "true"},
            )

        if combined_metadata_merge is not None:
            from ete_io.outputs import combined_metadata_row_csv_string

            existing = _download_storage_text(storage, "combined_metadata.csv")
            has_existing = bool(existing and existing.strip())
            new_csv = combined_metadata_row_csv_string(
                combined_metadata_merge, include_header=not has_existing
            )
            merged = (
                (existing.rstrip("\r\n") + "\n" + new_csv) if has_existing else new_csv
            )
            storage.upload(
                "combined_metadata.csv",
                merged.encode("utf-8"),
                {"content-type": "text/csv", "upsert": "true"},
            )

        return None
    except Exception as e:
        err_msg = str(e)
        if not err_msg and hasattr(e, "message"):
            err_msg = getattr(e, "message", "Unknown error")
        return err_msg or "Unknown Supabase error"
