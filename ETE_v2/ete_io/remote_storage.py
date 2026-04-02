"""
Upload eye test session CSV and metadata to Supabase Storage.
Ported from v1 as-is.

Voice clips: saved under LOG_DIR/sessions/audio during the session; when
REMOTE_STORAGE=supabase or SUPABASE_VOICE_UPLOAD=1, they are uploaded in one
batch at session end (see batch_upload_session_voice_clips), not per answer.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


def remote_supabase_remote_only() -> bool:
    """True when REMOTE_STORAGE=supabase: session logs are not written to local disk."""
    return (os.environ.get("REMOTE_STORAGE") or "").strip().lower() == "supabase"


def supabase_voice_upload_enabled() -> bool:
    """
    When True, successful / failed patient mic clips are uploaded to Supabase Storage.

    Enabled if REMOTE_STORAGE=supabase, or explicitly via SUPABASE_VOICE_UPLOAD=1|true|yes.
    Disabled if SUPABASE_VOICE_UPLOAD=0|false|no|off (even when REMOTE_STORAGE=supabase).
    """
    voice = (os.environ.get("SUPABASE_VOICE_UPLOAD") or "").strip().lower()
    if voice in ("0", "false", "no", "off"):
        return False
    if voice in ("1", "true", "yes"):
        return True
    remote = (os.environ.get("REMOTE_STORAGE") or "").strip().lower()
    return remote == "supabase"


def _supabase_client_optional():
    """Returns (client, error_message) or (None, error) if misconfigured."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None, "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_KEY) are required"
    try:
        from supabase import create_client
    except ImportError:
        return None, "Install supabase: pip install supabase"
    try:
        return create_client(url, key), ""
    except Exception as e:
        return None, str(e) or "Supabase client init failed"


def upload_voice_bytes_to_bucket(
    *,
    bucket: str,
    object_path: str,
    data: bytes,
    content_type: str = "audio/webm",
) -> Optional[str]:
    """
    Upload raw audio bytes to a Storage bucket.

    Returns:
        None on success or when voice upload is disabled (skip).
        Error string if upload was attempted but failed.
    """
    if not supabase_voice_upload_enabled():
        return None
    client, err = _supabase_client_optional()
    if client is None:
        return err
    try:
        storage = client.storage.from_(bucket)
        storage.upload(
            object_path,
            data,
            {"content-type": content_type, "upsert": "true"},
        )
        return None
    except Exception as e:
        msg = str(e)
        if not msg and hasattr(e, "message"):
            msg = getattr(e, "message", "Unknown error")
        return msg or "Unknown Supabase upload error"


VOICE_MATCHED_INTENTS_CSV_NAME = "voice_matched_intents.csv"

_VOICE_INTENT_FIELDNAMES = (
    "audio_filename",
    "storage_object_path",
    "matched_intent",
    "transcript",
    "chart_characters",
)


def _voice_stimulus_letters_str(rec: dict[str, Any]) -> str:
    raw = rec.get("stimulus_letters")
    if isinstance(raw, list):
        return "; ".join(str(x) for x in raw).strip()
    if raw is None:
        return ""
    return str(raw).strip()


def _voice_chart_characters(rec: dict[str, Any]) -> str:
    """Letters shown on the chart when present; else chart type/name (e.g. snellen)."""
    letters = _voice_stimulus_letters_str(rec)
    if letters:
        return letters
    return str(rec.get("chart_display") or rec.get("chart_type") or "").strip()


def build_success_voice_intents_csv(session_id: str, session_history: Sequence[dict[str, Any]]) -> str:
    """
    One row per successful voice answer that has a saved success audio file.

    ``matched_intent`` is the FSM ``response_value`` recorded for that step.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_VOICE_INTENT_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    for rec in session_history:
        if not isinstance(rec, dict):
            continue
        af = (rec.get("audio_file") or "").strip()
        if not af or "_failed_" in af:
            continue
        if "_step" not in af:
            continue
        writer.writerow(
            {
                "audio_filename": af,
                "storage_object_path": f"{session_id}/{af}",
                "matched_intent": rec.get("response_value", ""),
                "transcript": rec.get("transcript", ""),
                "chart_characters": _voice_chart_characters(rec),
            }
        )
    return buf.getvalue()


def _audio_content_type_for_suffix(suffix: str) -> str:
    s = (suffix or "").lower().lstrip(".")
    return {
        "webm": "audio/webm",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
    }.get(s, "application/octet-stream")


def batch_upload_session_voice_clips(
    session_id: str,
    audio_dir: Path | str,
    session_history: Optional[Sequence[dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Upload all voice clips for a session from the local audio_dir to Supabase.

    Success answers: filenames like ``{session_id}_step{N}_*.webm`` → success_voices bucket.
    Failed clips: filenames starting with ``{session_id}_failed_`` → failed bucket.

    Also uploads ``{session_id}/voice_matched_intents.csv`` to the success bucket
    (audio, intent, transcript, chart_characters).

    Returns None on success or when voice upload is disabled. If ``audio_dir`` is missing
    but ``session_history`` is provided, only the intents CSV is uploaded.

    Returns a combined error string if any upload fails.
    """
    if not supabase_voice_upload_enabled():
        return None
    path = Path(audio_dir)
    dir_ok = path.is_dir()

    success_bucket = (os.environ.get("SUPABASE_SUCCESS_VOICES_BUCKET") or "success_voices").strip()
    failed_bucket = (os.environ.get("SUPABASE_FAILED_VOICES_BUCKET") or "failed_videos").strip()
    prefix = f"{session_id}_"
    errors: list[str] = []

    if not dir_ok and session_history is None:
        return None

    if dir_ok:
        for fp in sorted(path.iterdir()):
            if not fp.is_file():
                continue
            name = fp.name
            if not name.startswith(prefix):
                continue
            try:
                data = fp.read_bytes()
            except OSError as e:
                errors.append(f"{name}: read {e}")
                continue
            if not data:
                continue

            if name.startswith(f"{session_id}_failed_"):
                bucket = failed_bucket
            elif "_step" in name:
                bucket = success_bucket
            else:
                continue

            object_path = f"{session_id}/{name}"
            ctype = _audio_content_type_for_suffix(fp.suffix)
            err = upload_voice_bytes_to_bucket(
                bucket=bucket, object_path=object_path, data=data, content_type=ctype
            )
            if err:
                errors.append(f"{name}: {err}")

    if session_history is not None:
        manifest_csv = build_success_voice_intents_csv(session_id, session_history)
        manifest_path = f"{session_id}/{VOICE_MATCHED_INTENTS_CSV_NAME}"
        err_m = upload_voice_bytes_to_bucket(
            bucket=success_bucket,
            object_path=manifest_path,
            data=manifest_csv.encode("utf-8"),
            content_type="text/csv; charset=utf-8",
        )
        if err_m:
            errors.append(f"{VOICE_MATCHED_INTENTS_CSV_NAME}: {err_m}")

    return "; ".join(errors) if errors else None


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
