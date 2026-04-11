"""Detect whether a customer has previously completed an eye test.

Used to choose between the 'first_test' and 'second_test' email templates.
Reads the eye test engine's combined_metadata.csv (maintained by ETE_v2/ete_io).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Path to the ETE_v2 sibling directory
_ETE_ROOT = Path(__file__).resolve().parent.parent / "ETE_v2"


def _metadata_csv_path() -> Path:
    """Resolve the path to combined_metadata.csv based on LOG_DIR env var."""
    log_dir_env = os.environ.get("LOG_DIR", "logs")
    log_dir = Path(log_dir_env)
    if not log_dir.is_absolute():
        log_dir = _ETE_ROOT / log_dir
    return log_dir / "combined_metadata.csv"


def has_completed_eye_test(phone: str) -> bool:
    """Returns True if this phone has any 'Completed' session in combined_metadata.csv."""
    if not phone:
        return False

    csv_path = _metadata_csv_path()
    if not csv_path.exists():
        log.debug("test_history: %s does not exist", csv_path)
        return False

    try:
        # Reuse the existing loader so the CSV format stays in sync with ETE_v2
        if str(_ETE_ROOT) not in sys.path:
            sys.path.insert(0, str(_ETE_ROOT))
        from ete_io.dashboard_data import load_metadata_rows  # type: ignore
        rows = load_metadata_rows(csv_path)
    except Exception as e:
        log.warning("test_history: failed to load metadata: %s", e)
        return False

    target = phone.strip()
    for r in rows:
        row_phone = (r.get("Customer_Phone") or "").strip()
        if row_phone != target:
            continue
        status = (r.get("Completion_Status") or "").strip().lower()
        if status == "completed":
            return True
    return False


def template_key_for(phone: str) -> str:
    """Returns 'second_test' if customer has a completed test, else 'first_test'."""
    return "second_test" if has_completed_eye_test(phone) else "first_test"
