from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[2]


def _resolve_results_root(results_root: str | Path) -> Path:
    path = Path(results_root).expanduser()
    if path.is_absolute():
        return path
    return APP_ROOT / path


def create_run_folder(results_root: str | Path, prefix: str) -> tuple[Path, str]:
    root = _resolve_results_root(results_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder, run_id


def save_dataframe_csv(df: pd.DataFrame, results_folder: str | Path, filename: str) -> Path:
    path = Path(results_folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(payload: dict, results_folder: str | Path, filename: str) -> Path:
    path = Path(results_folder) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def save_trace_csv(rows: Iterable[dict], results_folder: str | Path, filename: str) -> Path:
    df = pd.DataFrame(list(rows))
    return save_dataframe_csv(df, results_folder, filename)
