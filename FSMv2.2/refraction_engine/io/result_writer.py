from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from typing import Optional, Tuple, List, Dict


def generate_timestamp_id(prefix: Optional[str] = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if prefix:
        return f"{prefix}_{ts}"
    return ts


def create_run_folder(results_root: str, simulation_type: str, run_id: Optional[str] = None) -> Tuple[Path, str]:

    if run_id is None:
        run_id = generate_timestamp_id()

    folder = Path(results_root) / simulation_type / run_id
    folder.mkdir(parents=True, exist_ok=True)

    return folder, run_id


def save_trace_csv(rows: List[Dict], folder: Path, filename: str = "trace.csv") -> Path:

    df = pd.DataFrame(rows)

    path = folder / filename

    df.to_csv(path, index=False)

    return path


def save_dataframe_csv(df: pd.DataFrame, folder: Path, filename: str) -> Path:

    path = folder / filename

    df.to_csv(path, index=False)

    return path


def save_json(data: Dict, folder: Path, filename: str = "summary.json") -> Path:

    path = folder / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path