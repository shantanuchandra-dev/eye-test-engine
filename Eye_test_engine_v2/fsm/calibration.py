"""Calibration loader using stdlib csv (no pandas dependency)."""
import csv
from pathlib import Path
from typing import Any, Union


class CalibrationLoader:
    """Load calibration parameters from CSV with Section, Parameter_Key, Value columns."""

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        self._map = {}

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("Parameter_Key") or "").strip()
                if key and key.lower() != "nan":
                    self._map[key] = row.get("Value", "")

    def get_raw(self, key: str, default: Any = None) -> Any:
        return self._map.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        value = self._map.get(key, default)
        return self._coerce(value)

    @staticmethod
    def _coerce(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value

        text = str(value).strip()
        if text == "":
            return None
        if text.upper() == "TRUE":
            return True
        if text.upper() == "FALSE":
            return False

        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return text
