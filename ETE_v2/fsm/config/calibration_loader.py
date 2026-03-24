import csv as _csv
from pathlib import Path
from typing import Any, Union
import pandas as pd


class CalibrationLoader:
    """
    Expects a CSV with at least:
    Section, Parameter_Key, Value
    """

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        # Clean header / rows
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.df = self.df.dropna(how="all")

        if "Parameter_Key" not in self.df.columns or "Value" not in self.df.columns:
            raise ValueError(
                "calibration.csv must contain 'Parameter_Key' and 'Value' columns"
            )

        self.df["Parameter_Key"] = self.df["Parameter_Key"].astype(str).str.strip()

        self._map = {}
        self._rows = []  # Full calibration snapshot: Section, Key, Value, Unit
        for _, row in self.df.iterrows():
            key = str(row["Parameter_Key"]).strip()
            if key and key.lower() != "nan":
                self._map[key] = row["Value"]
                raw_val = row["Value"]
                val_str = None
                if raw_val is not None and not pd.isna(raw_val):
                    val_str = str(raw_val).strip()
                self._rows.append({
                    "section": str(row.get("Section", "")).strip(),
                    "parameter_key": key,
                    "value": val_str,
                    "unit_or_type": str(row.get("Unit_or_Type", "")).strip(),
                })

    def get_snapshot(self) -> list:
        """Return full calibration as list of {section, parameter_key, value, unit_or_type}."""
        return list(self._rows)

    def get_raw(self, key: str, default: Any = None) -> Any:
        return self._map.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        value = self._map.get(key, default)
        return self._coerce(value)

    @staticmethod
    def read_full(csv_path: Union[str, Path]) -> list:
        """Read calibration CSV and return all parameter rows with all columns."""
        p = Path(csv_path)
        df = pd.read_csv(p, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        rows = []
        for _, row in df.iterrows():
            key = str(row.get("Parameter_Key", "")).strip()
            if not key:
                continue
            rows.append({
                "section": str(row.get("Section", "")).strip(),
                "parameter_key": key,
                "value": str(row.get("Value", "")).strip(),
                "unit_or_type": str(row.get("Unit_or_Type", "")).strip(),
                "allowed_values": str(row.get("Allowed_Values", "")).strip(),
                "parameter_explanation": str(row.get("Parameter_Explanation", "")).strip(),
            })
        return rows

    @staticmethod
    def write_values(csv_path: Union[str, Path], updates: dict) -> int:
        """Update Value column for given Parameter_Keys. Preserves blank rows and quoting."""
        p = Path(csv_path)
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = _csv.reader(f)
            lines = list(reader)

        if not lines:
            return 0

        # Find column indices
        header = [c.strip() for c in lines[0]]
        try:
            key_idx = header.index("Parameter_Key")
            val_idx = header.index("Value")
        except ValueError:
            raise ValueError("CSV must have Parameter_Key and Value columns")

        count = 0
        for row in lines[1:]:
            if len(row) <= max(key_idx, val_idx):
                continue
            k = row[key_idx].strip()
            if k in updates:
                row[val_idx] = str(updates[k])
                count += 1

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerows(lines)

        return count

    @staticmethod
    def _coerce(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if pd.isna(value):
            return None

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