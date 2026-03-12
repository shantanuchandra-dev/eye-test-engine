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
        for _, row in self.df.iterrows():
            key = str(row["Parameter_Key"]).strip()
            if key and key.lower() != "nan":
                self._map[key] = row["Value"]

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