from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EyePrescription:
    sphere: Optional[float]
    cylinder: Optional[float]
    axis: Optional[float]

    def has_full_rx(self) -> bool:
        return (
            self.sphere is not None
            and self.cylinder is not None
            and self.axis is not None
        )


@dataclass
class AddPrescription:
    add_r: Optional[float] = None
    add_l: Optional[float] = None