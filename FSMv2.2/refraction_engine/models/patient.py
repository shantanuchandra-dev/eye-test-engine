from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .prescription import EyePrescription


@dataclass
class PatientInput:
    visit_id: str

    age: Optional[int] = None
    occupation: str = ""
    screen_time_hours: Optional[float] = None
    driving_hours: Optional[float] = None

    primary_reason: str = ""
    symptoms_text: str = ""
    satisfaction_with_current_rx: str = ""
    wear_type: str = ""
    distance_target_preference: str = ""
    priority: str = ""
    near_priority_declared: str = ""

    last_eye_test_months_ago: Optional[float] = None
    rx_change_was_large: bool = False
    fluctuating_vision_reported: bool = False

    diabetes: bool = False
    prior_eye_surgery: str = ""
    keratoconus: bool = False
    amblyopia: bool = False
    infection: bool = False
    optom_review_flag: bool = False

    autorefractor_re: Optional[EyePrescription] = None
    autorefractor_le: Optional[EyePrescription] = None

    lenso_re: Optional[EyePrescription] = None
    lenso_le: Optional[EyePrescription] = None

    lenso_add_r: Optional[float] = None
    lenso_add_l: Optional[float] = None

    truth_re: Optional[EyePrescription] = None
    truth_le: Optional[EyePrescription] = None