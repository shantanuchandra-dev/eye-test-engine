from __future__ import annotations

import random
from abc import ABC, abstractmethod

from fsm.simulation.virtual_patient import VirtualPatient


def _valid_options(row) -> list[str]:
    return [option for option in (row.opt_1, row.opt_2, row.opt_3, row.opt_4, row.opt_5, row.opt_6) if option]


class BehaviorModel(ABC):
    def __init__(self, seed=None, weight=1.0):
        self.random = random.Random(seed)
        self.weight = weight

    @property
    @abstractmethod
    def behavior_id(self):
        raise NotImplementedError

    @abstractmethod
    def respond(self, row, truth, case_context):
        raise NotImplementedError


class IdealResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "ideal"

    def respond(self, row, truth, case_context):
        return VirtualPatient(truth).respond(row)


class NoisyResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "noisy"

    def respond(self, row, truth, case_context):
        best = VirtualPatient(truth).respond(row)
        valid = _valid_options(row)
        if self.random.random() < 0.90 or not valid:
            return best
        others = [option for option in valid if option != best]
        return self.random.choice(others) if others else best


class HesitantResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "hesitant"

    def respond(self, row, truth, case_context):
        best = VirtualPatient(truth).respond(row)
        if row.state in ("E", "F", "H", "I", "G", "J", "K") and self.random.random() < 0.10:
            return "SAME"
        if row.state in ("B", "C", "D", "L", "P", "Q", "R", "U") and self.random.random() < 0.12:
            return "REPEAT"
        return best


class AccommodativeResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "accommodative"

    def respond(self, row, truth, case_context):
        best = VirtualPatient(truth).respond(row)
        if row.state in ("B", "D", "C", "L") and best == "BLURRY" and self.random.random() < 0.12:
            return "CLEAR"
        if row.state in ("G", "J") and self.random.random() < 0.15:
            return "RED"
        if row.state in ("P", "Q", "R") and self.random.random() < 0.12:
            return "BLURRY"
        return best


class InconsistentResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "inconsistent"

    def respond(self, row, truth, case_context):
        best = VirtualPatient(truth).respond(row)
        valid = _valid_options(row)
        if row.state in ("E", "F", "G", "H", "I", "J", "K", "U") and self.random.random() < 0.22 and valid:
            others = [option for option in valid if option != best]
            return self.random.choice(others) if others else best
        if row.state in ("B", "C", "D", "L", "P", "Q", "R") and self.random.random() < 0.15:
            return "REPEAT"
        return best


def get_behavior_models(seed=42):
    return [
        IdealResponder(seed=seed + 1, weight=0.45),
        AccommodativeResponder(seed=seed + 4, weight=0.20),
        NoisyResponder(seed=seed + 2, weight=0.15),
        HesitantResponder(seed=seed + 3, weight=0.15),
        InconsistentResponder(seed=seed + 5, weight=0.05),
    ]
