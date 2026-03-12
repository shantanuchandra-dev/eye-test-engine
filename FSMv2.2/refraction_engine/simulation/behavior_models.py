import random
from abc import ABC, abstractmethod

from refraction_engine.simulation.virtual_patient import VirtualPatient


class BehaviorModel(ABC):
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    @property
    @abstractmethod
    def behavior_id(self):
        pass

    @abstractmethod
    def respond(self, row, truth, case_context):
        pass


class IdealResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "ideal"

    def respond(self, row, truth, case_context):
        vp = VirtualPatient(truth)
        return vp.respond(row)


class NoisyResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "noisy"

    def respond(self, row, truth, case_context):
        vp = VirtualPatient(truth)
        best = vp.respond(row)
        valid = [o for o in [row.opt_1, row.opt_2, row.opt_3, row.opt_4, row.opt_5, row.opt_6] if o]

        if self.random.random() < 0.88:
            return best

        others = [x for x in valid if x != best]
        return self.random.choice(others) if others else best


class HesitantResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "hesitant"

    def respond(self, row, truth, case_context):
        vp = VirtualPatient(truth)
        best = vp.respond(row)

        if row.state in ("E", "F", "H", "I") and self.random.random() < 0.25:
            return "SAME"
        if row.state in ("E", "F", "H", "I") and self.random.random() < 0.10:
            return "CANT_TELL"
        if row.state in ("G", "J") and self.random.random() < 0.30:
            return "EQUAL"

        return best


class AccommodativeResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "accommodative"

    def respond(self, row, truth, case_context):
        vp = VirtualPatient(truth)
        best = vp.respond(row)

        if row.state in ("B", "D"):
            if self.random.random() < 0.25:
                return "READABLE"

        if row.state in ("G", "J"):
            if self.random.random() < 0.30:
                return "RED_CLEARER"

        return best


class InconsistentResponder(BehaviorModel):
    @property
    def behavior_id(self):
        return "inconsistent"

    def respond(self, row, truth, case_context):
        vp = VirtualPatient(truth)
        best = vp.respond(row)
        valid = [o for o in [row.opt_1, row.opt_2, row.opt_3, row.opt_4, row.opt_5, row.opt_6] if o]

        if row.state in ("F", "I", "G", "J") and self.random.random() < 0.35 and len(valid) > 1:
            others = [x for x in valid if x != best]
            return self.random.choice(others) if others else best

        if row.state in ("E", "H") and self.random.random() < 0.20:
            return "CANT_TELL"

        return best


def get_behavior_models(seed=42):
    return [
        IdealResponder(seed=seed + 1),
        NoisyResponder(seed=seed + 2),
        HesitantResponder(seed=seed + 3),
        AccommodativeResponder(seed=seed + 4),
        InconsistentResponder(seed=seed + 5),
    ]