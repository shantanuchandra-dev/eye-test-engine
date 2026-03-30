from __future__ import annotations

import random

import pandas as pd

from fsm.engines.refraction_fsm_engine import RefractionFSMEngine
from fsm.simulation.behavior_models import get_behavior_models
from fsm.simulation.case_generator import generate_case
from fsm.simulation.common import case_truth, execute_case
from fsm.simulation.profile_library import get_profiles


class PopulationRunner:
    def __init__(self, calibration):
        self.calibration = calibration
        self.engine = RefractionFSMEngine(calibration)

    def _base_result(self, case, behavior_model) -> dict:
        truth = case_truth(case)
        return {
            "case_id": case.case_id,
            "profile_id": case.profile_id,
            "behavior_id": behavior_model.behavior_id,
            "truth_re_sph": truth.re_sph,
            "truth_re_cyl": truth.re_cyl,
            "truth_re_axis": truth.re_axis,
            "truth_le_sph": truth.le_sph,
            "truth_le_cyl": truth.le_cyl,
            "truth_le_axis": truth.le_axis,
            "truth_add_r": truth.add_r,
            "truth_add_l": truth.add_l,
            "ar_re_sph": case.ar_re.sphere,
            "ar_re_cyl": case.ar_re.cylinder,
            "ar_re_axis": case.ar_re.axis,
            "ar_le_sph": case.ar_le.sphere,
            "ar_le_cyl": case.ar_le.cylinder,
            "ar_le_axis": case.ar_le.axis,
            "lenso_re_sph": case.lenso_re.sphere,
            "lenso_re_cyl": case.lenso_re.cylinder,
            "lenso_re_axis": case.lenso_re.axis,
            "lenso_le_sph": case.lenso_le.sphere,
            "lenso_le_cyl": case.lenso_le.cylinder,
            "lenso_le_axis": case.lenso_le.axis,
            "truth_add_valid": True,
        }

    def run_one(self, case, behavior_model, max_steps=200):
        result, _ = execute_case(
            engine=self.engine,
            case=case,
            behavior_model=behavior_model,
            max_steps=max_steps,
            collect_trace=False,
        )
        result.update(self._base_result(case, behavior_model))
        result.update(case.dv.__dict__)
        return result

    def run_one_with_trace(self, case, behavior_model, max_steps=200):
        result, trace_df = execute_case(
            engine=self.engine,
            case=case,
            behavior_model=behavior_model,
            max_steps=max_steps,
            collect_trace=True,
            trace_metadata={
                "test_id": case.case_id,
                "case_id": case.case_id,
                "profile_id": case.profile_id,
                "behavior_id": behavior_model.behavior_id,
            },
        )
        result.update(self._base_result(case, behavior_model))
        result.update(case.dv.__dict__)
        return result, (trace_df if trace_df is not None else pd.DataFrame())

    def run_population(self, n_truth, seed_base=1000, max_steps=200):
        random.seed(seed_base)

        profiles = get_profiles()
        behaviors = get_behavior_models(seed=seed_base)

        profile_weights = [p.get("population_weight", 1.0) for p in profiles]
        behavior_weights = [getattr(b, "weight", 1.0) for b in behaviors]
        rows = []

        for i in range(n_truth):
            profile = random.choices(profiles, weights=profile_weights, k=1)[0]
            behavior = random.choices(behaviors, weights=behavior_weights, k=1)[0]
            case_id = f"T{i:05d}_{profile['profile_id']}"
            case = generate_case(
                case_id=case_id,
                profile=profile,
                calibration=self.calibration,
                rng_seed=seed_base + i,
            )
            rows.append(self.run_one(case, behavior, max_steps=max_steps))

            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} simulations")

        return pd.DataFrame(rows)
