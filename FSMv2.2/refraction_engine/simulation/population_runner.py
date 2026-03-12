import pandas as pd

from refraction_engine.engines.refraction_fsm_engine import RefractionFSMEngine
from refraction_engine.simulation.behavior_models import get_behavior_models
from refraction_engine.simulation.case_generator import generate_case
from refraction_engine.simulation.profile_library import get_profiles
from refraction_engine.simulation.virtual_patient import TruthRx


def axis_error(a, b):
    d = abs(float(a) - float(b)) % 180
    return min(d, 180 - d)


class PopulationRunner:
    def __init__(self, calibration):
        self.calibration = calibration
        self.engine = RefractionFSMEngine(calibration)

    def run_one(self, case, behavior_model, max_steps=200):
        truth = TruthRx(
            re_sph=case.truth_rx["re_sph"],
            re_cyl=case.truth_rx["re_cyl"],
            re_axis=case.truth_rx["re_axis"],
            le_sph=case.truth_rx["le_sph"],
            le_cyl=case.truth_rx["le_cyl"],
            le_axis=case.truth_rx["le_axis"],
            add_r=case.truth_rx["add_r"],
            add_l=case.truth_rx["add_l"],
        )

        current = self.engine.initialize_row(
            visit_id=case.case_id,
            dv=case.dv,
            ar_re=case.ar_re,
            ar_le=case.ar_le,
        )

        steps = 0
        last_row = None

        while steps < max_steps:
            response = behavior_model.respond(
                row=current,
                truth=truth,
                case_context=case,
            )

            finalized = self.engine.apply_response(
                current=current,
                response_value=response,
                dv=case.dv,
                ar_re=case.ar_re,
                ar_le=case.ar_le,
            )

            last_row = finalized
            steps += 1

            if finalized.next_state in ("END", "ESCALATE"):
                break

            next_row = self.engine._build_next_row(finalized, case.dv)
            if next_row is None:
                break

            current = next_row

        if last_row is None:
            raise RuntimeError("Simulation produced no rows")

        outcome = "SUCCESS" if last_row.next_state == "END" else "ESCALATE"
        if steps >= max_steps and outcome != "SUCCESS":
            outcome = "TIMEOUT"

        re_sph_err = abs((last_row.re_sph or 0.0) - truth.re_sph)
        re_cyl_err = abs((last_row.re_cyl or 0.0) - truth.re_cyl)
        re_axis_err = axis_error((last_row.re_axis or 0.0), truth.re_axis)

        le_sph_err = abs((last_row.le_sph or 0.0) - truth.le_sph)
        le_cyl_err = abs((last_row.le_cyl or 0.0) - truth.le_cyl)
        le_axis_err = axis_error((last_row.le_axis or 0.0), truth.le_axis)

        success_within_tolerance = (
            re_sph_err <= 0.25
            and re_cyl_err <= 0.25
            and re_axis_err <= 5
            and le_sph_err <= 0.25
            and le_cyl_err <= 0.25
            and le_axis_err <= 5
        )

        return {
            "case_id": case.case_id,
            "profile_id": case.profile_id,
            "behavior_id": behavior_model.behavior_id,
            "steps": steps,
            "outcome": outcome,
            "termination_state": last_row.next_state,
            "failure_state": last_row.state,
            "success_within_tolerance": success_within_tolerance,

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

            "dv_start_source_policy": case.dv.dv_start_source_policy,
            "dv_endpoint_bias_policy": case.dv.dv_endpoint_bias_policy,
            "dv_step_size_policy": case.dv.dv_step_size_policy,
            "dv_fogging_policy": case.dv.dv_fogging_policy,
            "dv_expected_convergence_time": case.dv.dv_expected_convergence_time,
            "dv_confidence_requirement": case.dv.dv_confidence_requirement,
            "dv_near_test_required": case.dv.dv_near_test_required,
            "dv_anomaly_watch": case.dv.dv_anomaly_watch,

            "final_re_sph": last_row.re_sph,
            "final_re_cyl": last_row.re_cyl,
            "final_re_axis": last_row.re_axis,
            "final_le_sph": last_row.le_sph,
            "final_le_cyl": last_row.le_cyl,
            "final_le_axis": last_row.le_axis,
            "final_add_r": last_row.add_r,
            "final_add_l": last_row.add_l,

            "re_sph_err": re_sph_err,
            "re_cyl_err": re_cyl_err,
            "re_axis_err": re_axis_err,
            "le_sph_err": le_sph_err,
            "le_cyl_err": le_cyl_err,
            "le_axis_err": le_axis_err,
        }

    def run_population(self, n_truth, seed_base=1000):
        profiles = get_profiles()
        behaviors = get_behavior_models(seed=seed_base)

        rows = []
        counter = 0

        for truth_idx in range(n_truth):
            for profile in profiles:
                case_id = f"T{truth_idx:05d}_{profile['profile_id']}"
                case = generate_case(
                    case_id=case_id,
                    profile=profile,
                    calibration=self.calibration,
                    rng_seed=seed_base + truth_idx,
                )

                for behavior in behaviors:
                    result = self.run_one(case, behavior)
                    rows.append(result)
                    counter += 1

                    if counter % 1000 == 0:
                        print(f"Completed {counter} simulations")

        return pd.DataFrame(rows)