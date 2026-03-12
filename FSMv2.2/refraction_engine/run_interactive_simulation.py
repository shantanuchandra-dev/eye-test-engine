import pandas as pd

from refraction_engine.config.calibration_loader import CalibrationLoader
from refraction_engine.engines.refraction_fsm_engine import RefractionFSMEngine
from refraction_engine.models.derived_variables import DerivedVariables
from refraction_engine.io.result_writer import (
    create_run_folder,
    save_trace_csv,
    save_json,
)

RESULTS_ROOT = "results"


def main():

    calibration = CalibrationLoader("data/calibration.csv")
    engine = RefractionFSMEngine(calibration)

    dv = DerivedVariables(
        visit_id="SIM_INTERACTIVE",
        dv_age_bucket="Adult",
        dv_distance_priority="High",
        dv_near_priority="Medium",
        dv_symptom_risk_level="None",
        dv_medical_risk_level="None",
        dv_stability_level="Stable",
        dv_ar_lenso_mismatch_level_RE="Small",
        dv_ar_lenso_mismatch_level_LE="Small",
        dv_start_source_policy="Start_AR",
        dv_start_rx_RE_sph=-2.50,
        dv_start_rx_RE_cyl=-1.75,
        dv_start_rx_RE_axis=90,
        dv_start_rx_LE_sph=-2.50,
        dv_start_rx_LE_cyl=-3.75,
        dv_start_rx_LE_axis=70,
        dv_add_expected="None",
        dv_target_distance_va="6/6_target",
        dv_endpoint_bias_policy="Neutral",
        dv_step_size_policy="Standard",
        dv_max_delta_from_start_sph=2.0,
        dv_max_delta_from_ar_sph=3.0,
        dv_axis_tolerance_deg=5,
        dv_cyl_tolerance_D=0.25,
        dv_requires_optom_review=False,
        dv_anomaly_watch=True,
        dv_expected_convergence_time="Normal",
        dv_branching_guardrails="Relaxed",
        dv_confidence_requirement="Medium",
        dv_fogging_policy="Strong_Fog",
        dv_fogging_amount_D=1.0,
        dv_fogging_clearance_mode="StepDown_0.25",
        dv_fogging_required_confirmation=2,
        dv_axis_step_policy="Normal",
        dv_duochrome_max_flips=5,
        dv_near_test_required=False,
    )

    results_folder, run_id = create_run_folder(RESULTS_ROOT, "interactive")
    dv.visit_id = run_id

    current = engine.initialize_row(
        visit_id=run_id,
        dv=dv,
        ar_re=None,
        ar_le=None,
    )

    rows = []

    print("\n===== INTERACTIVE FSM REFRACTION SIMULATOR =====\n")
    print(f"Run ID: {run_id}")

    while True:

        print("\n--------------------------------")
        print(f"STEP: {current.step}")
        print(f"STATE: {current.state} | {current.phase_name}")
        print(f"CHART: {current.chart_param}")
        print(f"EYE: {current.eye}")

        print("\nCURRENT POWER")
        print(
            f"RE: {current.re_sph:.2f} / {current.re_cyl:.2f} x {current.re_axis}"
        )
        print(
            f"LE: {current.le_sph:.2f} / {current.le_cyl:.2f} x {current.le_axis}"
        )

        print("\nQUESTION:")
        print(current.question)

        options = [
            current.opt_1,
            current.opt_2,
            current.opt_3,
            current.opt_4,
            current.opt_5,
            current.opt_6,
        ]

        options = [o for o in options if o not in ("", None)]

        print("\nOPTIONS:")
        for i, o in enumerate(options):
            print(f"{i+1}. {o}")

        user = input("\nSelect option number (or type value): ").strip()

        if user.isdigit():
            idx = int(user) - 1
            if idx < len(options):
                response = options[idx]
            else:
                print("Invalid option")
                continue
        else:
            response = user

        finalized = engine.apply_response(
            current=current,
            response_value=response,
            dv=dv,
            ar_re=None,
            ar_le=None,
        )

        row_dict = finalized.__dict__.copy()
        row_dict["test_id"] = run_id
        rows.append(row_dict)

        print(f"\nResponse recorded: {response}")
        print(f"Next state: {finalized.next_state}")

        if finalized.next_state in ("END", "ESCALATE"):
            print("\nTEST TERMINATED:", finalized.next_state)
            break

        next_row = engine._build_next_row(finalized, dv)

        if next_row is None:
            print("\nFSM finished")
            break

        current = next_row

    trace_path = save_trace_csv(rows, results_folder, "trace.csv")

    summary = {
        "test_id": run_id,
        "simulation_type": "interactive",
        "total_steps": len(rows),
        "final_state": rows[-1]["state"] if rows else None,
        "termination_state": rows[-1]["next_state"] if rows else None,
        "final_re_sph": rows[-1]["re_sph"] if rows else None,
        "final_re_cyl": rows[-1]["re_cyl"] if rows else None,
        "final_re_axis": rows[-1]["re_axis"] if rows else None,
        "final_le_sph": rows[-1]["le_sph"] if rows else None,
        "final_le_cyl": rows[-1]["le_cyl"] if rows else None,
        "final_le_axis": rows[-1]["le_axis"] if rows else None,
        "trace_file": str(trace_path),
    }
    save_json(summary, results_folder, "summary.json")

    print(f"\nSimulation saved to {trace_path}")


if __name__ == "__main__":
    main()