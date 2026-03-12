import pandas as pd

from refraction_engine.config.calibration_loader import CalibrationLoader
from refraction_engine.engines.refraction_fsm_engine import RefractionFSMEngine
from refraction_engine.models.derived_variables import DerivedVariables
from refraction_engine.simulation.virtual_patient import TruthRx, VirtualPatient
from refraction_engine.io.result_writer import (
    create_run_folder,
    save_trace_csv,
    save_json,
)

RESULTS_ROOT = "results"


def axis_error(a, b):
    d = abs(float(a) - float(b)) % 180
    return min(d, 180 - d)


def main():
    calibration = CalibrationLoader("data/calibration.csv")
    engine = RefractionFSMEngine(calibration)

    results_folder, run_id = create_run_folder(RESULTS_ROOT, "single_patient")

    dv = DerivedVariables(
        visit_id=run_id,
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
        dv_endpoint_bias_policy="Undercorrect",
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

    truth = TruthRx(
        re_sph=-2.00,
        re_cyl=-1.50,
        re_axis=95,
        le_sph=-1.75,
        le_cyl=-3.50,
        le_axis=75,
        add_r=0.0,
        add_l=0.0,
    )

    patient = VirtualPatient(truth=truth)

    current = engine.initialize_row(
        visit_id=run_id,
        dv=dv,
        ar_re=None,
        ar_le=None,
    )

    rows = []
    max_steps = 200

    while len(rows) < max_steps:
        response = patient.respond(current)

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

        if finalized.next_state in ("END", "ESCALATE"):
            break

        next_row = engine._build_next_row(finalized, dv)
        if next_row is None:
            break

        current = next_row

    df = pd.DataFrame(rows)

    # Add truth columns so each trace is self-contained
    df["truth_re_sph"] = truth.re_sph
    df["truth_re_cyl"] = truth.re_cyl
    df["truth_re_axis"] = truth.re_axis
    df["truth_le_sph"] = truth.le_sph
    df["truth_le_cyl"] = truth.le_cyl
    df["truth_le_axis"] = truth.le_axis
    df["truth_add_r"] = truth.add_r
    df["truth_add_l"] = truth.add_l

    final_row = df.iloc[-1]

    # Final errors
    re_sph_err = abs(float(final_row["re_sph"]) - truth.re_sph)
    re_cyl_err = abs(float(final_row["re_cyl"]) - truth.re_cyl)
    re_axis_err = axis_error(float(final_row["re_axis"]), truth.re_axis)

    le_sph_err = abs(float(final_row["le_sph"]) - truth.le_sph)
    le_cyl_err = abs(float(final_row["le_cyl"]) - truth.le_cyl)
    le_axis_err = axis_error(float(final_row["le_axis"]), truth.le_axis)

    add_r_err = abs(float(final_row.get("add_r", 0.0)) - truth.add_r)
    add_l_err = abs(float(final_row.get("add_l", 0.0)) - truth.add_l)

    re_pass = re_sph_err <= 0.25 and re_cyl_err <= 0.25 and re_axis_err <= 5
    le_pass = le_sph_err <= 0.25 and le_cyl_err <= 0.25 and le_axis_err <= 5
    add_pass = add_r_err <= 0.25 and add_l_err <= 0.25
    overall_pass = re_pass and le_pass and add_pass

    # Add summary columns to every row for convenience
    df["final_re_sph"] = float(final_row["re_sph"])
    df["final_re_cyl"] = float(final_row["re_cyl"])
    df["final_re_axis"] = float(final_row["re_axis"])
    df["final_le_sph"] = float(final_row["le_sph"])
    df["final_le_cyl"] = float(final_row["le_cyl"])
    df["final_le_axis"] = float(final_row["le_axis"])
    df["final_add_r"] = float(final_row.get("add_r", 0.0))
    df["final_add_l"] = float(final_row.get("add_l", 0.0))

    df["re_sph_err"] = re_sph_err
    df["re_cyl_err"] = re_cyl_err
    df["re_axis_err"] = re_axis_err
    df["le_sph_err"] = le_sph_err
    df["le_cyl_err"] = le_cyl_err
    df["le_axis_err"] = le_axis_err
    df["add_r_err"] = add_r_err
    df["add_l_err"] = add_l_err

    df["re_within_tolerance"] = re_pass
    df["le_within_tolerance"] = le_pass
    df["add_within_tolerance"] = add_pass
    df["overall_convergence"] = overall_pass
    df["test_id"] = run_id

    trace_path = results_folder / "trace.csv"
    df.to_csv(trace_path, index=False)

    print(df[[
        "step", "state", "response_value", "chart_param",
        "re_sph", "re_cyl", "re_axis",
        "le_sph", "le_cyl", "le_axis",
        "next_state"
    ]])

    print("\n--- VIRTUAL PATIENT TRUTH ---")
    print(f"RE: sph={truth.re_sph}, cyl={truth.re_cyl}, axis={truth.re_axis}")
    print(f"LE: sph={truth.le_sph}, cyl={truth.le_cyl}, axis={truth.le_axis}")
    print(f"ADD: R={truth.add_r}, L={truth.add_l}")

    print("\n--- FSM FINAL OUTPUT ---")
    print(
        f"RE: sph={final_row['re_sph']}, cyl={final_row['re_cyl']}, axis={final_row['re_axis']}"
    )
    print(
        f"LE: sph={final_row['le_sph']}, cyl={final_row['le_cyl']}, axis={final_row['le_axis']}"
    )
    print(
        f"ADD: R={final_row.get('add_r', 0.0)}, L={final_row.get('add_l', 0.0)}"
    )

    print("\n--- ERRORS ---")
    print(f"RE errors : sph={re_sph_err:.2f}, cyl={re_cyl_err:.2f}, axis={re_axis_err:.1f}")
    print(f"LE errors : sph={le_sph_err:.2f}, cyl={le_cyl_err:.2f}, axis={le_axis_err:.1f}")
    print(f"ADD errors: R={add_r_err:.2f}, L={add_l_err:.2f}")

    print("\n--- CONVERGENCE CHECK ---")
    print(f"RE within tolerance : {re_pass}")
    print(f"LE within tolerance : {le_pass}")
    print(f"ADD within tolerance: {add_pass}")
    print(f"Overall convergence : {overall_pass}")

    summary = {
        "test_id": run_id,
        "simulation_type": "single_patient",
        "total_steps": int(len(df)),
        "termination_state": final_row["next_state"],
        "truth_re_sph": truth.re_sph,
        "truth_re_cyl": truth.re_cyl,
        "truth_re_axis": truth.re_axis,
        "truth_le_sph": truth.le_sph,
        "truth_le_cyl": truth.le_cyl,
        "truth_le_axis": truth.le_axis,
        "final_re_sph": float(final_row["re_sph"]),
        "final_re_cyl": float(final_row["re_cyl"]),
        "final_re_axis": float(final_row["re_axis"]),
        "final_le_sph": float(final_row["le_sph"]),
        "final_le_cyl": float(final_row["le_cyl"]),
        "final_le_axis": float(final_row["le_axis"]),
        "re_sph_err": float(re_sph_err),
        "re_cyl_err": float(re_cyl_err),
        "re_axis_err": float(re_axis_err),
        "le_sph_err": float(le_sph_err),
        "le_cyl_err": float(le_cyl_err),
        "le_axis_err": float(le_axis_err),
        "overall_convergence": bool(overall_pass),
    }
    save_json(summary, results_folder, "summary.json")

    print(f"\nSaved to {trace_path}")


if __name__ == "__main__":
    main()