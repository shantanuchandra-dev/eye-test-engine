from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsm.config.calibration_loader import CalibrationLoader
from fsm.engines.derived_variables_engine import DerivedVariablesEngine
from fsm.models.patient import PatientInput
from fsm.models.prescription import EyePrescription
from fsm.simulation.behavior_models import IdealResponder
from fsm.simulation.case_generator import SyntheticCase
from fsm.simulation.population_runner import PopulationRunner
from fsm.simulation.result_writer import create_run_folder, save_dataframe_csv, save_json


APP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = APP_ROOT / "results"
DEFAULT_CALIBRATION_PATH = APP_ROOT / "config" / "calibration.csv"


def build_single_case(run_id: str, calibration) -> SyntheticCase:
    truth_rx = {
        "re_sph": -2.00,
        "re_cyl": -1.50,
        "re_axis": 95.0,
        "le_sph": -1.75,
        "le_cyl": -1.25,
        "le_axis": 80.0,
        "add_r": 1.25,
        "add_l": 1.25,
    }

    ar_re = EyePrescription(-2.25, -1.25, 90.0)
    ar_le = EyePrescription(-2.00, -1.00, 75.0)
    lenso_re = EyePrescription(-2.50, -1.50, 100.0)
    lenso_le = EyePrescription(-2.00, -1.25, 85.0)

    patient = PatientInput(
        visit_id=run_id,
        patient_name="Simulation Patient",
        age=46,
        occupation="",
        screen_time_hours=7.0,
        driving_hours=1.0,
        primary_reason="Blurred distance",
        symptoms_text="Blurred distance, Blurred near",
        satisfaction_with_current_rx="Not satisfied",
        wear_type="Progressive",
        distance_target_preference="",
        priority="Comfort-first",
        near_priority_declared="High",
        last_eye_test_months_ago=18.0,
        rx_change_was_large=True,
        fluctuating_vision_reported=False,
        diabetes=False,
        prior_eye_surgery="None",
        keratoconus=False,
        amblyopia=False,
        infection=False,
        optom_review_flag=False,
        autorefractor_re=ar_re,
        autorefractor_le=ar_le,
        lenso_re=lenso_re,
        lenso_le=lenso_le,
        lenso_add_r=1.0,
        lenso_add_l=1.0,
    )

    dv = DerivedVariablesEngine(calibration).derive(patient)
    return SyntheticCase(
        case_id=run_id,
        profile_id="single_patient_debug",
        patient_input=patient,
        truth_rx=truth_rx,
        ar_re=ar_re,
        ar_le=ar_le,
        lenso_re=lenso_re,
        lenso_le=lenso_le,
        lenso_add_r=1.0,
        lenso_add_l=1.0,
        dv=dv,
    )


def main() -> None:
    calibration = CalibrationLoader(str(DEFAULT_CALIBRATION_PATH))
    results_folder, run_id = create_run_folder(RESULTS_ROOT, "single_patient")
    case = build_single_case(run_id, calibration)
    runner = PopulationRunner(calibration)
    result, trace_df = runner.run_one_with_trace(case, IdealResponder(seed=7))

    save_dataframe_csv(trace_df, results_folder, "trace.csv")
    save_json(result, results_folder, "summary.json")

    print("\n===== SINGLE PATIENT SIMULATION =====")
    print(f"Run ID: {run_id}")
    print(f"Outcome: {result['outcome']}")
    print(f"Termination state: {result['termination_state']}")
    print(f"Selected prescribed Rx source: {result.get('selected_prescribed_rx_source') or 'Achieved/Fallback'}")
    print(
        f"Final prescribed RE: {result['final_re_sph']:.2f} / "
        f"{result['final_re_cyl']:.2f} x {result['final_re_axis']}"
    )
    print(
        f"Final prescribed LE: {result['final_le_sph']:.2f} / "
        f"{result['final_le_cyl']:.2f} x {result['final_le_axis']}"
    )
    print(f"Distance success within tolerance: {result['distance_success_within_tolerance']}")
    print(f"Achieved distance success within tolerance: {result['achieved_distance_success_within_tolerance']}")
    print("\nSaved files:")
    print(results_folder / "trace.csv")
    print(results_folder / "summary.json")


if __name__ == "__main__":
    main()
