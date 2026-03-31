from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsm.config.calibration_loader import CalibrationLoader
from fsm.simulation.common import STATE_NAMES
from fsm.simulation.population_runner import PopulationRunner
from fsm.simulation.real_patient_cohort import compute_add_accuracy_metrics, compute_distance_accuracy_metrics
from fsm.simulation.result_writer import create_run_folder, save_dataframe_csv, save_json


APP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = APP_ROOT / "results"
DEFAULT_CALIBRATION_PATH = APP_ROOT / "config" / "calibration.csv"


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if len(df) == 0 or group_col not in df.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "total_cases",
                "completion_rate",
                "escalation_rate",
                "timeout_rate",
                "distance_accuracy_among_completed",
                "distance_success_rate_all_cases",
                "avg_steps",
            ]
        )
    rows = []
    for name, group_df in df.groupby(group_col):
        completed = group_df[group_df["outcome"] == "SUCCESS"]
        distance_metrics = compute_distance_accuracy_metrics(completed)
        rows.append(
            {
                group_col: name,
                "total_cases": int(len(group_df)),
                "completion_rate": round(float((group_df["outcome"] == "SUCCESS").mean()), 4),
                "escalation_rate": round(float((group_df["outcome"] == "ESCALATE").mean()), 4),
                "timeout_rate": round(float((group_df["outcome"] == "TIMEOUT").mean()), 4),
                "distance_accuracy_among_completed": round(float(distance_metrics["accuracy_among_completed"]), 4),
                "distance_success_rate_all_cases": round(float(group_df["distance_success_within_tolerance"].mean()), 4),
                "avg_steps": round(float(group_df["steps"].mean()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("distance_accuracy_among_completed")


def summarize_states(df: pd.DataFrame) -> pd.DataFrame:
    failures = df[df["outcome"] != "SUCCESS"]
    if len(failures) == 0:
        return pd.DataFrame(
            columns=[
                "state_code",
                "state_name",
                "escalations",
                "timeouts",
                "non_completions",
                "pct_of_non_completions",
            ]
        )
    rows = []
    total_failures = len(failures)
    for state, group_df in failures.groupby("failure_state"):
        rows.append(
            {
                "state_code": state,
                "state_name": STATE_NAMES.get(state, state),
                "escalations": int((group_df["outcome"] == "ESCALATE").sum()),
                "timeouts": int((group_df["outcome"] == "TIMEOUT").sum()),
                "non_completions": int(len(group_df)),
                "pct_of_non_completions": round(float(len(group_df) / total_failures), 4) if total_failures else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("non_completions", ascending=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic population simulator against the current ETE_v2 FSM.")
    parser.add_argument("--n-truth", type=int, default=10000, help="Number of synthetic cases to simulate.")
    parser.add_argument("--seed-base", type=int, default=1000, help="Seed base for profile/case generation.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum FSM steps per simulated case.")
    parser.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH), help="Path to the live calibration CSV.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT), help="Folder where simulation outputs should be written.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    calibration = CalibrationLoader(args.calibration_path)
    runner = PopulationRunner(calibration)
    results_folder, run_id = create_run_folder(args.results_root, "population")

    df = runner.run_population(
        n_truth=args.n_truth,
        seed_base=args.seed_base,
        max_steps=args.max_steps,
    )
    completed = df[df["outcome"] == "SUCCESS"]
    distance_metrics = compute_distance_accuracy_metrics(completed)
    add_metrics = compute_add_accuracy_metrics(completed)

    profile_table = summarize_group(df, "profile_id")
    behavior_table = summarize_group(df, "behavior_id")
    state_table = summarize_states(df)

    save_dataframe_csv(df, results_folder, "population_simulation_results.csv")
    save_dataframe_csv(profile_table, results_folder, "leaderboard_profiles.csv")
    save_dataframe_csv(behavior_table, results_folder, "leaderboard_behaviors.csv")
    save_dataframe_csv(state_table, results_folder, "fsm_noncompletion_distribution.csv")
    save_dataframe_csv(pd.DataFrame([distance_metrics | add_metrics]), results_folder, "rx_accuracy_metrics.csv")

    summary = {
        "run_id": run_id,
        "simulation_type": "population",
        "n_truth": int(args.n_truth),
        "seed_base": int(args.seed_base),
        "max_steps": int(args.max_steps),
        "total_simulations": int(len(df)),
        "completion_rate": float(round((df["outcome"] == "SUCCESS").mean(), 4)),
        "escalation_rate": float(round((df["outcome"] == "ESCALATE").mean(), 4)),
        "timeout_rate": float(round((df["outcome"] == "TIMEOUT").mean(), 4)),
        "distance_accuracy_among_completed": float(round(distance_metrics["accuracy_among_completed"], 4)),
        "distance_success_rate_all_cases": float(round(df["distance_success_within_tolerance"].mean(), 4)),
        "add_accuracy_among_completed_valid_add": float(round(add_metrics["add_accuracy_among_completed_valid_add"], 4)),
        "average_steps": float(round(df["steps"].mean(), 4)),
    }
    save_json(summary, results_folder, "summary.json")

    print("\n===== POPULATION SIMULATION =====")
    print(f"Run ID: {run_id}")
    print(f"Total simulations: {len(df)}")
    print(f"Completion rate: {summary['completion_rate']:.2%}")
    print(f"Escalation rate: {summary['escalation_rate']:.2%}")
    print(f"Timeout rate: {summary['timeout_rate']:.2%}")
    print(f"Distance accuracy among completed: {summary['distance_accuracy_among_completed']:.2%}")
    print(f"Average steps: {summary['average_steps']:.2f}")
    print("\nSaved files:")
    print(results_folder / "population_simulation_results.csv")
    print(results_folder / "summary.json")


if __name__ == "__main__":
    main()
