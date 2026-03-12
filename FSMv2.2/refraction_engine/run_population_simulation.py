import pandas as pd

from refraction_engine.config.calibration_loader import CalibrationLoader
from refraction_engine.simulation.population_runner import PopulationRunner
from refraction_engine.io.result_writer import (
    create_run_folder,
    save_dataframe_csv,
    save_json,
)

RESULTS_ROOT = "results"


STATE_NAMES = {
    "A": "Distance Baseline",
    "B": "Coarse Sphere RE",
    "D": "Coarse Sphere LE",
    "E": "JCC Axis RE",
    "F": "JCC Power RE",
    "G": "Duochrome RE",
    "H": "JCC Axis LE",
    "I": "JCC Power LE",
    "J": "Duochrome LE",
    "K": "Binocular Balance",
    "P": "Near Add RE",
    "Q": "Near Add LE",
    "R": "Near Binocular Validation",
}


def summarize_group(df, group_col):
    g = df.groupby(group_col)
    rows = []

    for name, d in g:
        total = len(d)

        completion = (d["outcome"] == "SUCCESS").mean()
        escalation = (d["outcome"] == "ESCALATE").mean()
        timeout = (d["outcome"] == "TIMEOUT").mean()

        completed = d[d["outcome"] == "SUCCESS"]

        if len(completed) > 0:
            rx_metrics = compute_rx_accuracy_metrics(completed)
            accuracy = rx_metrics["accuracy_among_completed"]
        else:
            accuracy = 0.0

        rows.append(
            {
                group_col: name,
                "total_cases": total,
                "completion_rate": round(completion, 4),
                "escalation_rate": round(escalation, 4),
                "timeout_rate": round(timeout, 4),
                "accuracy_among_completed": round(accuracy, 4),
                "avg_steps": round(d["steps"].mean(), 2),
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy_among_completed")


def summarize_states(df):
    failures = df[df["outcome"] != "SUCCESS"]
    g = failures.groupby("failure_state")
    rows = []

    total_failures = len(failures)

    for state, d in g:
        escalations = (d["outcome"] == "ESCALATE").sum()
        timeouts = (d["outcome"] == "TIMEOUT").sum()

        rows.append(
            {
                "state_code": state,
                "state_name": STATE_NAMES.get(state, state),
                "escalations": escalations,
                "timeouts": timeouts,
                "non_completions": len(d),
                "pct_of_non_completions": round(len(d) / total_failures, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("non_completions", ascending=False)


def compute_rx_accuracy_metrics(completed_df):
    re_sphere_025 = (completed_df["re_sph_err"] <= 0.25).mean()
    re_cyl_025 = (completed_df["re_cyl_err"] <= 0.25).mean()
    re_axis_5 = (completed_df["re_axis_err"] <= 5).mean()

    le_sphere_025 = (completed_df["le_sph_err"] <= 0.25).mean()
    le_cyl_025 = (completed_df["le_cyl_err"] <= 0.25).mean()
    le_axis_5 = (completed_df["le_axis_err"] <= 5).mean()

    cumulative_avg_accuracy = (
        re_sphere_025
        + re_cyl_025
        + re_axis_5
        + le_sphere_025
        + le_cyl_025
        + le_axis_5
    ) / 6.0

    return {
        "accuracy_among_completed": cumulative_avg_accuracy,
        "RE_sphere_within_0.25": re_sphere_025,
        "RE_cyl_within_0.25": re_cyl_025,
        "RE_axis_within_5deg": re_axis_5,
        "LE_sphere_within_0.25": le_sphere_025,
        "LE_cyl_within_0.25": le_cyl_025,
        "LE_axis_within_5deg": le_axis_5,
    }


def main():
    calibration = CalibrationLoader("data/calibration.csv")
    runner = PopulationRunner(calibration)

    results_folder, run_id = create_run_folder(RESULTS_ROOT, "population")

    n_truth = 100
    df = runner.run_population(n_truth=n_truth)

    total = len(df)

    completion_rate = (df["outcome"] == "SUCCESS").mean()
    escalation_rate = (df["outcome"] == "ESCALATE").mean()
    timeout_rate = (df["outcome"] == "TIMEOUT").mean()

    completed = df[df["outcome"] == "SUCCESS"]

    if len(completed) > 0:
        rx_metrics = compute_rx_accuracy_metrics(completed)
        accuracy_among_completed = rx_metrics["accuracy_among_completed"]
    else:
        rx_metrics = {
            "accuracy_among_completed": 0.0,
            "RE_sphere_within_0.25": 0.0,
            "RE_cyl_within_0.25": 0.0,
            "RE_axis_within_5deg": 0.0,
            "LE_sphere_within_0.25": 0.0,
            "LE_cyl_within_0.25": 0.0,
            "LE_axis_within_5deg": 0.0,
        }
        accuracy_among_completed = 0.0

    avg_steps = df["steps"].mean()

    print("\n--- SUMMARY ---")
    print("Total simulations:", total)

    print("\nOperational metrics")
    print("Completion rate:", round(completion_rate, 4))
    print("Escalation rate:", round(escalation_rate, 4))
    print("Timeout rate:", round(timeout_rate, 4))

    print("\nClinical metrics")
    print("Accuracy among completed tests:", round(accuracy_among_completed, 4))
    print("Average steps:", round(avg_steps, 4))

    print("\n--- RX PARAMETER ACCURACY ---")
    print("RE_sphere_within_0.25:", round(rx_metrics["RE_sphere_within_0.25"], 4))
    print("RE_cyl_within_0.25:", round(rx_metrics["RE_cyl_within_0.25"], 4))
    print("RE_axis_within_5deg:", round(rx_metrics["RE_axis_within_5deg"], 4))
    print("LE_sphere_within_0.25:", round(rx_metrics["LE_sphere_within_0.25"], 4))
    print("LE_cyl_within_0.25:", round(rx_metrics["LE_cyl_within_0.25"], 4))
    print("LE_axis_within_5deg:", round(rx_metrics["LE_axis_within_5deg"], 4))

    profile_table = summarize_group(df, "profile_id")
    behavior_table = summarize_group(df, "behavior_id")
    state_table = summarize_states(df)

    save_dataframe_csv(df, results_folder, "population_simulation_results.csv")
    save_dataframe_csv(profile_table, results_folder, "leaderboard_profiles.csv")
    save_dataframe_csv(behavior_table, results_folder, "leaderboard_behaviors.csv")
    save_dataframe_csv(state_table, results_folder, "fsm_noncompletion_distribution.csv")

    rx_metrics_df = pd.DataFrame([rx_metrics])
    save_dataframe_csv(rx_metrics_df, results_folder, "rx_accuracy_metrics.csv")

    summary = {
        "run_id": run_id,
        "simulation_type": "population",
        "n_truth": n_truth,
        "total_simulations": int(total),
        "completion_rate": float(round(completion_rate, 4)),
        "escalation_rate": float(round(escalation_rate, 4)),
        "timeout_rate": float(round(timeout_rate, 4)),
        "accuracy_among_completed": float(round(accuracy_among_completed, 4)),
        "average_steps": float(round(avg_steps, 4)),
    }
    save_json(summary, results_folder, "summary.json")

    print("\n--- PROFILE LEADERBOARD ---")
    print(profile_table)

    print("\n--- BEHAVIOR LEADERBOARD ---")
    print(behavior_table)

    print("\n--- FSM NON-COMPLETION DISTRIBUTION ---")
    print(state_table)

    print("\nSaved files:")
    print(results_folder / "population_simulation_results.csv")
    print(results_folder / "leaderboard_profiles.csv")
    print(results_folder / "leaderboard_behaviors.csv")
    print(results_folder / "fsm_noncompletion_distribution.csv")
    print(results_folder / "rx_accuracy_metrics.csv")
    print(results_folder / "summary.json")


if __name__ == "__main__":
    main()