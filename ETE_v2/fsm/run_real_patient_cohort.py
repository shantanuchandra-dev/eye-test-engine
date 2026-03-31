from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsm.simulation.real_patient_cohort import (
    DEFAULT_REAL_COHORT_BEHAVIOR_MODE,
    DEFAULT_REAL_COHORT_REPLAYS_PER_PATIENT,
    SUPPORTED_REAL_COHORT_BEHAVIOR_MODES,
    compute_distance_accuracy_metrics,
    format_duration_seconds,
    run_real_patient_cohort,
    summarize_states,
)

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = APP_ROOT / "config" / "calibration.csv"
DEFAULT_RESULTS_ROOT = APP_ROOT / "results"


def _display_label(value: Any) -> str:
    if value is None:
        return "Unknown"
    try:
        if value != value:
            return "Unknown"
    except Exception:
        pass
    return str(value)


def _print_distribution(title: str, counts) -> None:
    print(f"\n{title}")
    if counts.empty:
        print("  None")
        return

    total = int(counts.sum())
    for label, count in counts.items():
        numeric_count = int(count)
        pct = (numeric_count / total) if total else 0.0
        print(f"  {_display_label(label)}: {numeric_count} ({pct:.1%})")


def _print_metric(label: str, value: float) -> None:
    print(f"  {label}: {value:.2%}")


def _print_count_metric(label: str, count: int, total: int) -> None:
    pct = (count / total) if total else 0.0
    print(f"  {label}: {count} / {total} ({pct:.2%})")


def _print_scalar_metric(label: str, value: float) -> None:
    print(f"  {label}: {value:.2f}")


def _print_state_table(state_table) -> None:
    print("\nPrimary non-completion states")
    if state_table.empty:
        print("  None")
        return

    for row in state_table.itertuples(index=False):
        print(
            "  "
            f"{row.state_code} {row.state_name}: "
            f"{int(row.non_completions)} ({float(row.pct_of_non_completions):.1%})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the FSM against a real-patient cohort CSV by using step-12 Rx as "
            "the truth target and synthesizing the missing patient-history fields."
        )
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Absolute or relative path to the raw patient CSV.",
    )
    parser.add_argument(
        "--calibration-path",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="Path to the FSM calibration CSV.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Base folder where run outputs should be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on valid cases to replay. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum FSM steps per patient before forcing a timeout.",
    )
    parser.add_argument(
        "--behavior-mode",
        choices=list(SUPPORTED_REAL_COHORT_BEHAVIOR_MODES),
        default=DEFAULT_REAL_COHORT_BEHAVIOR_MODE,
        help=(
            "Response model used for real-cohort replay. "
            "Use patient_weighted for patient-conditioned sampling, "
            "deterministic for pure VirtualPatient, or a fixed behavior such as ideal."
        ),
    )
    parser.add_argument(
        "--replays-per-patient",
        type=int,
        default=DEFAULT_REAL_COHORT_REPLAYS_PER_PATIENT,
        help="Number of replay simulations to run per patient case.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260318,
        help="Base seed for patient-weighted behavior sampling.",
    )
    parser.add_argument(
        "--qms-id",
        default=None,
        help="Optional qms_id to replay a single real-patient case.",
    )
    parser.add_argument(
        "--require-near-test",
        action="store_true",
        help="Filter to only cases where dv_near_test_required=True before applying limit.",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save trace.csv for a single selected patient replay. Requires one case and one replay.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    results_folder, run_id, results_df, invalid_df, trial_df, summary = run_real_patient_cohort(
        csv_path=args.csv_path,
        calibration_path=args.calibration_path,
        results_root=args.results_root,
        limit=args.limit,
        max_steps=args.max_steps,
        behavior_mode=args.behavior_mode,
        replays_per_patient=args.replays_per_patient,
        seed_base=args.seed_base,
        qms_id=args.qms_id,
        require_near_test=args.require_near_test,
        save_trace=args.save_trace,
    )

    total = len(results_df)
    total_trials = len(trial_df)
    excluded = len(invalid_df)

    completion_rate = results_df["completion_probability"].mean() if total else 0.0
    escalation_rate = results_df["escalation_probability"].mean() if total else 0.0
    timeout_rate = results_df["timeout_probability"].mean() if total else 0.0
    avg_steps = results_df["steps"].mean() if total else 0.0
    completed_trials = int((trial_df["outcome"] == "SUCCESS").sum()) if total_trials else 0
    escalated_trials = int((trial_df["outcome"] == "ESCALATE").sum()) if total_trials else 0
    timed_out_trials = int((trial_df["outcome"] == "TIMEOUT").sum()) if total_trials else 0

    completed = trial_df[trial_df["outcome"] == "SUCCESS"]
    rx_metrics = compute_distance_accuracy_metrics(completed)
    state_table = summarize_states(trial_df)
    behavior_counts = (
        trial_df["behavior_id"].value_counts(dropna=False)
        if "behavior_id" in trial_df.columns
        else None
    )

    print("\n===== REAL PATIENT COHORT REPLAY =====")
    print(f"Run ID: {run_id}")

    print("\n--- SUMMARY ---")
    print(f"\nBehavior mode: {args.behavior_mode}")
    print(f"Replays per patient: {args.replays_per_patient}")
    print(f"Patient cases: {total}")
    print(f"Replay simulations: {total_trials}")
    print(f"Excluded records: {excluded}")
    if args.qms_id:
        print(f"Selected qms_id: {args.qms_id}")
    if args.require_near_test:
        print("Near-test filter: required")
    print(
        "Total runtime: "
        f"{summary.get('duration_display', format_duration_seconds(summary.get('duration_seconds', 0.0)))}"
    )

    print("\nDV profile distribution")
    _print_distribution("Age bucket", results_df["dv_age_bucket"].value_counts(dropna=False))
    _print_distribution(
        "Near test required",
        results_df["dv_near_test_required"].value_counts(dropna=False),
    )
    _print_distribution(
        "Fogging required",
        results_df["dv_fogging_required"].value_counts(dropna=False),
    )

    if behavior_counts is not None:
        _print_distribution("Replay behavior mix", behavior_counts)

    print("\nOperational metrics")
    _print_count_metric("Completed replays", completed_trials, total_trials)
    _print_count_metric("Escalated replays", escalated_trials, total_trials)
    _print_count_metric("Timed out replays", timed_out_trials, total_trials)
    _print_metric("Completion rate", completion_rate)
    _print_metric("Escalation rate", escalation_rate)
    _print_metric("Timeout rate", timeout_rate)

    print("\nClinical metrics")
    _print_metric("Accuracy among completed replays", rx_metrics["accuracy_among_completed"])
    _print_scalar_metric("Average steps", avg_steps)

    print("\nRX parameter accuracy")
    _print_metric("RE sphere within 0.25D", rx_metrics["RE_sphere_within_0.25"])
    _print_metric("RE cyl within 0.25D", rx_metrics["RE_cyl_within_0.25"])
    _print_metric("RE axis within 10 deg", rx_metrics["RE_axis_within_10deg"])
    _print_metric("LE sphere within 0.25D", rx_metrics["LE_sphere_within_0.25"])
    _print_metric("LE cyl within 0.25D", rx_metrics["LE_cyl_within_0.25"])
    _print_metric("LE axis within 10 deg", rx_metrics["LE_axis_within_10deg"])

    _print_state_table(state_table)

    print("\nSaved files:")
    print(results_folder / "cohort_replay_results.csv")
    print(results_folder / "cohort_replay_trials.csv")
    print(results_folder / "excluded_records.csv")
    print(results_folder / "rx_accuracy_metrics.csv")
    print(results_folder / "leaderboard_gender.csv")
    print(results_folder / "leaderboard_age_bucket.csv")
    print(results_folder / "fsm_noncompletion_distribution.csv")
    print(results_folder / "behavior_distribution.csv")
    if summary.get("trace_file"):
        print(summary["trace_file"])
    print(results_folder / "summary.json")


if __name__ == "__main__":
    main()
