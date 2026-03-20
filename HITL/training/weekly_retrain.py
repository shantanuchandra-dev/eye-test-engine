#!/usr/bin/env python3
"""Weekly retraining orchestrator.

Designed to run as a cron job every Monday at 3am:
    0 3 * * 1 cd /path/to/Eye_test_engine_v2 && venv/bin/python -m voice.training.weekly_retrain

Steps:
    1. Check if enough new annotations since last training
    2. Run Whisper fine-tuning
    3. Run A/B test: old model vs new model
    4. Auto-promote if new model is better
    5. Run confidence threshold optimization
    6. Run fuzzy matcher expansion analysis
    7. Send summary report
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FINETUNED_DIR = MODELS_DIR / "whisper-finetuned"
ANALYSIS_DIR = AUDIO_BASE_DIR / "_analysis"
RETRAIN_LOG = ANALYSIS_DIR / "retrain_log.jsonl"

MIN_NEW_ANNOTATIONS = 50


def count_new_annotations_since(last_train_date: str = None) -> int:
    """Count reviewed annotations since the last training date."""
    count = 0
    if not AUDIO_BASE_DIR.exists():
        return count

    for date_dir in sorted(AUDIO_BASE_DIR.iterdir()):
        if not date_dir.is_dir() or date_dir.name.startswith(".") or date_dir.name.startswith("_"):
            continue
        if last_train_date and date_dir.name < last_train_date:
            continue
        for sess_dir in sorted(date_dir.iterdir()):
            manifest = sess_dir / "manifest.jsonl"
            if not manifest.exists():
                continue
            with open(manifest, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        utt = json.loads(line)
                        if utt.get("reviewed") and not utt.get("is_garbage"):
                            if last_train_date is None or utt.get("reviewed_at", "") > last_train_date:
                                count += 1
                    except json.JSONDecodeError:
                        continue
    return count


def get_last_train_date() -> str:
    """Get the date of the last training run from the log."""
    if not RETRAIN_LOG.exists():
        return None
    last_line = None
    with open(RETRAIN_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line:
        try:
            return json.loads(last_line).get("date", None)
        except json.JSONDecodeError:
            pass
    return None


def get_current_model_version() -> str:
    """Get the current production model version."""
    if not FINETUNED_DIR.exists():
        return "small"
    versions = [d.name for d in FINETUNED_DIR.iterdir()
                if d.is_dir() and d.name.startswith("v") and (d / "ct2_model").exists()]
    if not versions:
        return "small"
    versions.sort(key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)
    return versions[-1]


def run_step(description: str, command: list) -> dict:
    """Run a training step as a subprocess."""
    print(f"\n{'─' * 60}")
    print(f"STEP: {description}")
    print(f"{'─' * 60}")
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=3600,
    )
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")
    return {
        "step": description,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }


def log_run(entry: dict):
    """Append a run entry to the retrain log."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def main():
    print("=" * 60)
    print(f"WEEKLY RETRAIN — {datetime.now().isoformat()}")
    print("=" * 60)

    python = sys.executable
    run_entry = {
        "date": datetime.now().isoformat(),
        "steps": [],
    }

    # Step 0: Check if enough data
    last_date = get_last_train_date()
    new_count = count_new_annotations_since(last_date)
    current_version = get_current_model_version()

    print(f"Last training: {last_date or 'never'}")
    print(f"Current model: {current_version}")
    print(f"New annotations since last train: {new_count}")

    if new_count < MIN_NEW_ANNOTATIONS:
        print(f"\nNot enough new data ({new_count} < {MIN_NEW_ANNOTATIONS}). Skipping training.")
        run_entry["skipped"] = True
        run_entry["reason"] = f"Only {new_count} new annotations"
        log_run(run_entry)
        # Still run analysis tools
    else:
        # Step 1: Fine-tune Whisper
        step1 = run_step("Whisper Fine-Tuning", [
            python, "-m", "voice.training.whisper_finetune",
            "--min-samples", str(MIN_NEW_ANNOTATIONS),
            "--epochs", "3",
        ])
        run_entry["steps"].append(step1)

        if step1["success"]:
            # Step 2: A/B test new model vs current
            new_version = get_current_model_version()  # should be the newly trained one
            if new_version != current_version:
                step2 = run_step("A/B Testing", [
                    python, "-m", "voice.training.ab_testing",
                    "--model-a", current_version if current_version != "small" else "small",
                    "--model-b", new_version,
                    "--max-utterances", "200",
                ])
                run_entry["steps"].append(step2)

    # Step 3: Confidence threshold optimization
    step3 = run_step("Confidence Threshold Optimization", [
        python, "-m", "voice.training.confidence_optimizer",
    ])
    run_entry["steps"].append(step3)

    # Step 4: Fuzzy matcher expansion
    step4 = run_step("Fuzzy Matcher Expansion Analysis", [
        python, "-m", "voice.training.matcher_expansion",
    ])
    run_entry["steps"].append(step4)

    # Log results
    log_run(run_entry)

    print(f"\n{'=' * 60}")
    print("WEEKLY RETRAIN COMPLETE")
    print(f"{'=' * 60}")

    # Print summary
    for step in run_entry.get("steps", []):
        status = "✓" if step.get("success") else "✗"
        print(f"  {status} {step['step']}")


if __name__ == "__main__":
    main()
