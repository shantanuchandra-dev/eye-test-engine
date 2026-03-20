#!/usr/bin/env python3
"""Confidence threshold optimizer for the fuzzy matcher.

Analyzes HITL-reviewed annotations to find the optimal confidence threshold
per response_type. Balances false positive rate vs false negative rate.

Usage:
    python -m voice.training.confidence_optimizer
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"


def load_reviewed_data():
    """Load all reviewed, non-garbage utterances."""
    data = []
    if not AUDIO_BASE_DIR.exists():
        return data

    for date_dir in sorted(AUDIO_BASE_DIR.iterdir()):
        if not date_dir.is_dir() or date_dir.name.startswith(".") or date_dir.name.startswith("_"):
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
                    except json.JSONDecodeError:
                        continue
                    if utt.get("reviewed") and not utt.get("is_garbage"):
                        data.append(utt)
    return data


def compute_metrics_at_threshold(data, threshold):
    """Compute accuracy metrics at a given confidence threshold.

    Returns dict with: true_positive, false_positive, false_negative,
    true_negative, precision, recall, f1.
    """
    tp = fp = fn = tn = 0

    for utt in data:
        confidence = utt.get("confidence", 0)
        matched = utt.get("matched_option")
        correct = utt.get("correct_option") or matched  # None means original was correct

        if confidence >= threshold and matched:
            # System accepted the match
            if correct == matched or correct is None:
                tp += 1  # Correctly accepted
            else:
                fp += 1  # Wrongly accepted (matcher was wrong)
        else:
            # System rejected (would ask to repeat)
            if matched and (correct == matched or correct is None):
                fn += 1  # Wrongly rejected (matcher was actually right)
            else:
                tn += 1  # Correctly rejected

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "total": total,
    }


def find_optimal_threshold(data, response_type=None):
    """Find optimal threshold by sweeping 40-100 in steps of 5.

    Optimizes for F1 score (balance of precision and recall).
    """
    if response_type:
        data = [u for u in data if u.get("response_type") == response_type]

    if len(data) < 5:
        return None

    best = None
    results = []
    for threshold in range(40, 101, 5):
        metrics = compute_metrics_at_threshold(data, threshold)
        results.append(metrics)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    return {"optimal": best, "sweep": results, "response_type": response_type, "sample_count": len(data)}


def run_analysis():
    """Run full threshold analysis and print results."""
    data = load_reviewed_data()
    print(f"Loaded {len(data)} reviewed utterances\n")

    if len(data) < 5:
        print("Not enough reviewed data for analysis. Need at least 5.")
        return {}

    # Global analysis
    print("=" * 70)
    print("GLOBAL THRESHOLD ANALYSIS")
    print("=" * 70)
    global_result = find_optimal_threshold(data)
    if global_result:
        opt = global_result["optimal"]
        print(f"  Optimal threshold: {opt['threshold']}%")
        print(f"  F1: {opt['f1']:.3f}  Precision: {opt['precision']:.3f}  Recall: {opt['recall']:.3f}")
        print(f"  Accuracy: {opt['accuracy']:.3f}  (TP:{opt['true_positive']} FP:{opt['false_positive']} FN:{opt['false_negative']} TN:{opt['true_negative']})")
        print()
        print("  Threshold sweep:")
        print(f"  {'Thresh':>7} {'F1':>6} {'Prec':>6} {'Recall':>7} {'Acc':>6} {'FP':>4} {'FN':>4}")
        for r in global_result["sweep"]:
            print(f"  {r['threshold']:>6}% {r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>7.3f} {r['accuracy']:>6.3f} {r['false_positive']:>4} {r['false_negative']:>4}")

    # Per response_type analysis
    response_types = set(u.get("response_type", "") for u in data)
    per_type_results = {}

    print(f"\n{'=' * 70}")
    print("PER RESPONSE-TYPE ANALYSIS")
    print("=" * 70)

    for rt in sorted(response_types):
        if not rt:
            continue
        result = find_optimal_threshold(data, response_type=rt)
        if result and result["optimal"]:
            per_type_results[rt] = result
            opt = result["optimal"]
            print(f"\n  {rt} ({result['sample_count']} samples)")
            print(f"    Optimal: {opt['threshold']}%  F1: {opt['f1']:.3f}  Prec: {opt['precision']:.3f}  Recall: {opt['recall']:.3f}")

    # Summary recommendation
    print(f"\n{'=' * 70}")
    print("RECOMMENDED THRESHOLDS")
    print("=" * 70)
    print(f"  Global: {global_result['optimal']['threshold']}%")
    for rt, result in sorted(per_type_results.items()):
        print(f"  {rt}: {result['optimal']['threshold']}%")

    # Save results
    output = {
        "analyzed_at": str(Path(".")),
        "total_reviewed": len(data),
        "global": global_result,
        "per_response_type": per_type_results,
    }
    output_path = AUDIO_BASE_DIR / "_analysis" / "confidence_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_path}")

    return output


if __name__ == "__main__":
    run_analysis()
