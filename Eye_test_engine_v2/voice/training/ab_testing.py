#!/usr/bin/env python3
"""A/B testing framework for comparing Whisper model versions.

Runs two models on the same set of reviewed audio utterances and compares
accuracy, review rate, and match quality.

Usage:
    python -m voice.training.ab_testing --model-a small --model-b v1

Where model-a/b can be:
    - "small" (base faster-whisper model)
    - "v1", "v2", ... (fine-tuned versions in voice/models/whisper-finetuned/)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WHISPER_BASE_DIR = MODELS_DIR / "whisper-small"
FINETUNED_DIR = MODELS_DIR / "whisper-finetuned"
AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"


def resolve_model_path(model_name: str) -> str:
    """Resolve a model name to a path for faster-whisper."""
    if model_name in ("small", "base", "tiny", "medium", "large"):
        # Base model — check local cache first
        model_bins = list(WHISPER_BASE_DIR.rglob("model.bin"))
        if model_bins:
            return str(model_bins[0].parent)
        return model_name  # will download

    # Fine-tuned version
    ct2_path = FINETUNED_DIR / model_name / "ct2_model"
    if ct2_path.exists():
        return str(ct2_path)

    hf_path = FINETUNED_DIR / model_name / "hf_model"
    if hf_path.exists():
        return str(hf_path)

    print(f"[A/B] Model '{model_name}' not found at {FINETUNED_DIR / model_name}")
    sys.exit(1)


def load_test_utterances(max_count: int = 200):
    """Load reviewed utterances with audio files for testing."""
    utterances = []
    if not AUDIO_BASE_DIR.exists():
        return utterances

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

                    if not utt.get("reviewed") or utt.get("is_garbage"):
                        continue

                    audio_path = sess_dir / utt.get("audio_file", "")
                    if not audio_path.exists():
                        continue

                    utt["_audio_path"] = str(audio_path)
                    utterances.append(utt)

                    if len(utterances) >= max_count:
                        return utterances

    return utterances


def transcribe_with_model(model, utterances):
    """Run transcription on all utterances with a given model."""
    import soundfile as sf

    results = []
    total_time = 0

    for utt in utterances:
        audio_path = utt["_audio_path"]
        audio, sr = sf.read(audio_path, dtype="float32")

        if sr != 16000:
            # Resample
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * 16000 / sr))

        start = time.time()
        segments, _ = model.transcribe(
            audio, language=utt.get("lang", "en"),
            beam_size=1, vad_filter=True,
        )
        transcript = " ".join(seg.text for seg in segments).strip()
        elapsed = time.time() - start
        total_time += elapsed

        results.append({
            "id": utt.get("id"),
            "transcript": transcript,
            "reference": utt.get("transcript_whisper", ""),
            "time_sec": round(elapsed, 3),
        })

    return results, total_time


def compare_results(results_a, results_b, utterances):
    """Compare transcription results from two models against ground truth."""
    from voice.fuzzy_matcher import match_transcript

    metrics_a = {"correct": 0, "wrong": 0, "no_match": 0, "total_time": 0}
    metrics_b = {"correct": 0, "wrong": 0, "no_match": 0, "total_time": 0}

    details = []

    for i, utt in enumerate(utterances):
        response_type = utt.get("response_type", "")
        correct_option = utt.get("correct_option") or utt.get("matched_option")

        # Model A
        transcript_a = results_a[i]["transcript"]
        match_a, conf_a = match_transcript(transcript_a, response_type)
        metrics_a["total_time"] += results_a[i]["time_sec"]

        if match_a == correct_option:
            metrics_a["correct"] += 1
        elif match_a:
            metrics_a["wrong"] += 1
        else:
            metrics_a["no_match"] += 1

        # Model B
        transcript_b = results_b[i]["transcript"]
        match_b, conf_b = match_transcript(transcript_b, response_type)
        metrics_b["total_time"] += results_b[i]["time_sec"]

        if match_b == correct_option:
            metrics_b["correct"] += 1
        elif match_b:
            metrics_b["wrong"] += 1
        else:
            metrics_b["no_match"] += 1

        # Track disagreements
        if match_a != match_b:
            details.append({
                "id": utt.get("id"),
                "correct": correct_option,
                "model_a": {"transcript": transcript_a, "match": match_a, "conf": conf_a},
                "model_b": {"transcript": transcript_b, "match": match_b, "conf": conf_b},
            })

    total = len(utterances)
    for m in [metrics_a, metrics_b]:
        m["accuracy"] = round(m["correct"] / total * 100, 1) if total else 0
        m["review_rate"] = round((m["wrong"] + m["no_match"]) / total * 100, 1) if total else 0
        m["avg_time"] = round(m["total_time"] / total, 3) if total else 0

    return metrics_a, metrics_b, details


def print_comparison(name_a, name_b, metrics_a, metrics_b, details, total):
    """Pretty-print the A/B comparison."""
    print(f"\n{'=' * 60}")
    print(f"A/B TEST RESULTS ({total} utterances)")
    print(f"{'=' * 60}")
    print(f"  {'':>20} {'Model A':>12} {'Model B':>12} {'Winner':>10}")
    print(f"  {'':>20} {'(' + name_a + ')':>12} {'(' + name_b + ')':>12}")
    print(f"  {'-' * 56}")

    def winner(a, b, higher_better=True):
        if a == b:
            return "Tie"
        return f"{'A' if (a > b) == higher_better else 'B'} wins"

    print(f"  {'Accuracy':>20} {metrics_a['accuracy']:>11}% {metrics_b['accuracy']:>11}% {winner(metrics_a['accuracy'], metrics_b['accuracy']):>10}")
    print(f"  {'Review Rate':>20} {metrics_a['review_rate']:>11}% {metrics_b['review_rate']:>11}% {winner(metrics_a['review_rate'], metrics_b['review_rate'], False):>10}")
    print(f"  {'Avg Time/utt':>20} {metrics_a['avg_time']:>10}s {metrics_b['avg_time']:>10}s {winner(metrics_a['avg_time'], metrics_b['avg_time'], False):>10}")
    print(f"  {'Correct':>20} {metrics_a['correct']:>12} {metrics_b['correct']:>12}")
    print(f"  {'Wrong':>20} {metrics_a['wrong']:>12} {metrics_b['wrong']:>12}")
    print(f"  {'No Match':>20} {metrics_a['no_match']:>12} {metrics_b['no_match']:>12}")

    if details:
        print(f"\n  Disagreements ({len(details)}):")
        for d in details[:10]:
            print(f"    {d['id']}: correct={d['correct']}")
            print(f"      A: \"{d['model_a']['transcript'][:40]}\" → {d['model_a']['match']}")
            print(f"      B: \"{d['model_b']['transcript'][:40]}\" → {d['model_b']['match']}")

    # Recommendation
    print(f"\n{'=' * 60}")
    if metrics_a["accuracy"] > metrics_b["accuracy"] and metrics_a["review_rate"] <= metrics_b["review_rate"]:
        print(f"  RECOMMENDATION: Model A ({name_a}) is better. Promote it.")
    elif metrics_b["accuracy"] > metrics_a["accuracy"] and metrics_b["review_rate"] <= metrics_a["review_rate"]:
        print(f"  RECOMMENDATION: Model B ({name_b}) is better. Promote it.")
    else:
        print(f"  RECOMMENDATION: Mixed results. Review disagreements manually.")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="A/B test Whisper models")
    parser.add_argument("--model-a", default="small", help="Model A name/version")
    parser.add_argument("--model-b", required=True, help="Model B name/version")
    parser.add_argument("--max-utterances", type=int, default=200)
    args = parser.parse_args()

    utterances = load_test_utterances(max_count=args.max_utterances)
    print(f"Loaded {len(utterances)} test utterances")

    if len(utterances) < 5:
        print("Not enough reviewed utterances for A/B testing.")
        sys.exit(0)

    from faster_whisper import WhisperModel

    path_a = resolve_model_path(args.model_a)
    path_b = resolve_model_path(args.model_b)

    print(f"\nModel A ({args.model_a}): {path_a}")
    print(f"Model B ({args.model_b}): {path_b}")

    print(f"\nTranscribing with Model A...")
    model_a = WhisperModel(path_a, device="cpu", compute_type="int8")
    results_a, time_a = transcribe_with_model(model_a, utterances)
    del model_a

    print(f"Transcribing with Model B...")
    model_b = WhisperModel(path_b, device="cpu", compute_type="int8")
    results_b, time_b = transcribe_with_model(model_b, utterances)
    del model_b

    metrics_a, metrics_b, details = compare_results(results_a, results_b, utterances)
    print_comparison(args.model_a, args.model_b, metrics_a, metrics_b, details, len(utterances))

    # Save results
    output = {
        "tested_at": datetime.now().isoformat(),
        "model_a": {"name": args.model_a, "path": path_a, "metrics": metrics_a},
        "model_b": {"name": args.model_b, "path": path_b, "metrics": metrics_b},
        "disagreements": details,
        "total_utterances": len(utterances),
    }
    output_path = AUDIO_BASE_DIR / "_analysis" / "ab_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
