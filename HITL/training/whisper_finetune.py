#!/usr/bin/env python3
"""Whisper fine-tuning script for Eye Test Engine domain audio.

Uses HITL-reviewed annotations to fine-tune the faster-whisper model
on domain-specific vocabulary (optometry terms, Indian accents).

Usage:
    python -m voice.training.whisper_finetune [--min-samples 50] [--epochs 3]

Requires: pip install datasets transformers[torch] evaluate jiwer
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FINETUNED_DIR = MODELS_DIR / "whisper-finetuned"
AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"


def load_training_data(min_samples: int = 50):
    """Load reviewed utterances as training pairs (audio_path, transcript)."""
    pairs = []
    if not AUDIO_BASE_DIR.exists():
        return pairs

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

                    # Only use reviewed, non-garbage utterances
                    if not utt.get("reviewed") or utt.get("is_garbage"):
                        continue

                    transcript = utt.get("transcript_whisper", "").strip()
                    if not transcript:
                        continue

                    audio_path = sess_dir / utt.get("audio_file", "")
                    if not audio_path.exists():
                        continue

                    pairs.append({
                        "audio_path": str(audio_path),
                        "transcript": transcript,
                        "language": utt.get("lang", "en"),
                    })

    print(f"[FINETUNE] Found {len(pairs)} reviewed utterances")
    if len(pairs) < min_samples:
        print(f"[FINETUNE] Need at least {min_samples} samples, have {len(pairs)}. Skipping.")
        return []
    return pairs


def get_next_version() -> str:
    """Get the next model version number (v1, v2, ...)."""
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in FINETUNED_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing:
        return "v1"
    nums = [int(v[1:]) for v in existing if v[1:].isdigit()]
    return f"v{max(nums) + 1}" if nums else "v1"


def finetune(pairs: list, epochs: int = 3, batch_size: int = 8):
    """Fine-tune Whisper on the training data.

    Uses HuggingFace transformers WhisperForConditionalGeneration.
    Saves the fine-tuned model as a CTranslate2 model for faster-whisper.
    """
    try:
        import torch
        from datasets import Dataset, Audio
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        import evaluate
    except ImportError as e:
        print(f"[FINETUNE] Missing dependency: {e}")
        print("Install with: pip install datasets transformers[torch] evaluate jiwer")
        return None

    version = get_next_version()
    output_dir = FINETUNED_DIR / version
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FINETUNE] Training version {version} with {len(pairs)} samples, {epochs} epochs")
    print(f"[FINETUNE] Output: {output_dir}")

    # Load base model
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Force decoder language tokens
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    model.config.suppress_tokens = []

    # Build dataset
    ds = Dataset.from_dict({
        "audio": [p["audio_path"] for p in pairs],
        "transcript": [p["transcript"] for p in pairs],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    ds = ds.map(prepare_dataset, remove_columns=["audio", "transcript"])

    # Split train/eval (90/10)
    split = ds.train_test_split(test_size=0.1, seed=42)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=50,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
    )

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Data collator
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Train
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"[FINETUNE] Starting training...")
    train_result = trainer.train()
    print(f"[FINETUNE] Training complete. Metrics: {train_result.metrics}")

    # Save the fine-tuned model
    trainer.save_model(str(output_dir / "hf_model"))
    processor.save_pretrained(str(output_dir / "hf_model"))

    # Convert to CTranslate2 format for faster-whisper
    try:
        import ctranslate2
        print(f"[FINETUNE] Converting to CTranslate2 format...")
        converter = ctranslate2.converters.TransformersConverter(
            str(output_dir / "hf_model")
        )
        converter.convert(str(output_dir / "ct2_model"), quantization="int8")
        print(f"[FINETUNE] CTranslate2 model saved to {output_dir / 'ct2_model'}")
    except Exception as e:
        print(f"[FINETUNE] CTranslate2 conversion failed: {e}")
        print("  The HF model is still available for use.")

    # Save training metadata
    meta = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "num_samples": len(pairs),
        "epochs": epochs,
        "base_model": model_name,
        "metrics": train_result.metrics,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[FINETUNE] Version {version} saved to {output_dir}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on HITL data")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Minimum reviewed samples required to train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    pairs = load_training_data(min_samples=args.min_samples)
    if not pairs:
        print("[FINETUNE] Not enough data. Exiting.")
        sys.exit(0)

    result = finetune(pairs, epochs=args.epochs, batch_size=args.batch_size)
    if result:
        print(f"[FINETUNE] Success! Model at: {result}")
    else:
        print("[FINETUNE] Training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
