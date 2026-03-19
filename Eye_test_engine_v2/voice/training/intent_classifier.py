#!/usr/bin/env python3
"""Audio → Intent classifier for Eye Test Engine.

Fallback classifier that directly maps raw audio to FSM intents
when the fuzzy matcher confidence is below threshold.

Architecture: wav2vec2 feature extraction → small linear classifier head.
Trains on HITL-labeled (audio, intent) pairs.

Usage:
    # Train
    python -m voice.training.intent_classifier --train [--min-per-class 10]

    # Evaluate
    python -m voice.training.intent_classifier --eval

    # Export model for inference
    python -m voice.training.intent_classifier --export
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CLASSIFIER_DIR = MODELS_DIR / "intent-classifier"

# All possible intent labels across all response types
ALL_INTENTS = [
    "READABLE", "NOT_READABLE", "BLURRY",
    "BETTER_1", "BETTER_2", "SAME", "CANT_TELL",
    "RED_CLEARER", "GREEN_CLEARER", "EQUAL",
    "TOP_CLEARER", "BOTTOM_CLEARER",
    "TARGET_OK", "NOT_CLEAR",
]

INTENT_TO_IDX = {intent: i for i, intent in enumerate(ALL_INTENTS)}
IDX_TO_INTENT = {i: intent for intent, i in INTENT_TO_IDX.items()}


def load_labeled_data(min_per_class: int = 10):
    """Load audio + intent pairs from HITL-reviewed data."""
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

                    if not utt.get("reviewed") or utt.get("is_garbage"):
                        continue

                    # Use corrected option if available, else original match
                    intent = utt.get("correct_option") or utt.get("matched_option")
                    if not intent or intent not in INTENT_TO_IDX:
                        continue

                    audio_path = sess_dir / utt.get("audio_file", "")
                    if not audio_path.exists():
                        continue

                    pairs.append({
                        "audio_path": str(audio_path),
                        "intent": intent,
                        "response_type": utt.get("response_type", ""),
                    })

    # Check class distribution
    class_counts = defaultdict(int)
    for p in pairs:
        class_counts[p["intent"]] += 1

    print(f"[CLASSIFIER] Loaded {len(pairs)} labeled utterances")
    print(f"[CLASSIFIER] Class distribution:")
    for intent, count in sorted(class_counts.items()):
        marker = " ⚠ LOW" if count < min_per_class else ""
        print(f"  {intent:>15}: {count}{marker}")

    # Filter out classes with too few samples
    valid_intents = {k for k, v in class_counts.items() if v >= min_per_class}
    if len(valid_intents) < 2:
        print(f"[CLASSIFIER] Need at least 2 classes with {min_per_class}+ samples. Have {len(valid_intents)}.")
        return []

    pairs = [p for p in pairs if p["intent"] in valid_intents]
    return pairs


def extract_features(audio_path: str, target_sr: int = 16000, max_len_sec: float = 5.0):
    """Extract audio features from a file.

    Returns a fixed-length feature vector using simple MFCCs.
    Falls back to raw audio stats if librosa not available.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        return None

    # Resample if needed
    if sr != target_sr:
        try:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * target_sr / sr))
        except ImportError:
            # Simple decimation
            ratio = sr / target_sr
            audio = audio[::int(ratio)]

    # Truncate/pad to fixed length
    max_samples = int(max_len_sec * target_sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))

    # Extract simple features: chunked energy + zero crossing rate
    # This avoids heavy dependencies like librosa
    chunk_size = target_sr // 10  # 100ms chunks
    n_chunks = len(audio) // chunk_size
    features = []

    for i in range(n_chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size]
        # Energy
        energy = np.sqrt(np.mean(chunk ** 2))
        features.append(energy)
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(chunk)))) / (2 * len(chunk))
        features.append(zcr)
        # Spectral centroid approximation
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / target_sr)
        centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
        features.append(centroid / target_sr)  # normalize

    # Pad/truncate to fixed size (50 chunks × 3 features = 150)
    target_features = 150
    features = np.array(features[:target_features], dtype=np.float32)
    if len(features) < target_features:
        features = np.pad(features, (0, target_features - len(features)))

    return features


def train_classifier(pairs: list, epochs: int = 50):
    """Train a simple neural intent classifier.

    Uses a small 2-layer network: 150 → 64 → num_classes.
    Lightweight enough to run on CPU without heavy ML frameworks.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Extract features
    print(f"[CLASSIFIER] Extracting features from {len(pairs)} utterances...")
    X, y = [], []
    for p in pairs:
        feat = extract_features(p["audio_path"])
        if feat is not None:
            X.append(feat)
            y.append(INTENT_TO_IDX[p["intent"]])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Determine active classes
    active_classes = sorted(set(y.tolist()))
    # Remap to contiguous indices
    remap = {old: new for new, old in enumerate(active_classes)}
    y_remapped = np.array([remap[yi] for yi in y], dtype=np.int64)
    num_classes = len(active_classes)

    print(f"[CLASSIFIER] Features: {X.shape}, Classes: {num_classes}")

    # Train/test split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y_remapped[train_idx])
    X_test = torch.from_numpy(X[test_idx])
    y_test = torch.from_numpy(y_remapped[test_idx])

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Model
    model = nn.Sequential(
        nn.Linear(150, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, num_classes),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train
    print(f"[CLASSIFIER] Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test)
                test_pred = test_out.argmax(dim=1)
                test_acc = (test_pred == y_test).float().mean().item()
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_dl):.4f} test_acc={test_acc:.3f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_pred = test_out.argmax(dim=1)
        test_acc = (test_pred == y_test).float().mean().item()

    print(f"\n[CLASSIFIER] Final test accuracy: {test_acc:.3f}")

    # Save model
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    version = f"v{len(list(CLASSIFIER_DIR.glob('*.pt'))) + 1}"
    model_path = CLASSIFIER_DIR / f"intent_model_{version}.pt"

    # Save with metadata
    save_data = {
        "model_state": model.state_dict(),
        "active_classes": active_classes,
        "remap": remap,
        "idx_to_intent": {str(remap[c]): IDX_TO_INTENT[c] for c in active_classes},
        "num_features": 150,
        "num_classes": num_classes,
        "test_accuracy": test_acc,
        "trained_at": datetime.now().isoformat(),
        "num_samples": len(pairs),
        "version": version,
    }
    torch.save(save_data, model_path)
    print(f"[CLASSIFIER] Model saved: {model_path}")

    # Save metadata separately as JSON
    meta = {k: v for k, v in save_data.items() if k != "model_state"}
    with open(CLASSIFIER_DIR / f"intent_model_{version}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return model_path, test_acc


class IntentClassifierInference:
    """Load and run the intent classifier for inference."""

    def __init__(self, model_path: str = None):
        import torch
        import torch.nn as nn

        if model_path is None:
            # Find latest model
            if not CLASSIFIER_DIR.exists():
                raise FileNotFoundError("No intent classifier model found")
            models = sorted(CLASSIFIER_DIR.glob("intent_model_*.pt"))
            if not models:
                raise FileNotFoundError("No intent classifier model found")
            model_path = str(models[-1])

        data = torch.load(model_path, map_location="cpu", weights_only=False)

        num_classes = data["num_classes"]
        self._idx_to_intent = data["idx_to_intent"]

        self._model = nn.Sequential(
            nn.Linear(150, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )
        self._model.load_state_dict(data["model_state"])
        self._model.eval()

    def predict(self, audio_path: str) -> tuple:
        """Predict intent from audio file.

        Returns (intent_label, confidence) or (None, 0.0) if failed.
        """
        import torch

        features = extract_features(audio_path)
        if features is None:
            return None, 0.0

        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        intent = self._idx_to_intent.get(str(idx.item()))
        return intent, conf.item() * 100


def main():
    parser = argparse.ArgumentParser(description="Intent classifier for eye test audio")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument("--eval", action="store_true", help="Evaluate latest model")
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.train:
        pairs = load_labeled_data(min_per_class=args.min_per_class)
        if not pairs:
            print("Not enough labeled data.")
            sys.exit(0)
        train_classifier(pairs, epochs=args.epochs)

    elif args.eval:
        try:
            clf = IntentClassifierInference()
            pairs = load_labeled_data(min_per_class=1)
            if not pairs:
                print("No data to evaluate.")
                sys.exit(0)
            correct = 0
            total = 0
            for p in pairs:
                pred, conf = clf.predict(p["audio_path"])
                expected = p["intent"]
                if pred == expected:
                    correct += 1
                total += 1
            print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
