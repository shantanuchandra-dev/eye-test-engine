#!/usr/bin/env python3
"""Download all models required for the voice pipeline into voice/models/.

Run once before starting the voice server:
    python -m voice.download_models

Downloads into voice/models/:
    - faster-whisper 'large-v3-turbo' (~1.5GB)  → voice/models/whisper-v3-turbo/
    - Piper TTS voices (5 × ~60MB each)         → voice/models/piper/
    - Silero VAD model (~34MB)                   → voice/models/silero/

Also called automatically by run.py if voice/models/ is empty.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Resolve the models directory relative to this file
MODELS_DIR = Path(__file__).resolve().parent / "models"
WHISPER_DIR = MODELS_DIR / "whisper-v3-turbo"
PIPER_DIR = MODELS_DIR / "piper"

# Piper voices to download (name → HuggingFace sub-path)
PIPER_VOICES = {
    "en_US-kusal-medium": "en/en_US/kusal/medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "hi_IN-pratham-medium": "hi/hi_IN/pratham/medium",
    "te_IN-venkatesh-medium": "te/te_IN/venkatesh/medium",
    "ml_IN-arjun-medium": "ml/ml_IN/arjun/medium",
}
PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def download_whisper():
    """Download faster-whisper large-v3-turbo to voice/models/whisper-v3-turbo/."""
    print(f"[1/3] Downloading faster-whisper 'large-v3-turbo' (~1.5GB) → {WHISPER_DIR}")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            "large-v3-turbo",
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_DIR),
        )
        print(f"  OK  large-v3-turbo saved to {WHISPER_DIR}")
        del model
    except Exception as e:
        print(f"  FAIL  {e}")
        return False
    return True


def download_piper():
    """Download Piper TTS voices (5 voices) to voice/models/piper/."""
    print(f"[2/3] Downloading Piper TTS voices ({len(PIPER_VOICES)} voices) → {PIPER_DIR}")
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for voice_name, voice_path in PIPER_VOICES.items():
        for ext in [".onnx", ".onnx.json"]:
            filename = f"{voice_name}{ext}"
            url = f"{PIPER_HF_BASE}/{voice_path}/{filename}"
            dest = PIPER_DIR / filename

            if dest.exists():
                print(f"  Already exists: {voice_name}{ext}")
                continue

            print(f"  Downloading: {voice_name}{ext}...")
            try:
                urllib.request.urlretrieve(url, str(dest))
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  OK  {filename} ({size_mb:.1f}MB)")
            except Exception as e:
                print(f"  FAIL  {filename}: {e}")
                all_ok = False

    return all_ok


SILERO_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def download_silero():
    """Download Silero VAD ONNX model (~2MB) to voice/models/."""
    # Store directly in MODELS_DIR (not a silero/ subdir) so pipeline finds it
    dest = MODELS_DIR / "silero_vad.onnx"
    print(f"[3/3] Downloading Silero VAD ONNX → {dest}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True
    try:
        urllib.request.urlretrieve(SILERO_ONNX_URL, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  OK  silero_vad.onnx ({size_mb:.1f}MB)")
    except Exception as e:
        print(f"  FAIL  {e}")
        return False
    return True


def check_models():
    """Check which models are already downloaded. Returns dict of booleans."""
    return {
        "whisper": WHISPER_DIR.exists() and any(WHISPER_DIR.rglob("model.bin")),
        "piper": PIPER_DIR.exists() and any(PIPER_DIR.glob("*.onnx")),
        "silero": (MODELS_DIR / "silero_vad.onnx").exists(),
    }


def models_ready() -> bool:
    """Return True if all required models are downloaded."""
    status = check_models()
    return all(status.values())


def ensure_models():
    """Download any missing models. Called by run.py on startup."""
    status = check_models()
    if all(status.values()):
        return True

    print("=" * 55)
    print("Voice models missing — downloading automatically...")
    print(f"Target: {MODELS_DIR}")
    print("=" * 55)

    results = []

    if status["whisper"]:
        print(f"[1/3] Whisper large-v3-turbo: already present")
        results.append(True)
    else:
        results.append(download_whisper())

    results.append(download_piper())  # always check — individual voices may be missing

    if status["silero"]:
        print(f"[3/3] Silero VAD: already present")
        results.append(True)
    else:
        results.append(download_silero())

    if all(results):
        print("\nAll models ready!")
        return True
    else:
        print("\nSome downloads failed. Voice may not work correctly.")
        return False


if __name__ == "__main__":
    print("=" * 55)
    print("Voice Pipeline — Model Download")
    print(f"Target: {MODELS_DIR}")
    print("=" * 55)
    print()

    existing = check_models()
    print(f"Status: whisper={'OK' if existing['whisper'] else 'MISSING'} "
          f"piper={'OK' if existing['piper'] else 'MISSING'} "
          f"silero={'OK' if existing['silero'] else 'MISSING'}")
    print()

    results = []

    if existing["whisper"]:
        print(f"[1/3] Whisper large-v3-turbo: already present, skipping")
        results.append(True)
    else:
        results.append(download_whisper())

    results.append(download_piper())

    if existing["silero"]:
        print(f"[3/3] Silero VAD: already present, skipping")
        results.append(True)
    else:
        results.append(download_silero())

    print()
    if all(results):
        print("All models downloaded successfully!")
        print(f"Location: {MODELS_DIR}")
        print()
        print("Models:")
        print(f"  Whisper: large-v3-turbo (STT)")
        print(f"  Piper voices (TTS):")
        for v in PIPER_VOICES:
            print(f"    - {v}")
        print(f"  Silero VAD")
    else:
        print("Some downloads failed. Check the errors above.")
        sys.exit(1)
