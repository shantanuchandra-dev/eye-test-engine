#!/usr/bin/env python3
"""Download models required for the voice pipeline into voice/models/.

Run once before starting the voice server:
    python -m voice.download_models

Downloads into voice/models/:
    - faster-whisper 'small' model (~460MB)  → voice/models/whisper-small/
    - Piper TTS voices (~60MB each)          → voice/models/piper/
    - Silero VAD model (~2MB)                → voice/models/silero/
"""

import os
import sys
import urllib.request
from pathlib import Path

# Resolve the models directory relative to this file
MODELS_DIR = Path(__file__).resolve().parent / "models"
WHISPER_DIR = MODELS_DIR / "whisper-v3-turbo"
PIPER_DIR = MODELS_DIR / "piper"
SILERO_DIR = MODELS_DIR / "silero"

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
    """Download the faster-whisper model to voice/models/whisper-small/."""
    print(f"[1/3] Downloading faster-whisper 'large-v3-turbo' model → {WHISPER_DIR}")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            "large-v3-turbo",
            device="cpu",
            compute_type="int8",
            download_root=str(WHISPER_DIR),
        )
        print(f"  OK  faster-whisper 'large-v3-turbo' model saved to {WHISPER_DIR}")
        del model
    except Exception as e:
        print(f"  FAIL  {e}")
        return False
    return True


def download_piper():
    """Download Piper TTS voices to voice/models/piper/."""
    print(f"[2/3] Downloading Piper TTS voices → {PIPER_DIR}")
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


def download_silero():
    """Download the Silero VAD model to voice/models/silero/."""
    print(f"[3/3] Downloading Silero VAD model → {SILERO_DIR}")
    SILERO_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(str(SILERO_DIR))
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        torch.hub.set_dir(original_hub_dir)
        print(f"  OK  Silero VAD model saved to {SILERO_DIR}")
        del model
    except Exception as e:
        print(f"  FAIL  {e}")
        return False
    return True


def check_models():
    """Check which models are already downloaded."""
    status = {}
    status["whisper"] = WHISPER_DIR.exists() and any(WHISPER_DIR.iterdir())
    status["piper"] = PIPER_DIR.exists() and any(PIPER_DIR.iterdir())
    status["silero"] = SILERO_DIR.exists() and any(SILERO_DIR.iterdir())
    return status


if __name__ == "__main__":
    print("=" * 55)
    print("Voice Pipeline — Local Model Download")
    print(f"Target: {MODELS_DIR}")
    print("=" * 55)
    print()

    existing = check_models()
    results = []

    if existing["whisper"]:
        print(f"[1/3] faster-whisper already present in {WHISPER_DIR}, skipping")
        results.append(True)
    else:
        results.append(download_whisper())

    # Always check Piper — individual voices may be missing
    results.append(download_piper())

    if existing["silero"]:
        print(f"[3/3] Silero VAD already present in {SILERO_DIR}, skipping")
        results.append(True)
    else:
        results.append(download_silero())

    print()
    if all(results):
        print("All models downloaded successfully!")
        print(f"Location: {MODELS_DIR}")
        print()
        print("Voices available:")
        for v in PIPER_VOICES:
            print(f"  - {v}")
    else:
        print("Some downloads failed. Check the errors above.")
        sys.exit(1)
