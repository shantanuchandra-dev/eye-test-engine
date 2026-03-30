"""
Build and maintain the pre-rendered ElevenLabs TTS cache (FSM + UI phrases → MP3).

The Flask app serves files from tts_cache/ as GET /api/tts/<sha256>.mp3; Sarvam clips
live in sarvam_tts_cache/ as GET /api/tts-sarvam/<sha256>.mp3 (see sarvam_tts_cache.py).
The browser falls back to speechSynthesis when a clip is missing.

Generate cache (requires ELEVENLABS_API_KEY and network):
  cd ETE_v2 && python elevenlabs_tts_cache.py generate

Environment:
  ELEVENLABS_API_KEY   — required for generate / play
  ELEVENLABS_TTS_VOICE_ID — voice ID (default: ARIA_VOICE_ID below; free tier needs My Voices ID)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Free tier: API cannot use preset library voices (e.g. Aria). Use a voice from
# ElevenLabs → Voices → My Voices (or create/clone one), copy its ID, and set:
#   ELEVENLABS_TTS_VOICE_ID=<voice_id>
ARIA_VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"

VOICE_ID = os.environ.get("ELEVENLABS_TTS_VOICE_ID", ARIA_VOICE_ID)
MODEL_ID = os.environ.get("ELEVENLABS_TTS_MODEL_ID", "eleven_multilingual_v2")

_ETE_ROOT = Path(__file__).resolve().parent
DEFAULT_TTS_CACHE_DIR = _ETE_ROOT / "tts_cache"


def phrase_hash(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _get_client():
    from elevenlabs.client import ElevenLabs

    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set ELEVENLABS_API_KEY in the environment or .env")
    return ElevenLabs(api_key=key)


def _write_mp3_from_convert(audio_stream, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        if isinstance(audio_stream, (bytes, bytearray)):
            f.write(audio_stream)
            return
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)


def generate_fsm_audio_cache(
    output_dir: Path | None = None,
    *,
    skip_existing: bool = True,
) -> dict:
    """
    Generate MP3 files named <sha256>.mp3 for every phrase in fsm_tts_phrases.
    Returns a small report dict.
    """
    from elevenlabs.core.api_error import ApiError

    from fsm_tts_phrases import collect_all_tts_phrases

    out = Path(output_dir or DEFAULT_TTS_CACHE_DIR)
    client = _get_client()
    phrases = collect_all_tts_phrases()
    created = 0
    skipped = 0
    errors: list[str] = []

    for text in phrases:
        dest = out / f"{phrase_hash(text)}.mp3"
        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            skipped += 1
            continue
        try:
            audio = client.text_to_speech.convert(
                voice_id=VOICE_ID,
                text=text,
                model_id=MODEL_ID,
            )
            _write_mp3_from_convert(audio, dest)
            created += 1
        except ApiError as e:
            msg = f"{phrase_hash(text)[:12]}… ({e.status_code}): {e}"
            errors.append(msg)
            if e.status_code == 402:
                raise RuntimeError(
                    "ElevenLabs returned 402: library voices may be unavailable on the free API. "
                    "Set ELEVENLABS_TTS_VOICE_ID in .env to a voice ID from My Voices, "
                    "or upgrade your ElevenLabs plan."
                ) from e
        except Exception as e:
            errors.append(f"{phrase_hash(text)[:12]}…: {e!r}")

    manifest = {
        "voice_id": VOICE_ID,
        "model_id": MODEL_ID,
        "phrase_count": len(phrases),
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def play_sample(text: str | None = None) -> None:
    """One-off playback (dev check)."""
    from elevenlabs.core.api_error import ApiError
    from elevenlabs.play import play

    sample = text or (
        "Thanks for reaching out today. I wanted to walk you through what we found in the review "
        "and what happens next on our side."
    )
    client = _get_client()
    try:
        audio = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=sample,
            model_id=MODEL_ID,
        )
        play(audio)
    except ApiError as e:
        if e.status_code == 402:
            raise RuntimeError(
                "ElevenLabs returned 402: library voices are not available on the free API. "
                "Set ELEVENLABS_TTS_VOICE_ID in .env to a voice ID from My Voices, "
                "or upgrade your ElevenLabs plan."
            ) from e
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ElevenLabs FSM TTS cache builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Write MP3s for all FSM/UI phrases")
    p_gen.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TTS_CACHE_DIR,
        help="Directory for *.mp3 (default: ETE_v2/tts_cache)",
    )
    p_gen.add_argument(
        "--regenerate",
        action="store_true",
        help="Overwrite existing MP3 files",
    )

    p_play = sub.add_parser("play", help="Speak a sample line (smoke test)")
    p_play.add_argument("text", nargs="?", default=None, help="Optional text to speak")

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        report = generate_fsm_audio_cache(args.output, skip_existing=not args.regenerate)
        print(json.dumps(report, indent=2))
        if report.get("errors"):
            return 1
        return 0
    if args.cmd == "play":
        play_sample(args.text)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
