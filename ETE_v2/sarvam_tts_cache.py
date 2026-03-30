"""
Build and maintain pre-rendered Sarvam AI TTS cache (same phrase set as ElevenLabs).

MP3 files live under sarvam_tts_cache/ as <sha256>.mp3 (UTF-8 phrase text, same as fsm_tts_phrases).
The Flask app serves them as GET /api/tts-sarvam/<sha256>.mp3.

Generate (requires SARVAM_API_KEY and network):
  cd ETE_v2 && python sarvam_tts_cache.py generate

Retrim cached MP3 tails (ffmpeg, fixes trailing click/hum after Sarvam speech; no API):
  cd ETE_v2 && python sarvam_tts_cache.py retrim
  cd ETE_v2 && python sarvam_tts_cache.py retrim --hash 67aff576

Environment:
  SARVAM_API_KEY          — required (dashboard: https://dashboard.sarvam.ai)
  SARVAM_TTS_SPEAKER      — bulbul:v3 speaker, default shubh (lowercase)
  SARVAM_TTS_MODEL        — default bulbul:v3
  SARVAM_MP3_NO_TRIM      — set to 1 to skip ffmpeg tail trim after download
  SARVAM_MP3_TRIM_THRESHOLD_DB — default -35 (noise floor for tail trim)
  SARVAM_MP3_TRIM_MIN_DURATION — default 0.08 seconds of trailing silence to remove
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

_ETE_ROOT = Path(__file__).resolve().parent
DEFAULT_SARVAM_CACHE_DIR = _ETE_ROOT / "sarvam_tts_cache"

SPEAKER = os.environ.get("SARVAM_TTS_SPEAKER", "Ishita").strip().lower()
MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3").strip()


def phrase_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _target_language_code(text: str) -> str:
    """Use hi-IN if text contains Devanagari, else en-IN (code-mixed handled by model)."""
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            return "hi-IN"
    return "en-IN"


def synthesize_sarvam_mp3_bytes(text: str, api_key: str) -> bytes:
    payload = {
        "text": text,
        "target_language_code": _target_language_code(text),
        "speaker": SPEAKER,
        "model": MODEL,
        "output_audio_codec": "mp3",
        "pace": 1.0,
    }
    r = requests.post(
        SARVAM_TTS_URL,
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    try:
        data = r.json()
    except Exception:
        r.raise_for_status()
        raise RuntimeError(f"Sarvam TTS: non-JSON response ({r.status_code})") from None

    if r.status_code >= 400:
        err = data.get("error") if isinstance(data, dict) else None
        msg = err.get("message", data) if isinstance(err, dict) else data
        raise RuntimeError(f"Sarvam TTS HTTP {r.status_code}: {msg}")

    audios = data.get("audios") or []
    if not audios:
        raise RuntimeError("Sarvam TTS returned empty audios")

    return base64.b64decode("".join(audios))


def postprocess_sarvam_mp3_bytes(mp3_bytes: bytes) -> bytes:
    """Trim trailing silence / low-level noise common at end of Sarvam MP3s (requires ffmpeg)."""
    if not mp3_bytes:
        return mp3_bytes
    if os.environ.get("SARVAM_MP3_NO_TRIM", "").strip().lower() in ("1", "true", "yes"):
        return mp3_bytes
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return mp3_bytes

    min_dur = os.environ.get("SARVAM_MP3_TRIM_MIN_DURATION", "0.08").strip()
    thr = os.environ.get("SARVAM_MP3_TRIM_THRESHOLD_DB", "-35").strip()
    if not thr.endswith("dB"):
        thr = f"{thr}dB"
    # Reverse → trim leading silence (tail of original) → reverse back
    af = (
        f"areverse,silenceremove=start_periods=1:start_duration={min_dur}:"
        f"start_threshold={thr}:detection=peak,areverse"
    )
    in_path: str | None = None
    out_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fin:
            fin.write(mp3_bytes)
            in_path = fin.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fout:
            out_path = fout.name
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            in_path,
            "-af",
            af,
            "-c:a",
            "libmp3lame",
            "-q:a",
            "3",
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0:
            return mp3_bytes
        out = Path(out_path).read_bytes()
        return out if len(out) > 100 else mp3_bytes
    except (OSError, subprocess.SubprocessError):
        return mp3_bytes
    finally:
        for p in (in_path, out_path):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass


def retrim_cache_dir(
    cache_dir: Path,
    *,
    hash_prefix: str = "",
) -> dict:
    """Re-run tail trim on existing *.mp3 (no Sarvam API)."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise RuntimeError(f"Not a directory: {cache_dir}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH (required for retrim)")

    hp = hash_prefix.strip().lower()
    updated = 0
    skipped = 0
    errors: list[str] = []

    for path in sorted(cache_dir.glob("*.mp3")):
        name = path.stem.lower()
        if hp and not name.startswith(hp):
            skipped += 1
            continue
        try:
            raw = path.read_bytes()
            new = postprocess_sarvam_mp3_bytes(raw)
            if len(new) < 100:
                errors.append(f"{path.name}: trim produced empty or tiny output")
                continue
            path.write_bytes(new)
            updated += 1
        except OSError as e:
            errors.append(f"{path.name}: {e!r}")

    return {"updated": updated, "skipped": skipped, "errors": errors}


def generate_fsm_audio_cache(
    output_dir: Path | None = None,
    *,
    skip_existing: bool = True,
) -> dict:
    from fsm_tts_phrases import collect_all_tts_phrases

    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set SARVAM_API_KEY in the environment or ETE_v2/.env")

    out = Path(output_dir or DEFAULT_SARVAM_CACHE_DIR)
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
            raw = synthesize_sarvam_mp3_bytes(text, key)
            raw = postprocess_sarvam_mp3_bytes(raw)
            out.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            created += 1
        except Exception as e:
            errors.append(f"{phrase_hash(text)[:12]}…: {e!r}")

    manifest = {
        "provider": "sarvam",
        "speaker": SPEAKER,
        "model": MODEL,
        "phrase_count": len(phrases),
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sarvam AI FSM TTS cache builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Write MP3s for all FSM/UI phrases")
    p_gen.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SARVAM_CACHE_DIR,
        help="Directory for *.mp3 (default: ETE_v2/sarvam_tts_cache)",
    )
    p_gen.add_argument(
        "--regenerate",
        action="store_true",
        help="Overwrite existing MP3 files",
    )

    p_trim = sub.add_parser(
        "retrim",
        help="Trim trailing silence/noise on cached MP3s via ffmpeg (no API calls)",
    )
    p_trim.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SARVAM_CACHE_DIR,
        help="Cache directory (default: ETE_v2/sarvam_tts_cache)",
    )
    p_trim.add_argument(
        "--hash",
        type=str,
        default="",
        help="Only files whose name starts with this hex prefix (e.g. 67aff576)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        report = generate_fsm_audio_cache(args.output, skip_existing=not args.regenerate)
        print(json.dumps(report, indent=2))
        return 1 if report.get("errors") else 0
    if args.cmd == "retrim":
        report = retrim_cache_dir(args.output, hash_prefix=args.hash)
        print(json.dumps(report, indent=2))
        return 1 if report.get("errors") else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
