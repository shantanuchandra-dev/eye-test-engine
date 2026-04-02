"""
Build and maintain pre-rendered Sarvam AI TTS cache (FSM + UI phrases from fsm_tts_phrases).

MP3 files live under sarvam_tts_cache/ (Shruti) or sarvam_tts_cache_ishita/ (Ishita) as <sha256>.mp3
(UTF-8 phrase text, same as fsm_tts_phrases). The Flask app serves them as
GET /api/tts-sarvam/<sha256>.mp3?speaker=ishita|shruti when present;
with SARVAM_API_KEY it also synthesizes on the fly for known phrases, and POST /api/tts-sarvam/synthesize
handles arbitrary exam text (optional JSON speaker).

Generate (requires SARVAM_API_KEY and network):
  cd ETE_v2 && python sarvam_tts_cache.py generate
  cd ETE_v2 && python sarvam_tts_cache.py generate --speaker shruti --output sarvam_tts_cache
  cd ETE_v2 && python sarvam_tts_cache.py generate --speaker ishita --output sarvam_tts_cache_ishita
  cd ETE_v2 && python sarvam_tts_cache.py generate --regenerate   # overwrite all clips
  cd ETE_v2 && python sarvam_tts_cache.py generate --regenerate --no-trim  # raw Sarvam MP3s, no ffmpeg
  cd ETE_v2 && python sarvam_tts_cache.py generate --regenerate --prune-after  # refresh + delete orphans

Regenerate a single clip by SHA-256 (hex) of the UTF-8 phrase (must match fsm_tts_phrases):
  cd ETE_v2 && python sarvam_tts_cache.py generate-one f36cefa56a15c17ec64837a8228b418c9570366d1be3db4f4e5c5a43ed74e6ce
  cd ETE_v2 && python sarvam_tts_cache.py generate-one f36cefa56a15   # unique prefix ok
  cd ETE_v2 && python sarvam_tts_cache.py generate-one HASH --no-trim  # raw Sarvam MP3, no ffmpeg trim/pad

Prune MP3s whose text is no longer in fsm_tts_phrases (no API):
  cd ETE_v2 && python sarvam_tts_cache.py prune
  cd ETE_v2 && python sarvam_tts_cache.py prune --dry-run

Retrim cached MP3 tails (ffmpeg, fixes trailing click/hum after Sarvam speech; no API):
  cd ETE_v2 && python sarvam_tts_cache.py retrim
  cd ETE_v2 && python sarvam_tts_cache.py retrim --hash 67aff576

Environment:
  SARVAM_API_KEY          — required (dashboard: https://dashboard.sarvam.ai)
  SARVAM_TTS_SPEAKER      — bulbul:v3 speaker id for CLI default (ishita or shruti; default from fsm_tts_phrases)
  SARVAM_TTS_MODEL        — default bulbul:v3
  SARVAM_MP3_NO_TRIM      — set to 1 to skip ffmpeg post-process (trim + pad) after download
  SARVAM_MP3_TRIM_THRESHOLD_DB — default -50 (only trim quiet tail noise; -35 was too aggressive)
  SARVAM_MP3_TRIM_MIN_DURATION — default 0.15 s of trailing silence before trim cuts (was 0.08)
  SARVAM_MP3_END_PAD_SEC  — default 0.12 s silence appended after trim (less abrupt stop); set 0 to disable
  SARVAM_TTS_PACE         — default 1.1 (bulbul:v3 allows 0.5–2.0)
  SARVAM_SPEECH_SAMPLE_RATE — default 48000 (Hz; REST supports 8000–48000 strings)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import unicodedata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from fsm_tts_phrases import (
    DEFAULT_SARVAM_TTS_SPEAKER_ID,
    SARVAM_TTS_SPEAKER_IDS,
    sarvam_cache_dir_basename,
)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

_ETE_ROOT = Path(__file__).resolve().parent


def default_sarvam_speaker_id() -> str:
    raw = os.environ.get("SARVAM_TTS_SPEAKER", "").strip().lower()
    if raw in SARVAM_TTS_SPEAKER_IDS:
        return raw
    return DEFAULT_SARVAM_TTS_SPEAKER_ID


def resolve_sarvam_speaker_id(speaker: str | None) -> str:
    sid = (speaker or "").strip().lower()
    if sid not in SARVAM_TTS_SPEAKER_IDS:
        raise ValueError(
            f"speaker must be one of {sorted(SARVAM_TTS_SPEAKER_IDS)}, got {speaker!r}"
        )
    return sid


def default_sarvam_cache_dir() -> Path:
    return _ETE_ROOT / sarvam_cache_dir_basename(default_sarvam_speaker_id())


# Back-compat: legacy single-cache default path (Shruti tree)
DEFAULT_SARVAM_CACHE_DIR = _ETE_ROOT / "sarvam_tts_cache"

MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3").strip()
# Default 1.1× speed, 48 kHz output (bulbul:v3 REST); override via env if needed.
SARVAM_TTS_PACE = float(os.environ.get("SARVAM_TTS_PACE", "1.1").strip() or "1.1")
SARVAM_SPEECH_SAMPLE_RATE = os.environ.get("SARVAM_SPEECH_SAMPLE_RATE", "48000").strip() or "48000"


def phrase_hash(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _target_language_code(text: str) -> str:
    """Use hi-IN if text contains Devanagari, else en-IN (code-mixed handled by model)."""
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            return "hi-IN"
    return "en-IN"


def synthesize_sarvam_mp3_bytes(
    text: str,
    api_key: str,
    *,
    speaker: str | None = None,
) -> bytes:
    sid = resolve_sarvam_speaker_id(speaker) if speaker else default_sarvam_speaker_id()
    payload = {
        "text": text,
        "target_language_code": _target_language_code(text),
        "speaker": sid,
        "model": MODEL,
        "output_audio_codec": "mp3",
        "pace": SARVAM_TTS_PACE,
        "speech_sample_rate": SARVAM_SPEECH_SAMPLE_RATE,
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
    """Trim trailing silence/noise (gentle defaults) and add a short end pad so clips don't sound chopped."""
    if not mp3_bytes:
        return mp3_bytes
    if os.environ.get("SARVAM_MP3_NO_TRIM", "").strip().lower() in ("1", "true", "yes"):
        return mp3_bytes
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return mp3_bytes

    # Gentler than -35 dB / 0.08 s — those settings often ate the tail of words (fricatives, Hindi endings).
    min_dur = os.environ.get("SARVAM_MP3_TRIM_MIN_DURATION", "0.15").strip()
    thr = os.environ.get("SARVAM_MP3_TRIM_THRESHOLD_DB", "-50").strip()
    if not thr.endswith("dB"):
        thr = f"{thr}dB"
    pad_raw = os.environ.get("SARVAM_MP3_END_PAD_SEC", "0.12").strip()
    try:
        pad_sec = max(0.0, float(pad_raw or "0"))
    except ValueError:
        pad_sec = 0.12

    # Reverse → trim leading silence (tail of original) → reverse → optional short silence pad
    af = (
        f"areverse,silenceremove=start_periods=1:start_duration={min_dur}:"
        f"start_threshold={thr}:detection=peak,areverse"
    )
    if pad_sec > 0:
        af = f"{af},apad=pad_dur={pad_sec}"
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


def prune_sarvam_cache(
    cache_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Remove *.mp3 files whose SHA256 does not match any phrase from collect_all_tts_phrases."""
    from fsm_tts_phrases import collect_all_tts_phrases

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise RuntimeError(f"Not a directory: {cache_dir}")

    want_hashes = {phrase_hash(t) for t in collect_all_tts_phrases()}
    removed: list[str] = []
    kept = 0
    for path in sorted(cache_dir.glob("*.mp3")):
        stem = path.stem.lower()
        if stem in want_hashes:
            kept += 1
            continue
        removed.append(path.name)
        if not dry_run:
            path.unlink(missing_ok=True)

    return {
        "dry_run": dry_run,
        "wanted_phrase_count": len(want_hashes),
        "kept_mp3": kept,
        "removed_count": len(removed),
        "removed_sample": removed[:40],
    }


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
    speaker: str | None = None,
    skip_existing: bool = True,
    prune_after: bool = False,
    no_trim: bool = False,
) -> dict:
    from fsm_tts_phrases import collect_all_tts_phrases

    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set SARVAM_API_KEY in the environment or ETE_v2/.env")

    sid = resolve_sarvam_speaker_id(speaker) if speaker else default_sarvam_speaker_id()
    out = Path(output_dir or (_ETE_ROOT / sarvam_cache_dir_basename(sid)))
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
            raw = synthesize_sarvam_mp3_bytes(text, key, speaker=sid)
            if not no_trim:
                raw = postprocess_sarvam_mp3_bytes(raw)
            out.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            created += 1
        except Exception as e:
            errors.append(f"{phrase_hash(text)[:12]}…: {e!r}")

    manifest = {
        "provider": "sarvam",
        "speaker": sid,
        "model": MODEL,
        "pace": SARVAM_TTS_PACE,
        "speech_sample_rate_hz": SARVAM_SPEECH_SAMPLE_RATE,
        "postprocess": "none" if no_trim else "ffmpeg_trim_pad",
        "phrase_count": len(phrases),
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if prune_after:
        prune_report = prune_sarvam_cache(out, dry_run=False)
        manifest["prune"] = prune_report

    return manifest


def resolve_phrase_for_hash(want_hash: str, phrases: list[str]) -> tuple[str, str]:
    """Return (phrase_text, full_hash_hex) for a full hash or a unique prefix."""
    h = want_hash.strip().lower().removesuffix(".mp3")
    if len(h) < 8:
        raise ValueError("hash must be at least 8 hex characters")

    by_full = [t for t in phrases if phrase_hash(t).lower() == h]
    if len(by_full) == 1:
        t = by_full[0]
        return t, phrase_hash(t)

    prefixed: list[tuple[str, str]] = []
    for t in phrases:
        ph = phrase_hash(t).lower()
        if ph.startswith(h):
            prefixed.append((t, ph))

    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise RuntimeError(
            f"Ambiguous hash prefix {h!r}: {len(prefixed)} phrases match; use full 64-char hash"
        )
    raise RuntimeError(
        f"No phrase in fsm_tts_phrases matches hash {h!r} "
        "(typo, or phrase removed from phrase list?)"
    )


def generate_one_fsm_clip(
    output_dir: Path,
    want_hash: str,
    *,
    speaker: str | None = None,
    no_trim: bool = False,
) -> dict:
    """Synthesize and write exactly one MP3 for the phrase identified by hash (full or unique prefix)."""
    from fsm_tts_phrases import collect_all_tts_phrases

    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set SARVAM_API_KEY in the environment or ETE_v2/.env")

    sid = resolve_sarvam_speaker_id(speaker) if speaker else default_sarvam_speaker_id()
    phrases = collect_all_tts_phrases()
    text, full_hash = resolve_phrase_for_hash(want_hash, phrases)
    out = Path(output_dir)
    dest = out / f"{full_hash}.mp3"
    out.mkdir(parents=True, exist_ok=True)

    raw = synthesize_sarvam_mp3_bytes(text, key, speaker=sid)
    if not no_trim:
        raw = postprocess_sarvam_mp3_bytes(raw)
    dest.write_bytes(raw)

    return {
        "hash": full_hash,
        "phrase_preview": text[:120] + ("…" if len(text) > 120 else ""),
        "path": str(dest),
        "bytes": len(raw),
        "postprocess": "none" if no_trim else "ffmpeg_trim_pad",
    }


def _infer_speaker_from_cache_dir(out: Path) -> str | None:
    name = Path(out).name
    if name == "sarvam_tts_cache":
        return "shruti"
    if name == "sarvam_tts_cache_ishita":
        return "ishita"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sarvam AI FSM TTS cache builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Write MP3s for all FSM/UI phrases")
    p_gen.add_argument(
        "--speaker",
        type=str,
        default="",
        help="Sarvam speaker id: ishita | shruti (default: env SARVAM_TTS_SPEAKER or fsm default)",
    )
    p_gen.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for *.mp3 (default: ETE_v2/sarvam_tts_cache_* for chosen speaker)",
    )
    p_gen.add_argument(
        "--regenerate",
        action="store_true",
        help="Overwrite existing MP3 files",
    )
    p_gen.add_argument(
        "--prune-after",
        action="store_true",
        help="After generate, delete *.mp3 not in current fsm_tts_phrases set",
    )
    p_gen.add_argument(
        "--no-trim",
        action="store_true",
        help="Write raw Sarvam MP3s (skip ffmpeg tail trim and end pad for every phrase)",
    )

    p_one = sub.add_parser(
        "generate-one",
        help="Regenerate one cached clip by SHA-256 of phrase (from fsm_tts_phrases)",
    )
    p_one.add_argument(
        "hash",
        type=str,
        help="64-char hex hash, or unique prefix (e.g. first 12 chars)",
    )
    p_one.add_argument(
        "--speaker",
        type=str,
        default="",
        help="Sarvam speaker id: ishita | shruti (default: env or fsm default)",
    )
    p_one.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Cache directory (default: ETE_v2/sarvam_tts_cache_* for chosen speaker)",
    )
    p_one.add_argument(
        "--no-trim",
        action="store_true",
        help="Write raw Sarvam MP3 (skip ffmpeg tail trim and end pad)",
    )

    p_prune = sub.add_parser(
        "prune",
        help="Delete cached *.mp3 whose phrase is no longer in fsm_tts_phrases",
    )
    p_prune.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Cache directory (default: ETE_v2/sarvam_tts_cache_* for default speaker)",
    )
    p_prune.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting",
    )

    p_trim = sub.add_parser(
        "retrim",
        help="Trim trailing silence/noise on cached MP3s via ffmpeg (no API calls)",
    )
    p_trim.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Cache directory (default: ETE_v2/sarvam_tts_cache_* for default speaker)",
    )
    p_trim.add_argument(
        "--hash",
        type=str,
        default="",
        help="Only files whose name starts with this hex prefix (e.g. 67aff576)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "generate":
        spk = (getattr(args, "speaker", None) or "").strip() or None
        out_dir = args.output
        if out_dir is None:
            sid = resolve_sarvam_speaker_id(spk) if spk else default_sarvam_speaker_id()
            out_dir = _ETE_ROOT / sarvam_cache_dir_basename(sid)
        elif not spk:
            spk = _infer_speaker_from_cache_dir(out_dir)
        report = generate_fsm_audio_cache(
            out_dir,
            speaker=spk,
            skip_existing=not args.regenerate,
            prune_after=getattr(args, "prune_after", False),
            no_trim=getattr(args, "no_trim", False),
        )
        print(json.dumps(report, indent=2))
        return 1 if report.get("errors") else 0
    if args.cmd == "generate-one":
        try:
            spk = (getattr(args, "speaker", None) or "").strip() or None
            out_dir = args.output
            if out_dir is None:
                sid = resolve_sarvam_speaker_id(spk) if spk else default_sarvam_speaker_id()
                out_dir = _ETE_ROOT / sarvam_cache_dir_basename(sid)
            elif not spk:
                spk = _infer_speaker_from_cache_dir(out_dir)
            report = generate_one_fsm_clip(
                out_dir,
                args.hash,
                speaker=spk,
                no_trim=getattr(args, "no_trim", False),
            )
            print(json.dumps(report, indent=2))
            return 0
        except (RuntimeError, ValueError) as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            return 1
    if args.cmd == "prune":
        out = args.output or default_sarvam_cache_dir()
        report = prune_sarvam_cache(out, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "retrim":
        out = args.output or default_sarvam_cache_dir()
        report = retrim_cache_dir(out, hash_prefix=args.hash)
        print(json.dumps(report, indent=2))
        return 1 if report.get("errors") else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
