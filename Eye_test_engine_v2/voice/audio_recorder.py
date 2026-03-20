"""Audio recorder for HITL annotation pipeline.

Records every patient utterance during voice sessions as FLAC files
with full metadata in a JSONL manifest. Stored in ~/.eye_test_audio/.

Directory structure:
    ~/.eye_test_audio/
        2026-03-20/
            session_xxx/
                manifest.jsonl      # one JSON line per utterance
                utt_001.flac        # audio file
                utt_002.flac
                ...
"""

import json
import os
import struct
import wave
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"

# Minimum utterance duration (seconds) to record — skip very short noise
MIN_UTTERANCE_DURATION = 0.3


class AudioRecorder:
    """Records utterances for a single voice session."""

    def __init__(self, session_id: str, session_orchestrator, lang: str = "en",
                 mic_device: str = "", sample_rate: int = 16000, stt_engine: str = "local"):
        self._session_id = session_id
        self._lang = lang
        self._mic_device = mic_device
        self._sample_rate = sample_rate
        self._stt_engine = stt_engine
        self._utt_counter = 0

        # Create session directory: ~/.eye_test_audio/YYYY-MM-DD/session_xxx/
        today = datetime.now().strftime("%Y-%m-%d")
        self._session_dir = AUDIO_BASE_DIR / today / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._session_dir / "manifest.jsonl"

        # Extract patient metadata from session orchestrator
        self._patient_meta = self._extract_patient_meta(session_orchestrator)

    def _extract_patient_meta(self, orchestrator) -> dict:
        """Extract patient intake data from the session orchestrator."""
        meta = {
            "session_id": self._session_id,
            "session_start": datetime.now().isoformat(),
            "lang": self._lang,
            "mic_device": self._mic_device,
        }

        if orchestrator and orchestrator.patient_input:
            p = orchestrator.patient_input
            meta.update({
                "patient_age": p.age,
                "patient_occupation": p.occupation or "",
                "screen_time_hours": p.screen_time_hours,
                "driving_hours": p.driving_hours,
                "primary_reason": p.primary_reason or "",
                "symptoms_text": p.symptoms_text or "",
                "satisfaction": p.satisfaction_with_current_rx or "",
                "wear_type": p.wear_type or "",
                "priority": p.priority or "",
                "diabetes": p.diabetes,
                "prior_eye_surgery": p.prior_eye_surgery or "",
                "keratoconus": p.keratoconus,
                "amblyopia": p.amblyopia,
            })

            # AR readings
            if p.autorefractor_re:
                meta["ar_re"] = {
                    "sph": p.autorefractor_re.sphere,
                    "cyl": p.autorefractor_re.cylinder,
                    "axis": p.autorefractor_re.axis,
                }
            if p.autorefractor_le:
                meta["ar_le"] = {
                    "sph": p.autorefractor_le.sphere,
                    "cyl": p.autorefractor_le.cylinder,
                    "axis": p.autorefractor_le.axis,
                }
            if p.lenso_re:
                meta["lenso_re"] = {
                    "sph": p.lenso_re.sphere,
                    "cyl": p.lenso_re.cylinder,
                    "axis": p.lenso_re.axis,
                }
            if p.lenso_le:
                meta["lenso_le"] = {
                    "sph": p.lenso_le.sphere,
                    "cyl": p.lenso_le.cylinder,
                    "axis": p.lenso_le.axis,
                }

        return meta

    def record_utterance(
        self,
        audio_int16: np.ndarray,
        transcript: str,
        response_type: str,
        matched_option: Optional[str],
        confidence: float,
        fsm_state: str,
        phase_name: str,
        ambient_rms: float = 0.0,
    ) -> dict:
        """Record a single utterance and append metadata to the manifest.

        Args:
            audio_int16: Raw int16 PCM audio at self._sample_rate.
            transcript: Whisper transcription text.
            response_type: FSM response type (e.g. "READABILITY").
            matched_option: Matched FSM option or None.
            confidence: Fuzzy match confidence (0-100).
            fsm_state: Current FSM state (e.g. "B", "E").
            phase_name: Human-readable phase name.
            ambient_rms: RMS of the audio (ambient noise indicator).

        Returns:
            Utterance metadata dict.
        """
        duration_sec = len(audio_int16) / self._sample_rate
        if duration_sec < MIN_UTTERANCE_DURATION:
            return {}

        self._utt_counter += 1
        utt_id = f"utt_{self._utt_counter:04d}"
        flac_filename = f"{utt_id}.flac"
        flac_path = self._session_dir / flac_filename

        # Save as FLAC (via WAV → FLAC conversion)
        self._save_flac(audio_int16, flac_path)

        # Compute ambient noise level
        if ambient_rms == 0.0 and len(audio_int16) > 0:
            audio_float = audio_int16.astype(np.float32) / 32768.0
            ambient_rms = float(np.sqrt(np.mean(audio_float ** 2)))

        # Build metadata
        was_understood = matched_option is not None
        meta = {
            "id": utt_id,
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "audio_file": flac_filename,
            "duration_sec": round(duration_sec, 2),
            "sample_rate": self._sample_rate,
            "transcript_whisper": transcript,
            "response_type": response_type,
            "matched_option": matched_option,
            "confidence": round(confidence, 1),
            "was_understood": was_understood,
            "needs_review": not was_understood or confidence < 80.0,
            "fsm_state": fsm_state,
            "phase_name": phase_name,
            "lang": self._lang,
            "mic_device": self._mic_device,
            "stt_engine": self._stt_engine,
            "ambient_rms": round(ambient_rms, 4),
            # Review fields (filled by HITL reviewer)
            "reviewed": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "correct_option": None,  # reviewer's correction
            "review_notes": None,
            "is_garbage": False,  # marked as noise/non-speech
        }

        # Append to manifest
        with open(self._manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        print(f"[RECORDER] {utt_id}: {duration_sec:.1f}s "
              f"'{transcript[:40]}' → {matched_option or '?'} "
              f"({'OK' if was_understood else 'REVIEW'})")

        return meta

    def _save_flac(self, audio_int16: np.ndarray, flac_path: Path):
        """Save int16 PCM audio as FLAC file."""
        # Write WAV to temp, convert to FLAC
        wav_path = flac_path.with_suffix(".wav")
        try:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(audio_int16.tobytes())

            # Convert WAV → FLAC using ffmpeg (if available) or keep as WAV
            try:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_path), "-c:a", "flac", str(flac_path)],
                    capture_output=True, timeout=10,
                )
                if result.returncode == 0:
                    wav_path.unlink()  # remove temp WAV
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Fallback: keep as WAV if ffmpeg not available
            wav_path.rename(flac_path.with_suffix(".wav"))
            # Update the filename in the path reference
        except Exception as e:
            print(f"[RECORDER] Save error: {e}")

    def get_session_dir(self) -> Path:
        return self._session_dir

    def get_patient_meta(self) -> dict:
        return self._patient_meta

    def write_session_summary(self):
        """Write session-level summary to the session directory."""
        summary = {
            **self._patient_meta,
            "session_end": datetime.now().isoformat(),
            "total_utterances": self._utt_counter,
            "audio_dir": str(self._session_dir),
        }
        summary_path = self._session_dir / "session_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def load_all_utterances(date_filter: str = None, needs_review: bool = None,
                        reviewed: bool = None, session_id: str = None) -> list:
    """Load utterance metadata from all sessions, with optional filters.

    Args:
        date_filter: "YYYY-MM-DD" to filter by date.
        needs_review: True to show only unreviewed flagged items.
        reviewed: True to show only reviewed items.
        session_id: Filter by specific session.

    Returns:
        List of utterance metadata dicts.
    """
    utterances = []

    if not AUDIO_BASE_DIR.exists():
        return utterances

    # Walk date directories
    date_dirs = sorted(AUDIO_BASE_DIR.iterdir())
    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue
        if date_filter and date_dir.name != date_filter:
            continue

        # Walk session directories
        for sess_dir in sorted(date_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            if session_id and sess_dir.name != session_id:
                continue

            manifest = sess_dir / "manifest.jsonl"
            if not manifest.exists():
                continue

            with open(manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        utt = json.loads(line)
                        utt["_date"] = date_dir.name
                        utt["_audio_path"] = str(sess_dir / utt.get("audio_file", ""))

                        # Apply filters
                        if needs_review is not None and utt.get("needs_review") != needs_review:
                            continue
                        if reviewed is not None and utt.get("reviewed") != reviewed:
                            continue

                        utterances.append(utt)
                    except json.JSONDecodeError:
                        continue

    return utterances


def update_utterance(session_id: str, utt_id: str, updates: dict) -> bool:
    """Update a specific utterance in its manifest file.

    Args:
        session_id: Session ID.
        utt_id: Utterance ID (e.g. "utt_0001").
        updates: Dict of fields to update.

    Returns:
        True if updated, False if not found.
    """
    # Find the manifest file
    for date_dir in AUDIO_BASE_DIR.iterdir():
        if not date_dir.is_dir():
            continue
        sess_dir = date_dir / session_id
        manifest = sess_dir / "manifest.jsonl"
        if not manifest.exists():
            continue

        # Read all lines, update the matching one, rewrite
        lines = []
        found = False
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    utt = json.loads(line)
                    if utt.get("id") == utt_id:
                        utt.update(updates)
                        found = True
                    lines.append(json.dumps(utt, ensure_ascii=False))
                except json.JSONDecodeError:
                    lines.append(line)

        if found:
            with open(manifest, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return True

    return False


def export_training_dataset(output_dir: str, format: str = "whisper") -> dict:
    """Export reviewed utterances as a training dataset.

    Args:
        output_dir: Directory to write the dataset.
        format: "whisper" (audio + transcript) or "intent" (audio + label).

    Returns:
        Stats dict with counts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    utterances = load_all_utterances(reviewed=True)
    # Exclude garbage
    utterances = [u for u in utterances if not u.get("is_garbage")]

    entries = []
    for utt in utterances:
        audio_path = utt.get("_audio_path", "")
        if not Path(audio_path).exists():
            continue

        correct_option = utt.get("correct_option") or utt.get("matched_option")
        transcript = utt.get("transcript_whisper", "")

        if format == "whisper":
            entries.append({
                "audio": audio_path,
                "transcript": transcript,
                "language": utt.get("lang", "en"),
            })
        elif format == "intent":
            if correct_option:
                entries.append({
                    "audio": audio_path,
                    "intent": correct_option,
                    "response_type": utt.get("response_type", ""),
                    "language": utt.get("lang", "en"),
                })

    # Write manifest
    manifest_path = out_path / f"dataset_{format}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats = {
        "total_reviewed": len(utterances),
        "exported": len(entries),
        "format": format,
        "output": str(manifest_path),
    }
    return stats
