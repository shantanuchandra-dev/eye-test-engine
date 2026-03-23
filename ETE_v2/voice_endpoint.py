"""
Backend voice transcription endpoint using faster-whisper.
Matches FSMv3.1_R2's local STT pipeline exactly.

Frontend sends recorded audio as base64 WAV via POST.
Backend transcribes with faster-whisper and returns transcript + match result.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded transcriber singleton
_transcriber = None
_transcriber_loading = False

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


def _get_transcriber():
    """Lazy-load the faster-whisper transcriber (heavy ~1.5GB model)."""
    global _transcriber, _transcriber_loading

    if _transcriber is not None:
        return _transcriber

    if _transcriber_loading:
        return None

    _transcriber_loading = True
    model_path = os.environ.get("WHISPER_MODEL_PATH", "models/whisper-large-v3-turbo-ct2")
    cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", "4"))
    language = os.environ.get("WHISPER_LANGUAGE", "auto")

    try:
        from fsm.audio.local_stt import create_local_transcriber
        logger.info(f"Loading faster-whisper model from {model_path}...")
        _transcriber = create_local_transcriber(
            backend="voice-local-fw",
            ct2_model_path=model_path,
            cpu_threads=cpu_threads,
            language=language,
        )
        logger.info("faster-whisper model loaded successfully")
        return _transcriber
    except Exception as e:
        logger.warning(f"Could not load faster-whisper: {e}")
        _transcriber_loading = False
        return None


def transcribe_audio(
    audio_base64: str,
    language_hint: Optional[str] = None,
) -> dict:
    """Transcribe base64-encoded PCM16 audio using faster-whisper.

    Args:
        audio_base64: Base64-encoded raw PCM16 mono 16kHz audio
        language_hint: Optional language hint ('en', 'hi', etc.)

    Returns:
        dict with: text, detected_language, language_probability, stt_seconds, backend
    """
    transcriber = _get_transcriber()
    if transcriber is None:
        return {"error": "faster-whisper not available", "text": "", "backend": "none"}

    # Decode base64 to raw PCM bytes
    try:
        pcm_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Invalid base64 audio: {e}", "text": "", "backend": "none"}

    if len(pcm_bytes) < SAMPLE_RATE * SAMPLE_WIDTH * 0.3:
        return {"error": "Audio too short (< 0.3s)", "text": "", "backend": transcriber.backend_name}

    # Write to temp WAV file (faster-whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)

    try:
        stt_start = time.perf_counter()
        result = transcriber.transcribe_result(
            tmp_path,
            language_override=language_hint,
        )
        stt_elapsed = time.perf_counter() - stt_start

        return {
            "text": result.text,
            "detected_language": result.detected_language,
            "language_probability": result.language_probability,
            "stt_seconds": round(stt_elapsed, 3),
            "backend": transcriber.backend_name,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"error": str(e), "text": "", "backend": transcriber.backend_name}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def is_whisper_available() -> bool:
    """Check if faster-whisper can be loaded."""
    try:
        import faster_whisper  # noqa: F401
        model_path = os.environ.get("WHISPER_MODEL_PATH", "models/whisper-large-v3-turbo-ct2")
        return Path(model_path).exists()
    except ImportError:
        return False
