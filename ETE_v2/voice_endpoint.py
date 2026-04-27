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


# Vasista22 hallucinates Hindi news-broadcast text after short clinical
# utterances. Stripping the tail at the start of any of these markers brings
# intent-matching agreement from 52% to 87% on the clinic dev split (see
# decision doc 019 and eval/report.strip_vasista22_hallucination).
_VASISTA22_HALLUCINATION_MARKERS = (
    "प्रादेशिक समाचारों",
    "समाचारों के इस बुलेटिन",
    "बुलेटिन में आपका",
    "उन्होंने बताया कि",
    "उन्होंने कहा कि",
    "इस अवसर पर उन्होंने",
    "आपका फिर से स्वागत",
    "अवगत कराया",
    "प्रदेश में पिछले",
    "प्रदेश में अब तक",
)


def _strip_vasista22_hallucination_tail(text: str) -> str:
    if not text:
        return text
    for marker in _VASISTA22_HALLUCINATION_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            return text[:idx].strip()
    return text
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
    model_path = os.environ.get("WHISPER_MODEL_PATH", "models/vasista22-whisper-hindi-small-ct2")
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
    audio_format: str = "pcm16",
) -> dict:
    """Transcribe base64-encoded audio using faster-whisper.

    Args:
        audio_base64: Base64-encoded audio data
        language_hint: Optional language hint ('en', 'hi', etc.)
        audio_format: 'pcm16' for raw PCM16 mono 16kHz, 'webm' for WebM/Opus blob

    Returns:
        dict with: text, detected_language, language_probability, stt_seconds, backend
    """
    transcriber = _get_transcriber()
    if transcriber is None:
        return {"error": "faster-whisper not available", "text": "", "backend": "none"}

    # Decode base64
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Invalid base64 audio: {e}", "text": "", "backend": "none"}

    if len(audio_bytes) < 100:
        return {"error": "Audio data too small", "text": "", "backend": transcriber.backend_name}

    tmp_path = None
    try:
        if audio_format == "webm":
            # Save WebM blob directly — faster-whisper/ffmpeg can decode it
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(audio_bytes)
        else:
            # Raw PCM16 mono 16kHz — wrap in WAV
            if len(audio_bytes) < SAMPLE_RATE * SAMPLE_WIDTH * 0.3:
                return {"error": "Audio too short (< 0.3s)", "text": "", "backend": transcriber.backend_name}
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(SAMPLE_WIDTH)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_bytes)

        stt_start = time.perf_counter()
        result = transcriber.transcribe_result(
            tmp_path,
            language_override=language_hint,
        )
        stt_elapsed = time.perf_counter() - stt_start

        # Vasista22 appends Hindi news-broadcast hallucinations on short
        # clinical clips ("...प्रादेशिक समाचारों के इस बुलेटिन में..."). Strip
        # the tail before returning so the matcher sees the patient's actual
        # response. Empirically: lifts intent-classification agreement from
        # 52% to 87% on dev split. See eval/report.strip_vasista22_hallucination.
        cleaned_text = _strip_vasista22_hallucination_tail(result.text)

        return {
            "text": cleaned_text,
            "raw_text": result.text,
            "detected_language": result.detected_language,
            "language_probability": result.language_probability,
            "avg_logprob": result.avg_logprob,
            "no_speech_prob": result.no_speech_prob,
            "word_confidences": result.word_confidences,
            "stt_seconds": round(stt_elapsed, 3),
            "backend": transcriber.backend_name,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"error": str(e), "text": "", "backend": transcriber.backend_name}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def is_whisper_available() -> bool:
    """Check if faster-whisper can be loaded."""
    try:
        import faster_whisper  # noqa: F401
        model_path = os.environ.get("WHISPER_MODEL_PATH", "models/vasista22-whisper-hindi-small-ct2")
        return Path(model_path).exists()
    except ImportError:
        return False
