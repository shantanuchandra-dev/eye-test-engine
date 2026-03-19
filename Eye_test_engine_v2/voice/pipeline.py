"""Voice pipeline for Eye Test Engine v2.

Direct pipeline (no pipecat PipelineTask): mic audio → VAD → STT → fuzzy match.
TTS is handled by DirectTTSProcessor which sends audio directly to the browser.

All models are loaded from voice/models/ (local directory).
Run `python -m voice.download_models` to download them first.
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from faster_whisper import WhisperModel

from voice.fuzzy_matcher import match_transcript

# ── Local model paths ────────────────────────────────────────────────────
_VOICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = _VOICE_DIR / "models"
WHISPER_MODEL_DIR = MODELS_DIR / "whisper-small"
PIPER_MODEL_DIR = MODELS_DIR / "piper"
SILERO_MODEL_DIR = MODELS_DIR / "silero"

DEFAULT_PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-pratham-medium",
}

SAMPLE_RATE = 16000
CHANNELS = 1


def _resolve_whisper_model() -> str:
    if WHISPER_MODEL_DIR.exists():
        model_bins = list(WHISPER_MODEL_DIR.rglob("model.bin"))
        if model_bins:
            return str(model_bins[0].parent)
    return "small"


def _resolve_piper_voice(voice_name: Optional[str] = None, lang: str = "en") -> str:
    if voice_name is None:
        voice_name = DEFAULT_PIPER_VOICES.get(lang, DEFAULT_PIPER_VOICES["en"])
    onnx_path = PIPER_MODEL_DIR / f"{voice_name}.onnx"
    if onnx_path.exists():
        return voice_name
    if PIPER_MODEL_DIR.exists():
        onnx_files = list(PIPER_MODEL_DIR.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0].stem
    return voice_name


def _resolve_silero_hub_dir() -> Optional[str]:
    if SILERO_MODEL_DIR.exists() and any(SILERO_MODEL_DIR.iterdir()):
        return str(SILERO_MODEL_DIR)
    return None


class DirectTTSProcessor:
    """Synthesizes speech via Piper and sends audio bytes directly to the WebSocket."""

    def __init__(self, voice_path: str, ws_send_bytes, ws_send_json):
        from piper import PiperVoice
        self._voice = PiperVoice.load(voice_path)
        self._ws_send_bytes = ws_send_bytes
        self._ws_send_json = ws_send_json
        self._sample_rate = self._voice.config.sample_rate
        self._speaking = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def speak(self, text: str):
        if not text:
            return
        self._speaking = True
        await self._ws_send_json({"type": "tts_start", "text": text})
        try:
            chunks = await asyncio.to_thread(self._synthesize, text)
            for chunk_bytes in chunks:
                if not self._speaking:
                    break
                await self._ws_send_bytes(b'\x01' + chunk_bytes)
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            self._speaking = False
            await self._ws_send_json({"type": "tts_end"})

    def _synthesize(self, text: str) -> list:
        chunks = []
        for audio_chunk in self._voice.synthesize(text):
            chunks.append(audio_chunk.audio_int16_bytes)
        return chunks

    def stop(self):
        self._speaking = False


class VoicePipeline:
    """Direct voice pipeline: VAD → STT → fuzzy match → FSM → TTS.

    Runs without pipecat PipelineTask. Audio frames are pushed directly
    into process_audio() from the WebSocket handler.
    """

    def __init__(self, session, ws_send_json, tts, silero_hub_dir=None,
                 whisper_model="small", confidence_threshold=60.0):
        self.session = session
        self.ws_send_json = ws_send_json
        self.tts = tts
        self._confidence_threshold = confidence_threshold

        # Load Silero VAD
        if silero_hub_dir:
            torch.hub.set_dir(silero_hub_dir)
        self._vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        self._vad_speaking = False
        self._vad_buffer = np.array([], dtype=np.int16)
        self._vad_chunk_size = 512  # Silero needs 512 samples at 16kHz

        # Load faster-whisper
        print(f"[VOICE] Loading Whisper model: {whisper_model}")
        self._whisper = WhisperModel(whisper_model, device="cpu", compute_type="int8")

        # Audio buffer for STT (accumulates while user is speaking)
        self._speech_buffer = np.array([], dtype=np.int16)

        self._running = True

    async def process_audio(self, audio_int16: bytes):
        """Process an audio chunk from the browser mic."""
        if not self._running:
            return

        try:
            samples = np.frombuffer(audio_int16, dtype=np.int16)
        except Exception as e:
            print(f"[VOICE] Audio parse error: {e}")
            return

        if not hasattr(self, '_audio_frame_count'):
            self._audio_frame_count = 0
        self._audio_frame_count += 1
        if self._audio_frame_count <= 3:
            print(f"[VOICE] Audio frame #{self._audio_frame_count}: {len(samples)} samples, max={np.max(np.abs(samples)) if len(samples) > 0 else 0}")

        # Feed to VAD in 512-sample chunks
        self._vad_buffer = np.concatenate([self._vad_buffer, samples])

        while len(self._vad_buffer) >= self._vad_chunk_size:
            chunk = self._vad_buffer[:self._vad_chunk_size]
            self._vad_buffer = self._vad_buffer[self._vad_chunk_size:]

            # Run Silero VAD
            try:
                audio_float = chunk.astype(np.float32) / 32768.0
                tensor = torch.from_numpy(audio_float)
                confidence = self._vad_model(tensor, SAMPLE_RATE).item()
                is_speech = confidence > 0.5
            except Exception as e:
                print(f"[VAD] Error: {e}")
                continue

            if is_speech and not self._vad_speaking:
                self._vad_speaking = True
                self._speech_buffer = np.array([], dtype=np.int16)
                await self.ws_send_json({"type": "vad", "speaking": True})

            elif not is_speech and self._vad_speaking:
                self._vad_speaking = False
                await self.ws_send_json({"type": "vad", "speaking": False})
                # Speech ended — run STT on accumulated buffer
                if len(self._speech_buffer) > SAMPLE_RATE * 0.3:  # min 0.3s
                    await self._transcribe_and_process()

            # Accumulate speech audio
            if self._vad_speaking:
                self._speech_buffer = np.concatenate([self._speech_buffer, chunk])

    async def _transcribe_and_process(self):
        """Run STT on the speech buffer and process the result."""
        audio = self._speech_buffer.copy()
        self._speech_buffer = np.array([], dtype=np.int16)

        if len(audio) < SAMPLE_RATE * 0.3:
            return

        # Convert to float32 for whisper
        audio_float = audio.astype(np.float32) / 32768.0

        # Run transcription in a thread
        segments = await asyncio.to_thread(self._run_whisper, audio_float)
        transcript = " ".join(segments).strip()

        if not transcript:
            return

        # Get current FSM state
        row = self.session.current_row
        if row is None:
            return

        response_type = row.response_type
        print(f"[VOICE] Transcript: '{transcript}' | response_type: {response_type}")

        await self.ws_send_json({"type": "transcript", "text": transcript})

        # Fuzzy match
        matched_option, confidence = match_transcript(
            transcript, response_type, self._confidence_threshold
        )

        if matched_option:
            print(f"[VOICE] Matched: {matched_option} (confidence: {confidence:.1f})")
            await self.ws_send_json({
                "type": "match",
                "option": matched_option,
                "confidence": confidence,
                "transcript": transcript,
            })

            next_state = self.session.process_response(matched_option)
            await self.ws_send_json({
                "type": "state_update",
                "data": {"status": "active", **next_state},
            })

            if next_state.get("is_terminal"):
                await self.tts.speak("The test is now complete. Thank you.")
                return

            if next_state.get("auto_flip") and next_state.get("jcc_flip") == "flip1":
                flip_wait = next_state.get("flip_wait_seconds", 2)
                asyncio.create_task(self._handle_jcc_auto_flip(flip_wait))
                return

            question = next_state.get("question", "")
            if question:
                await self.tts.speak(question)
        else:
            print(f"[VOICE] No match for: '{transcript}'")
            await self.ws_send_json({"type": "no_match", "transcript": transcript})
            await self.tts.speak("I didn't catch that clearly. Could you please repeat?")

    def _run_whisper(self, audio_float: np.ndarray) -> list:
        """Run faster-whisper transcription (called in thread)."""
        segments, _ = self._whisper.transcribe(
            audio_float,
            language="en",
            beam_size=3,
            vad_filter=False,  # We already did VAD
        )
        return [seg.text for seg in segments]

    async def _handle_jcc_auto_flip(self, wait_seconds: float):
        await asyncio.sleep(wait_seconds)
        next_state = self.session.process_response("AUTO_FLIP")
        await self.ws_send_json({
            "type": "state_update",
            "data": {"status": "active", **next_state},
        })
        question = next_state.get("question", "")
        if question:
            await self.tts.speak(question)

    def stop(self):
        self._running = False


def build_pipeline(session, ws_send_json, ws_send_bytes,
                   whisper_model=None, piper_voice=None, lang="en"):
    """Build the voice pipeline.

    Returns (voice_pipeline, tts_processor).
    """
    whisper_model = whisper_model or _resolve_whisper_model()
    piper_voice = piper_voice or _resolve_piper_voice(lang=lang)
    silero_hub = _resolve_silero_hub_dir()

    piper_onnx = str(PIPER_MODEL_DIR / f"{piper_voice}.onnx")
    print(f"[VOICE] Whisper: {whisper_model}")
    print(f"[VOICE] Piper:   {piper_onnx}")
    print(f"[VOICE] Silero:  {silero_hub or 'default'}")

    tts = DirectTTSProcessor(
        voice_path=piper_onnx,
        ws_send_bytes=ws_send_bytes,
        ws_send_json=ws_send_json,
    )

    pipeline = VoicePipeline(
        session=session,
        ws_send_json=ws_send_json,
        tts=tts,
        silero_hub_dir=silero_hub,
        whisper_model=whisper_model,
    )

    return pipeline, tts
