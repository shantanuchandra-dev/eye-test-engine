"""Voice pipeline for Eye Test Engine v2.

Direct pipeline (no pipecat PipelineTask): mic audio → VAD → STT → fuzzy match.
TTS is handled by DirectTTSProcessor which sends audio directly to the browser.

All models are loaded from voice/models/ (local directory).
Run `python -m voice.download_models` to download them first.
"""

import asyncio
import os
import numpy as np
from pathlib import Path
from typing import Optional

# Suppress HuggingFace Hub warnings (faster-whisper pulls in huggingface_hub).
# We only use local models so HF network access is never needed at runtime.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import onnxruntime
from faster_whisper import WhisperModel

from voice.chart_reading import ChartReadingDetector, looks_like_chart_letter_utterance, resolve_chart_letters
from voice.chart_reading import listen_seconds as coarse_sphere_listen_seconds
from voice.fuzzy_matcher import match_transcript
from voice.regional_languages import (
    SUPPORTED_LANGUAGES, WHISPER_LANG_CODES,
    get_translation as _regional_translate,
    get_followup as _regional_followup,
    get_rephrased as _regional_rephrased,
    get_message as _regional_message,
    get_keywords as _regional_keywords,
)

SAMPLE_RATE = 16000


# ── Silero VAD via ONNX Runtime (no torch) ────────────────────────────────

def _load_silero_vad_onnx():
    """Load Silero VAD ONNX model. Returns (session, initial_state, sr_tensor)."""
    onnx_path = SILERO_ONNX_PATH

    if not onnx_path.exists():
        # Auto-download from silero-vad GitHub
        import urllib.request
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        print(f"[VAD] Downloading Silero VAD ONNX → {onnx_path}")
        urllib.request.urlretrieve(url, str(onnx_path))

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(onnx_path), sess_options=opts)

    # Initial hidden state: 2 × 1 × 64 zeros (LSTM h and c)
    state = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array([SAMPLE_RATE], dtype=np.int64)
    print(f"[VAD] Silero VAD ONNX loaded: {onnx_path}")
    return session, state, sr


def _run_silero_vad_onnx(session, state, audio_float: np.ndarray, sr):
    """Run a single VAD inference. Returns (confidence, new_state)."""
    # Silero expects (1, chunk_size) float32
    inp = audio_float.reshape(1, -1).astype(np.float32)
    ort_inputs = {
        "input": inp,
        "state": state,
        "sr": sr,
    }
    out, new_state = session.run(None, ort_inputs)
    return float(out[0][0]), new_state


# Exit keywords — if the user says any of these, confirm before stopping.
EXIT_KEYWORDS = [
    "stop", "stop the test", "ruko", "band karo", "bas",
    "escalate", "need human", "doctor bulao", "optometrist",
    "help", "need help", "madad", "sahayata",
    "cancel", "abort", "quit", "end test",
    # Devanagari Hindi
    "रुको", "बंद करो", "बस", "रोको",
    "मदद", "सहायता", "डॉक्टर बुलाओ",
    "बंद", "रुकिए",
]

def _is_exit_request(transcript: str) -> bool:
    """Check if the transcript is an exit/escalation request."""
    text = transcript.lower().strip()
    for kw in EXIT_KEYWORDS:
        if kw in text:
            return True
    return False

# Rephrased questions for when patient doesn't respond within 3 seconds.
# Simpler language, explicit options spelled out.
import random

REPHRASED_QUESTIONS = {
    "READABILITY": "Can you read the letters?",
    "NEAR_READABILITY": "Can you read the text?",
    "COMPARE_1_2": "First one or second one?",
    "COLOR_CHOICE": "Which side is clearer?",
    "TOP_BOTTOM": "Which row is clearer?",
    "NEAR_BINOC": "Is it clear?",
}

# Varied follow-ups for repeated questions in the same phase.
# A random one is picked each time to avoid sounding robotic.
SHORT_FOLLOWUP_POOL = {
    "READABILITY": [
        "How about now? Clear, blurry, or can't read?",
        "And these letters? Clear, blurry, or not readable?",
        "Can you read this line? Clear, blurry, or no?",
        "What about now? Readable, blurry, or not at all?",
        "This row — clear, blurry, or can't make it out?",
        "Better or worse? Clear, blurry, or unreadable?",
        "How does this look? Clear, a bit blurry, or can't read?",
        "Next line. Clear, blurry, or not readable?",
    ],
    # Phase B: right eye coarse sphere — short, no options dictated
    "state_B": [
        "How about now?",
        "And this line?",
        "What about these letters?",
        "Any change?",
        "How does this look?",
        "Next line.",
        "And now?",
        "Can you read this?",
    ],
    # Phase D: left eye coarse sphere — short, no options dictated
    "state_D": [
        "How about now?",
        "And this line?",
        "What about these letters?",
        "Any change?",
        "How does this look?",
        "Next line.",
        "And now?",
        "Can you read this?",
    ],
    "NEAR_READABILITY": [
        "And now? Clear, blurry, or can't read?",
        "How about the small text now? Clear, blurry, or no?",
        "Can you read this? Clear, blurry, or not readable?",
        "This text — clear, blurry, or can't make it out?",
    ],
    "COMPARE_1_2": [
        "And this time?",
        "Any difference?",
        "How about now?",
    ],
    "state_E": [
        "And this time?",
        "Which looked sharper?",
        "Any difference?",
        "How about now?",
    ],
    "state_F": [
        "And now?",
        "Which was clearer?",
        "Any difference?",
        "How about this time?",
    ],
    "state_H": [
        "And this time?",
        "Which looked sharper?",
        "Any difference?",
        "How about now?",
    ],
    "state_I": [
        "And now?",
        "Which was clearer?",
        "Any difference?",
        "How about this time?",
    ],
    "COLOR_CHOICE": [
        "Which side is clearer?",
        "Red or green?",
        "Any difference?",
    ],
    "TOP_BOTTOM": [
        "Top, bottom, or same?",
        "Upper row or lower row? Or equal?",
        "Which row is clearer — top or bottom?",
    ],
    "NEAR_BINOC": [
        "Clear, or not clear?",
        "Comfortable to read? Yes or no?",
    ],
}


def _pick_followup(response_type: str, state: str = "") -> str:
    """Pick a random short follow-up for a repeated same-phase question."""
    pool = SHORT_FOLLOWUP_POOL.get(f"state_{state}") or SHORT_FOLLOWUP_POOL.get(response_type)
    if pool:
        return random.choice(pool)
    return ""


# ── Hinglish translations (used when hi_IN voice is selected) ─────────
# Maps FSM question → Hinglish equivalent. Spoken by hi_IN Piper voice.
HINDI_QUESTIONS = {
    "Looking at the letters, are they clear, slightly blurry, or not readable?":
        "अक्षरों को देखिए, क्या ये साफ़ दिख रहे हैं, थोड़े धुंधले हैं, या पढ़ नहीं पा रहे?",
    "Looking at the letters now, are they clear, a bit blurry, or not readable?":
        "अब अक्षरों को देखिए, साफ़ हैं, धुंधले हैं, या पढ़ नहीं पा रहे?",
    "Look carefully at the dot pattern. Between view one and two, which one makes the dots look sharper or better aligned? Is it one, two, about the same, or hard to tell?":
        "बिंदुओं को ध्यान से देखिए। पहला दृश्य बेहतर है या दूसरा? या दोनों एक जैसे हैं? या समझ नहीं आ रहा?",
    "Again looking at the dot pattern, which view makes the dots look clearer or sharper — one, two, about the same, or hard to tell?":
        "फिर से बिंदुओं को देखिए। कौन सा दृश्य ज़्यादा साफ़ है — पहला, दूसरा, एक जैसा, या पता नहीं?",
    "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
        "लाल और हरे रंग पर अक्षर देखिए। कौन सी तरफ़ ज़्यादा साफ़ है — लाल, हरा, या दोनों एक जैसे?",
    "Looking at the dot pattern, which one looks sharper — one, two, about the same, or hard to tell?":
        "बिंदुओं को देखिए, कौन सा तेज़ है — पहला, दूसरा, एक जैसा, या पता नहीं?",
    "Comparing the two views of the dots, which looks clearer — one, two, about the same, or hard to tell?":
        "दोनों दृश्यों की तुलना करें। कौन सा साफ़ है — पहला, दूसरा, एक जैसा, या पता नहीं?",
    "Looking again at the red and green backgrounds, which side makes the letters clearer — red, green, or about the same?":
        "फिर से लाल और हरा देखिए। कौन सी तरफ़ बेहतर है — लाल, हरा, या एक जैसा?",
    "Look at the two rows of letters. Which row looks clearer — the top row, the bottom row, or do they look about the same?":
        "दो पंक्तियाँ देखिए। कौन सी पंक्ति साफ़ है — ऊपर वाली, नीचे वाली, या दोनों एक जैसी?",
    "Looking at the near text, is it clear to read, a bit blurry, or not readable?":
        "पास का लिखावट देखिए। साफ़ है, धुंधला है, या पढ़ नहीं पा रहे?",
    "Looking at the near text again, is it clear, blurry, or not readable?":
        "फिर से पास का लिखावट देखिए। साफ़, धुंधला, या पढ़ नहीं पा रहे?",
    "Looking at the near text with both eyes, is it clear and comfortable, or still not clear?":
        "दोनों आँखों से पास का लिखावट देखिए। साफ़ और आरामदायक है, या अभी भी साफ़ नहीं?",
}

HINDI_REPHRASED = {
    "READABILITY": "साफ़ है, धुंधला है, या पढ़ नहीं पा रहे?",
    "NEAR_READABILITY": "साफ़ है, धुंधला है, या नहीं पढ़ पा रहे?",
    "COMPARE_1_2": "पहला, दूसरा, या एक जैसा?",
    "COLOR_CHOICE": "लाल, हरा, या एक जैसा?",
    "TOP_BOTTOM": "ऊपर, नीचे, या एक जैसा?",
    "NEAR_BINOC": "साफ़ है, या नहीं?",
}

HINDI_FOLLOWUP_POOL = {
    "READABILITY": [
        "अब कैसा दिख रहा है?",
        "और ये अक्षर?",
        "कोई बदलाव?",
        "ये कैसा लग रहा है?",
        "अगली पंक्ति।",
        "और अब?",
        "बेहतर है या ख़राब?",
    ],
    "state_B": [
        "अब कैसा दिख रहा है?",
        "और ये पंक्ति?",
        "ये अक्षर कैसे हैं?",
        "कोई बदलाव?",
        "कैसा लग रहा है?",
        "अगली पंक्ति।",
        "और अब?",
        "ये पढ़ पा रहे हैं?",
    ],
    "state_D": [
        "अब कैसा दिख रहा है?",
        "और ये पंक्ति?",
        "ये अक्षर कैसे हैं?",
        "कोई बदलाव?",
        "कैसा लग रहा है?",
        "अगली पंक्ति।",
        "और अब?",
        "ये पढ़ पा रहे हैं?",
    ],
    "COMPARE_1_2": [
        "और इस बार?",
        "कोई फ़र्क़?",
        "अब कैसा?",
    ],
    "state_E": [
        "और इस बार?",
        "कोई फ़र्क़?",
        "अब कैसा?",
        "कौन सा तेज़ था?",
    ],
    "state_F": [
        "और अब?",
        "कोई फ़र्क़?",
        "इस बार कैसा?",
        "कौन सा साफ़ था?",
    ],
    "state_H": [
        "और इस बार?",
        "कोई फ़र्क़?",
        "अब कैसा?",
        "कौन सा तेज़ था?",
    ],
    "state_I": [
        "और अब?",
        "कोई फ़र्क़?",
        "इस बार कैसा?",
        "कौन सा साफ़ था?",
    ],
    "COLOR_CHOICE": [
        "कौन सी तरफ़ साफ़ है?",
        "लाल या हरा?",
        "कोई फ़र्क़?",
    ],
    "TOP_BOTTOM": [
        "ऊपर या नीचे?",
        "कौन सी पंक्ति साफ़ है?",
    ],
    "NEAR_READABILITY": [
        "अब कैसा है?",
        "ये पढ़ पा रहे हो?",
    ],
    "NEAR_BINOC": [
        "साफ़ है या नहीं?",
        "आराम से पढ़ पा रहे हैं?",
    ],
}


# Voice-friendly versions of FSM questions — just the question, no options listed.
VOICE_QUESTIONS = {
    "Looking at the letters, are they clear, slightly blurry, or not readable?":
        "Looking at the letters, can you read them clearly?",
    "Looking at the letters now, are they clear, a bit blurry, or not readable?":
        "Now looking at the letters, can you read them clearly?",
    "Look carefully at the dot pattern. Between view one and two, which one makes the dots look sharper or better aligned? Is it one, two, about the same, or hard to tell?":
        "Look carefully at the dot pattern. Which view makes the dots look sharper?",
    "Again looking at the dot pattern, which view makes the dots look clearer or sharper — one, two, about the same, or hard to tell?":
        "Looking at the dots again, which view is clearer?",
    "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
        "Looking at the red and green backgrounds, which side looks clearer?",
    "Looking at the dot pattern, which one looks sharper — one, two, about the same, or hard to tell?":
        "Looking at the dots, which one looks sharper?",
    "Comparing the two views of the dots, which looks clearer — one, two, about the same, or hard to tell?":
        "Comparing the two views, which looks clearer?",
    "Looking again at the red and green backgrounds, which side makes the letters clearer — red, green, or about the same?":
        "Looking at the red and green again, which side is clearer?",
    "Look at the two rows of letters. Which row looks clearer — the top row, the bottom row, or do they look about the same?":
        "Look at the two rows. Which row looks clearer?",
    "Looking at the near text, is it clear to read, a bit blurry, or not readable?":
        "Looking at the near text, can you read it clearly?",
    "Looking at the near text again, is it clear, blurry, or not readable?":
        "Looking at the near text again, can you read it?",
    "Looking at the near text with both eyes, is it clear and comfortable, or still not clear?":
        "With both eyes, is the near text clear and comfortable?",
}


def _strip_intents(text: str) -> str:
    """Return the voice-friendly version of a question (no options dictated).
    Also strips JCC phase name prefixes like 'JCC Axis (Right Eye) — '.
    """
    # Check exact match first
    if text in VOICE_QUESTIONS:
        return VOICE_QUESTIONS[text]
    # Strip JCC phase prefix: "JCC Axis (Right Eye) — This is Flip 2. Which was better?"
    if text.startswith("JCC ") and " — " in text:
        text = text.split(" — ", 1)[1]
    return text


def _translate_to_hindi(text: str) -> str:
    """Translate a question to Hindi. Returns original if no translation found."""
    return HINDI_QUESTIONS.get(text, text)


def _pick_hindi_followup(response_type: str, state: str = "") -> str:
    """Pick a random Hindi follow-up."""
    pool = HINDI_FOLLOWUP_POOL.get(f"state_{state}") or HINDI_FOLLOWUP_POOL.get(response_type)
    if pool:
        return random.choice(pool)
    return ""

# ── Local model paths ────────────────────────────────────────────────────
_VOICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = _VOICE_DIR / "models"
WHISPER_MODEL_DIR = MODELS_DIR / "whisper-v3-turbo"
WHISPER_MODEL_DIR_FALLBACK = MODELS_DIR / "whisper-small"
PIPER_MODEL_DIR = MODELS_DIR / "piper"
SILERO_ONNX_PATH = MODELS_DIR / "silero_vad.onnx"

DEFAULT_PIPER_VOICES = {
    "en": "en_US-kusal-medium",
    "hi": "hi_IN-pratham-medium",
}

CHANNELS = 1


def _resolve_whisper_model() -> str:
    """Resolve local Whisper model path. Prefers v3-turbo, falls back to small.

    Never returns a HuggingFace model ID — only local paths.
    Run `python -m voice.download_models` first to download.
    """
    for model_dir in [WHISPER_MODEL_DIR, WHISPER_MODEL_DIR_FALLBACK]:
        if model_dir.exists():
            model_bins = list(model_dir.rglob("model.bin"))
            if model_bins:
                print(f"[VOICE] Whisper model: {model_dir.name}")
                return str(model_bins[0].parent)
    raise FileNotFoundError(
        f"No Whisper model found in {WHISPER_MODEL_DIR} or {WHISPER_MODEL_DIR_FALLBACK}. "
        f"Run: python -m voice.download_models"
    )


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


def _resolve_silero_onnx_path() -> Optional[str]:
    """Return path to silero_vad.onnx if it exists in models/."""
    if SILERO_ONNX_PATH.exists():
        return str(SILERO_ONNX_PATH)
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

    def __init__(self, session, ws_send_json, tts,
                 whisper_model=None, confidence_threshold=60.0, lang="en",
                 session_id="", mic_device="", stt_engine="local"):
        """
        Args:
            whisper_model: Pre-loaded WhisperModel instance, or a path string.
            lang: Language code ('en' or 'hi'). 'hi' enables Hindi translation.
            session_id: Session ID for audio recording.
            mic_device: Mic device label from browser.
            stt_engine: "local" (faster-whisper) or "deepgram" (cloud).
        """
        self.session = session
        self.ws_send_json = ws_send_json
        self.tts = tts
        self._confidence_threshold = confidence_threshold
        self._lang = lang
        self._stt_engine = stt_engine

        # Intent classifier fallback (loaded lazily if model exists)
        self._intent_classifier = None
        try:
            from voice.training.intent_classifier import IntentClassifierInference
            self._intent_classifier = IntentClassifierInference()
            print("[VOICE] Intent classifier loaded as fallback")
        except (FileNotFoundError, ImportError):
            pass  # No classifier model yet, that's fine

        # Audio recorder
        from voice.audio_recorder import AudioRecorder
        self._recorder = AudioRecorder(
            session_id=session_id,
            session_orchestrator=session,
            lang=lang,
            mic_device=mic_device,
            stt_engine=stt_engine,
        )

        # Load Silero VAD (ONNX — no torch dependency)
        self._vad_session, self._vad_state, self._vad_sr = _load_silero_vad_onnx()
        self._vad_speaking = False
        self._vad_buffer = np.array([], dtype=np.int16)
        self._vad_chunk_size = 512  # Silero needs 512 samples at 16kHz
        self._speech_streak = 0     # consecutive speech chunks (debounce)
        self._silence_streak = 0    # consecutive silence chunks (debounce)
        self._speech_trigger = 3    # need 3 consecutive speech chunks (~96ms) to trigger
        self._silence_trigger = 8   # need 8 consecutive silence chunks (~256ms) to stop

        # STT engine
        self._deepgram_client = None
        if stt_engine == "deepgram":
            api_key = os.environ.get("DEEPGRAM_API_KEY", "")
            if api_key:
                from voice.deepgram_stt import DeepgramSTTClient
                self._deepgram_client = DeepgramSTTClient(
                    api_key=api_key,
                    lang=lang,
                    on_transcript=self._on_deepgram_transcript,
                    on_vad=self._on_deepgram_vad,
                )
                print(f"[VOICE] STT engine: Deepgram (cloud)")
            else:
                print(f"[VOICE] WARNING: DEEPGRAM_API_KEY not set, falling back to local Whisper")
                self._stt_engine = "local"

        if self._stt_engine == "local":
            if isinstance(whisper_model, WhisperModel):
                self._whisper = whisper_model
            else:
                model_path = whisper_model or _resolve_whisper_model()
                print(f"[VOICE] Loading Whisper model: {model_path}")
                self._whisper = WhisperModel(model_path, device="cpu", compute_type="int8")
        else:
            self._whisper = None  # Not needed when using Deepgram

        # Audio buffer for STT (accumulates while user is speaking — local mode only)
        self._speech_buffer = np.array([], dtype=np.int16)

        # Silence timer: if no speech after a question, rephrase (3s default; longer on B/D chart read)
        self._silence_timer = None
        self._silence_timeout_sec = 3.0
        self._has_rephrased = False  # only rephrase once per question

        # Track previous state to use short follow-ups for repeated same-phase questions
        self._prev_state = None

        # Exit confirmation state
        self._awaiting_exit_confirm = False

        self._running = True

    async def start(self):
        """Start async components (Deepgram connection)."""
        if self._deepgram_client:
            try:
                await self._deepgram_client.connect()
            except Exception as e:
                print(f"[VOICE] Deepgram connect failed: {e}, falling back to local")
                self._stt_engine = "local"
                self._deepgram_client = None

    async def flush_audio_buffers(self):
        """Drop all buffered/in-progress audio.

        Called on phase transitions so stale audio from the previous
        phase doesn't bleed into STT for the new phase.
        """
        self._vad_buffer = np.array([], dtype=np.int16)
        self._speech_buffer = np.array([], dtype=np.int16)
        self._speech_streak = 0
        self._silence_streak = 0
        if self._vad_speaking:
            self._vad_speaking = False
            await self.ws_send_json({"type": "vad", "speaking": False})
        # Flush Deepgram's server-side buffer
        if self._deepgram_client and self._deepgram_client.is_connected:
            await self._deepgram_client.finalize()
        print("[VOICE] Audio buffers flushed (phase transition)")

    async def process_audio(self, audio_int16: bytes):
        """Process an audio chunk from the browser mic."""
        if not self._running:
            return

        try:
            samples = np.frombuffer(audio_int16, dtype=np.int16)
        except Exception as e:
            print(f"[VOICE] Audio parse error: {e}")
            return

        # Skip near-empty frames — don't waste VAD/STT on silence
        if len(samples) == 0 or np.max(np.abs(samples)) < 50:
            return

        if not hasattr(self, '_audio_frame_count'):
            self._audio_frame_count = 0
        self._audio_frame_count += 1
        if self._audio_frame_count <= 3:
            print(f"[VOICE] Audio frame #{self._audio_frame_count}: {len(samples)} samples, max={np.max(np.abs(samples))}")

        # Stream to Deepgram if using cloud STT
        if self._deepgram_client and self._deepgram_client.is_connected:
            await self._deepgram_client.send_audio(audio_int16)

        # Feed to VAD in 512-sample chunks
        self._vad_buffer = np.concatenate([self._vad_buffer, samples])

        while len(self._vad_buffer) >= self._vad_chunk_size:
            chunk = self._vad_buffer[:self._vad_chunk_size]
            self._vad_buffer = self._vad_buffer[self._vad_chunk_size:]

            # Run Silero VAD with amplitude gate
            try:
                audio_float = chunk.astype(np.float32) / 32768.0
                # Skip VAD if audio is near-silence (RMS below threshold)
                rms = np.sqrt(np.mean(audio_float ** 2))
                if rms < 0.01:
                    # Too quiet — treat as silence without running VAD
                    is_speech = False
                else:
                    confidence, self._vad_state = _run_silero_vad_onnx(
                        self._vad_session, self._vad_state,
                        audio_float, self._vad_sr,
                    )
                    is_speech = confidence > 0.6
            except Exception as e:
                print(f"[VAD] Error: {e}")
                continue

            if is_speech:
                self._speech_streak += 1
                self._silence_streak = 0
            else:
                self._silence_streak += 1
                self._speech_streak = 0

            # Debounced start: need several consecutive speech chunks
            if not self._vad_speaking and self._speech_streak >= self._speech_trigger:
                self._vad_speaking = True
                self._speech_buffer = np.array([], dtype=np.int16)
                self._cancel_silence_timer()  # patient is speaking, cancel rephrase
                await self.ws_send_json({"type": "vad", "speaking": True})

            # Debounced stop: need several consecutive silence chunks
            elif self._vad_speaking and self._silence_streak >= self._silence_trigger:
                self._vad_speaking = False
                await self.ws_send_json({"type": "vad", "speaking": False})
                # Only run local Whisper STT — Deepgram handles its own endpointing
                if self._stt_engine == "local" and len(self._speech_buffer) > SAMPLE_RATE * 0.3:
                    await self._transcribe_and_process()

            # Accumulate speech audio (for local STT + audio recording)
            if self._vad_speaking:
                self._speech_buffer = np.concatenate([self._speech_buffer, chunk])

    def _apply_silence_timeout_for_current_state(self):
        row = self.session.current_row
        if row and row.state in ("B", "D") and row.response_type == "READABILITY":
            letters = resolve_chart_letters(row.chart_param or "400")
            self._silence_timeout_sec = coarse_sphere_listen_seconds(letters)
        else:
            self._silence_timeout_sec = 3.0

    def start_silence_timer(self):
        """Start the silence timer. Called after TTS finishes a question."""
        self._cancel_silence_timer()
        self._apply_silence_timeout_for_current_state()
        self._silence_timer = asyncio.create_task(self._silence_timer_task())

    def _cancel_silence_timer(self):
        if self._silence_timer is not None:
            self._silence_timer.cancel()
            self._silence_timer = None

    async def _silence_timer_task(self):
        """Wait for silence timeout, then trigger rephrase."""
        try:
            await asyncio.sleep(self._silence_timeout_sec)
            await self._on_silence_timeout()
        except asyncio.CancelledError:
            pass

    async def _on_silence_timeout(self):
        """Called when patient hasn't spoken within the silence window after a question."""
        if not self._running or self._vad_speaking or self._has_rephrased:
            return

        row = self.session.current_row
        if row is None:
            return

        response_type = row.response_type
        if self._lang == "en":
            rephrased = REPHRASED_QUESTIONS.get(response_type)
        elif self._lang == "hi":
            rephrased = HINDI_REPHRASED.get(response_type)
        else:
            rephrased = _regional_rephrased(self._lang, response_type) or REPHRASED_QUESTIONS.get(response_type)
        if not rephrased:
            return

        self._has_rephrased = True
        print(f"[VOICE] Silence timeout — rephrasing ({response_type})")
        await self.tts.speak(rephrased)
        # After rephrasing, start another timer (but won't rephrase again due to _has_rephrased)

    async def _on_deepgram_transcript(self, transcript: str, is_final: bool):
        """Callback from DeepgramSTTClient when a transcript is received."""
        if not is_final or not transcript.strip():
            return
        # Use the same processing path as local Whisper
        # But skip the Whisper transcription step — we already have the transcript
        await self._process_transcript(transcript, self._speech_buffer.copy())
        self._speech_buffer = np.array([], dtype=np.int16)

    async def _on_deepgram_vad(self, is_speaking: bool):
        """Callback from Deepgram's VAD events."""
        # Deepgram VAD supplements our Silero VAD for UI feedback
        # We already handle this via Silero, so this is just a backup
        pass

    async def _process_transcript(self, transcript: str, audio: np.ndarray):
        """Process a transcript (from either Whisper or Deepgram) through fuzzy matching."""
        if not transcript:
            return

        print(f"[VOICE] Transcript ({self._stt_engine}): '{transcript}'")
        await self.ws_send_json({"type": "transcript", "text": transcript})

        # ── Handle exit confirmation response ──
        if self._awaiting_exit_confirm:
            self._awaiting_exit_confirm = False
            text_lower = transcript.lower().strip()
            is_yes = any(w in text_lower for w in ["yes", "haan", "ha", "confirm", "stop", "sure", "okay", "ok"])
            if is_yes:
                print(f"[VOICE] Exit confirmed")
                await self.ws_send_json({"type": "exit_confirmed"})
                if self._lang == "hi":
                    end_msg = "ठीक है, परीक्षा रोक रहे हैं।"
                elif self._lang != "en":
                    end_msg = _regional_message(self._lang, "exit_yes") or "Okay, stopping the test."
                else:
                    end_msg = "Okay, stopping the test. Please wait."
                await self.tts.speak(end_msg)
                self._cancel_silence_timer()
                return
            else:
                print(f"[VOICE] Exit cancelled, resuming")
                if self._lang == "hi":
                    resume_msg = "ठीक है, परीक्षा जारी है।"
                elif self._lang != "en":
                    resume_msg = _regional_message(self._lang, "exit_no") or "Okay, let's continue."
                else:
                    resume_msg = "Okay, let's continue."
                await self.tts.speak(resume_msg)
                self.start_silence_timer()
                return

        # ── Check for exit keywords ──
        if _is_exit_request(transcript):
            print(f"[VOICE] Exit keyword detected: '{transcript}'")
            self._awaiting_exit_confirm = True
            self._cancel_silence_timer()
            if self._lang == "hi":
                confirm_msg = "क्या आप परीक्षा रोकना चाहते हैं? हाँ या ना बोलिए।"
            elif self._lang != "en":
                confirm_msg = _regional_message(self._lang, "exit_confirm") or "Do you want to stop? Say yes or no."
            else:
                confirm_msg = "Do you want to stop the test? Say yes or no."
            await self.tts.speak(confirm_msg)
            return

        # Get current FSM state
        row = self.session.current_row
        if row is None:
            return

        response_type = row.response_type

        matched_option = None
        confidence = 0.0
        chart_param = row.chart_param or "400"
        if (
            row.state in ("B", "D")
            and response_type == "READABILITY"
            and looks_like_chart_letter_utterance(transcript, chart_param)
        ):
            letters = resolve_chart_letters(chart_param)
            detector = ChartReadingDetector()
            matched_option = detector.get_chart_intent(
                [row.opt_1, row.opt_2, row.opt_3],
                transcript,
                letters,
            )
            confidence = 100.0
            print(f"[VOICE] Chart LCS intent: {matched_option} (chart_param={chart_param})")
        else:
            matched_option, confidence = match_transcript(
                transcript, response_type, self._confidence_threshold
            )

        # Compute ambient RMS
        if len(audio) > 0:
            audio_float = audio.astype(np.float32) / 32768.0
            ambient_rms = float(np.sqrt(np.mean(audio_float ** 2)))
        else:
            ambient_rms = 0.0

        # Record utterance audio for HITL
        try:
            self._recorder.record_utterance(
                audio_int16=audio,
                transcript=transcript,
                response_type=response_type,
                matched_option=matched_option,
                confidence=confidence,
                fsm_state=row.state,
                phase_name=row.phase_name or "",
                ambient_rms=ambient_rms,
            )
        except Exception as e:
            print(f"[RECORDER] Error: {e}")

        if matched_option:
            print(f"[VOICE] Matched: {matched_option} (confidence: {confidence:.1f})")
            await self.ws_send_json({
                "type": "match",
                "option": matched_option,
                "confidence": confidence,
                "transcript": transcript,
            })

            next_state = self.session.process_response(matched_option)
            await self.flush_audio_buffers()
            await self.ws_send_json({
                "type": "state_update",
                "data": {"status": "active", **next_state},
            })

            if next_state.get("is_terminal"):
                self._cancel_silence_timer()
                if self._lang == "en":
                    end_msg = "The eye test is now complete. Thank you for your patience."
                elif self._lang == "hi":
                    end_msg = "परीक्षा पूरी हो गई है। धन्यवाद।"
                else:
                    end_msg = _regional_message(self._lang, "test_complete") or "The eye test is now complete. Thank you."
                await self.tts.speak(end_msg)
                await self.ws_send_json({"type": "test_complete"})
                return

            if next_state.get("auto_flip") and next_state.get("jcc_flip") == "flip1":
                flip_wait = next_state.get("flip_wait_seconds", 2)
                _flip_pairs_en = [
                    ("This is one.", "This is two. Which is better, one or two?"),
                    ("This is the first.", "And this is the second. First or second?"),
                    ("Option one.", "Option two. Which option was better?"),
                    ("Number one.", "Number two. Which number was better?"),
                    ("Here is view one.", "Here is view two. Which view was better?"),
                ]
                _flip_pairs_hi = [
                    ("यह है एक।", "यह है दो। कौन सा बेहतर है, एक या दो?"),
                    ("यह पहला है।", "यह दूसरा है। पहला या दूसरा?"),
                    ("विकल्प एक।", "विकल्प दो। कौन सा विकल्प बेहतर था?"),
                    ("नंबर एक।", "नंबर दो। कौन सा नंबर बेहतर था?"),
                    ("पहला दृश्य।", "दूसरा दृश्य। कौन सा दृश्य बेहतर था?"),
                ]
                if self._lang == "hi":
                    pair = random.choice(_flip_pairs_hi)
                elif self._lang != "en":
                    # Use regional flip messages
                    f1 = _regional_message(self._lang, "flip1")
                    f2 = _regional_message(self._lang, "flip2")
                    pair = (f1 or "This is one.", f2 or "This is two. Which is better?")
                else:
                    pair = random.choice(_flip_pairs_en)
                flip1_msg = pair[0]
                self._pending_flip2_msg = pair[1]
                asyncio.create_task(self._handle_jcc_flip1_then_flip2(flip1_msg, flip_wait))
                return

            question = next_state.get("question", "")
            if question:
                self._has_rephrased = False
                self._cancel_silence_timer()
                current_state = next_state.get("state")
                response_type = next_state.get("response_type", "")
                if current_state and current_state == self._prev_state:
                    if self._lang == "en":
                        followup = _pick_followup(response_type, state=current_state)
                    elif self._lang == "hi":
                        followup = _pick_hindi_followup(response_type, state=current_state)
                    else:
                        followup = _regional_followup(self._lang, response_type, state=current_state) or _pick_followup(response_type, state=current_state)
                    if followup:
                        question = followup
                elif self._lang == "en":
                    question = _strip_intents(question)
                elif self._lang == "hi":
                    question = _translate_to_hindi(question)
                else:
                    question = _regional_translate(self._lang, question) or _strip_intents(question)
                self._prev_state = current_state
                await self.tts.speak(question)
                self.start_silence_timer()
        else:
            # Try intent classifier fallback
            if self._intent_classifier and hasattr(self._recorder, '_session_dir'):
                try:
                    last_utt = self._recorder._utt_counter
                    audio_file = self._recorder._session_dir / f"utt_{last_utt:04d}.flac"
                    if not audio_file.exists():
                        audio_file = self._recorder._session_dir / f"utt_{last_utt:04d}.wav"
                    if audio_file.exists():
                        clf_intent, clf_conf = self._intent_classifier.predict(str(audio_file))
                        if clf_intent and clf_conf > 70:
                            print(f"[VOICE] Classifier fallback: {clf_intent} ({clf_conf:.0f}%)")
                            await self.ws_send_json({
                                "type": "match", "option": clf_intent,
                                "confidence": clf_conf, "transcript": transcript, "source": "classifier",
                            })
                            next_state = self.session.process_response(clf_intent)
                            await self.flush_audio_buffers()
                            await self.ws_send_json({"type": "state_update", "data": {"status": "active", **next_state}})
                            question = next_state.get("question", "")
                            if question:
                                self._has_rephrased = False
                                self._cancel_silence_timer()
                                if self._lang == "hi":
                                    question = _translate_to_hindi(question)
                                elif self._lang != "en":
                                    question = _regional_translate(self._lang, question) or _strip_intents(question)
                                else:
                                    question = _strip_intents(question)
                                self._prev_state = next_state.get("state")
                                await self.tts.speak(question)
                                self.start_silence_timer()
                            return
                except Exception as e:
                    print(f"[VOICE] Classifier fallback error: {e}")

            print(f"[VOICE] No match for: '{transcript}'")
            await self.ws_send_json({"type": "no_match", "transcript": transcript})
            if self._lang == "en":
                no_match_msg = "I didn't catch that clearly. Could you please repeat?"
            elif self._lang == "hi":
                no_match_msg = "समझ नहीं आया। कृपया फिर से बोलिए।"
            else:
                no_match_msg = _regional_message(self._lang, "no_match") or "I didn't catch that. Please repeat."
            await self.tts.speak(no_match_msg)
            self.start_silence_timer()

    async def _transcribe_and_process(self):
        """Run local Whisper STT on the speech buffer, then process."""
        audio = self._speech_buffer.copy()
        self._speech_buffer = np.array([], dtype=np.int16)

        if len(audio) < SAMPLE_RATE * 0.3:
            return

        audio_float = audio.astype(np.float32) / 32768.0
        segments = await asyncio.to_thread(self._run_whisper, audio_float)
        transcript = " ".join(segments).strip()

        if not transcript:
            return

        await self._process_transcript(transcript, audio)

    def _run_whisper(self, audio_float: np.ndarray) -> list:
        """Run faster-whisper transcription (called in thread)."""
        # Map language to Whisper language code
        lang = WHISPER_LANG_CODES.get(self._lang, "en")
        segments, _ = self._whisper.transcribe(
            audio_float,
            language=lang,
            beam_size=1,
            vad_filter=True,
        )
        return [seg.text for seg in segments]

    async def _handle_jcc_flip1_then_flip2(self, flip1_msg: str, wait_seconds: float):
        """Speak Flip 1 instruction, wait, then auto-flip to Flip 2 and ask."""
        await self.tts.speak(flip1_msg)
        await asyncio.sleep(wait_seconds)

        next_state = self.session.process_response("AUTO_FLIP")
        await self.flush_audio_buffers()
        await self.ws_send_json({
            "type": "state_update",
            "data": {"status": "active", **next_state},
        })
        # Use a clean flip2 question — NOT the verbose orchestrator message
        flip2_msg = getattr(self, '_pending_flip2_msg', None)
        if not flip2_msg:
            if self._lang == "hi":
                flip2_msg = "यह दूसरा है। पहला या दूसरा?"
            elif self._lang != "en":
                flip2_msg = _regional_message(self._lang, "flip2") or "This is two. Which is better?"
            else:
                flip2_msg = "This is two. Which is better, one or two?"
        self._has_rephrased = False
        self._cancel_silence_timer()
        await self.tts.speak(flip2_msg)
        self.start_silence_timer()

    def stop(self):
        self._running = False
        self._cancel_silence_timer()
        if self._deepgram_client:
            asyncio.create_task(self._deepgram_client.close())
        try:
            self._recorder.write_session_summary()
        except Exception as e:
            print(f"[RECORDER] Summary error: {e}")


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
