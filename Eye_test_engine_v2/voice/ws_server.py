"""FastAPI WebSocket server for voice pipeline.

Runs on a separate port (default 8766) alongside the Flask API server.
"""

import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice.pipeline import (
    VoicePipeline, DirectTTSProcessor, MetaTTSProcessor,
    _resolve_whisper_model, _resolve_piper_voice, _resolve_silero_hub_dir,
    PIPER_MODEL_DIR, SAMPLE_RATE,
)

fastapi_app = FastAPI(title="Eye Test Voice Pipeline")

_sessions_ref = None

# Pre-loaded models (shared across sessions, loaded once at startup)
_whisper_model = None
_silero_hub = None
_mms_model = None
_mms_tokenizer = None


def set_sessions_ref(sessions_dict):
    global _sessions_ref
    _sessions_ref = sessions_dict


def preload_models():
    """Call once at startup to pre-load heavy models."""
    global _whisper_model, _silero_hub, _mms_model, _mms_tokenizer
    from faster_whisper import WhisperModel

    whisper_path = _resolve_whisper_model()
    _silero_hub = _resolve_silero_hub_dir()

    print(f"[VOICE] Pre-loading Whisper model: {whisper_path}")
    _whisper_model = WhisperModel(whisper_path, device="cpu", compute_type="int8")
    print(f"[VOICE] Whisper model loaded")
    print(f"[VOICE] Silero hub: {_silero_hub or 'default'}")

    # Pre-load Meta MMS-TTS Hindi model
    try:
        from transformers import VitsModel, AutoTokenizer
        print(f"[VOICE] Pre-loading Meta MMS-TTS Hindi model...")
        _mms_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
        _mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
        print(f"[VOICE] Meta MMS-TTS loaded (sample_rate={_mms_model.config.sampling_rate})")
    except Exception as e:
        print(f"[VOICE] Meta MMS-TTS failed to load: {e} (Hindi Meta voice will be unavailable)")


@fastapi_app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str, lang: str = "en", voice: str = "", stt: str = "deepgram"):
    await websocket.accept()

    if _sessions_ref is None:
        await websocket.send_text(json.dumps({"type": "error", "message": "Voice server not initialized"}))
        await websocket.close(code=1011)
        return

    if session_id not in _sessions_ref:
        await websocket.send_text(json.dumps({"type": "error", "message": f"Session {session_id} not found"}))
        await websocket.close(code=1008)
        return

    session = _sessions_ref[session_id]

    async def ws_send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def ws_send_bytes(data: bytes):
        try:
            await websocket.send_bytes(data)
        except Exception:
            pass

    # Build TTS based on selected voice
    # Meta MMS voices: meta-mms-hindi, meta-mms-tamil, meta-mms-kannada, etc.
    MMS_VOICE_MAP = {
        "meta-mms-hindi": "facebook/mms-tts-hin",
        "meta-mms-tamil": "facebook/mms-tts-tam",
        "meta-mms-kannada": "facebook/mms-tts-kan",
        "meta-mms-marathi": "facebook/mms-tts-mar",
        "meta-mms-gujarati": "facebook/mms-tts-guj",
        "meta-mms-bengali": "facebook/mms-tts-ben",
        "meta-mms-punjabi": "facebook/mms-tts-pan",
    }

    if voice in MMS_VOICE_MAP:
        mms_model_id = MMS_VOICE_MAP[voice]
        print(f"[VOICE WS] Using voice: Meta MMS {mms_model_id}", flush=True)
        if voice == "meta-mms-hindi" and _mms_model:
            # Use pre-loaded Hindi model
            tts = MetaTTSProcessor(
                ws_send_bytes=ws_send_bytes,
                ws_send_json=ws_send_json,
                model=_mms_model,
                tokenizer=_mms_tokenizer,
            )
        else:
            # Load MMS model on-demand for other languages
            tts = MetaTTSProcessor(
                ws_send_bytes=ws_send_bytes,
                ws_send_json=ws_send_json,
                model_id=mms_model_id,
            )
    else:
        piper_voice = _resolve_piper_voice(voice_name=voice if voice else None, lang=lang)
        piper_onnx = str(PIPER_MODEL_DIR / f"{piper_voice}.onnx")
        print(f"[VOICE WS] Using voice: Piper {piper_voice}", flush=True)
        tts = DirectTTSProcessor(
            voice_path=piper_onnx,
            ws_send_bytes=ws_send_bytes,
            ws_send_json=ws_send_json,
        )

    # Build pipeline using pre-loaded whisper model
    try:
        pipeline = VoicePipeline(
            session=session,
            ws_send_json=ws_send_json,
            tts=tts,
            silero_hub_dir=_silero_hub,
            whisper_model=_whisper_model,
            lang=lang,
            session_id=session_id,
            stt_engine=stt,
        )
        # Connect Deepgram if using cloud STT
        await pipeline.start()
        print(f"[VOICE WS] Ready: session={session_id}", flush=True)
    except Exception as e:
        print(f"[VOICE WS] Pipeline init FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()
        await ws_send_json({"type": "error", "message": f"Pipeline init failed: {e}"})
        await websocket.close(code=1011)
        return

    # Send initial state
    state = session._build_response()
    await ws_send_json({
        "type": "state_update",
        "data": {"status": "active", **state},
    })

    # Speak the first question, then start silence timer
    question = state.get("question", "")
    if question and not (state.get("auto_flip") and state.get("jcc_flip") == "flip1"):
        if lang == "hi":
            from voice.pipeline import _translate_to_hindi
            question = _translate_to_hindi(question)
        async def _speak_first_question():
            await tts.speak(question)
            pipeline.start_silence_timer()
        asyncio.create_task(_speak_first_question())

    await ws_send_json({"type": "voice_ready", "tts_sample_rate": tts.sample_rate})

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                try:
                    await pipeline.process_audio(data["bytes"])
                except Exception as e:
                    print(f"[VOICE WS] process_audio error: {e}")
                    import traceback
                    traceback.print_exc()

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "stop":
                        break
                    elif msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[VOICE WS] Error: {e}")
    finally:
        pipeline.stop()
        tts.stop()
        print(f"[VOICE WS] Closed: session={session_id}")
