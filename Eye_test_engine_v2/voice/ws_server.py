"""FastAPI WebSocket server for voice pipeline.

Runs on a separate port (default 8766) alongside the Flask API server.
"""

import asyncio
import json
import os

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice.pipeline import (
    VoicePipeline, DirectTTSProcessor,
    _resolve_whisper_model, _resolve_piper_voice,
    _resolve_silero_onnx_path,
    PIPER_MODEL_DIR, SAMPLE_RATE,
)

fastapi_app = FastAPI(title="Eye Test Voice Pipeline")

_sessions_ref = None

# Pre-loaded models (shared across sessions, loaded once at startup)
_whisper_model = None


def set_sessions_ref(sessions_dict):
    global _sessions_ref
    _sessions_ref = sessions_dict


def preload_models():
    """Call once at startup to pre-load heavy models."""
    global _whisper_model
    from faster_whisper import WhisperModel

    silero_onnx = _resolve_silero_onnx_path()
    if not silero_onnx.is_file():
        print(f"[VOICE] WARNING: Silero VAD ONNX missing at {silero_onnx}")
        print("[VOICE] Run: python -m voice.download_models")
    else:
        print(f"[VOICE] Silero VAD ONNX: {silero_onnx}")

    stt_default = os.environ.get("VOICE_STT", "whisper").lower().strip()
    has_deepgram = bool(os.environ.get("DEEPGRAM_API_KEY", "").strip())
    if stt_default == "deepgram" and has_deepgram:
        print("[VOICE] Skipping Whisper preload (VOICE_STT=deepgram, DEEPGRAM_API_KEY set)")
        _whisper_model = None
    else:
        whisper_path = _resolve_whisper_model()
        print(f"[VOICE] Pre-loading Whisper model: {whisper_path}")
        _whisper_model = WhisperModel(whisper_path, device="cpu", compute_type="int8")
        print("[VOICE] Whisper model loaded")


@fastapi_app.websocket("/ws/voice/{session_id}")
async def voice_websocket(
    websocket: WebSocket,
    session_id: str,
    lang: str = "en",
    voice: str = "",
):
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

    env_stt = os.environ.get("VOICE_STT", "whisper").lower().strip()
    # Read stt from query only if present; omitting stt uses VOICE_STT (not hard-coded whisper).
    stt_q = (websocket.query_params.get("stt") or "").strip()
    effective_stt = stt_q.lower() if stt_q else (env_stt or "whisper")
    if effective_stt not in ("whisper", "deepgram"):
        effective_stt = "whisper"
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    deepgram_model = os.environ.get("DEEPGRAM_MODEL", "nova-2").strip()

    if effective_stt == "deepgram" and not deepgram_key:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Deepgram STT requested but DEEPGRAM_API_KEY is not set on the server.",
        }))
        await websocket.close(code=1008)
        return

    if effective_stt == "whisper" and _whisper_model is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Local Whisper is not loaded. Unset VOICE_STT=deepgram or use ?stt=deepgram with DEEPGRAM_API_KEY.",
        }))
        await websocket.close(code=1008)
        return

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

    if voice == "meta-mms-hindi":
        print("[VOICE WS] voice=meta-mms-hindi → Piper Hindi (hi_IN-pratham-medium)", flush=True)
        piper_voice = _resolve_piper_voice(voice_name=None, lang="hi")
    else:
        piper_voice = _resolve_piper_voice(voice_name=voice if voice else None, lang=lang)
    piper_onnx = str(PIPER_MODEL_DIR / f"{piper_voice}.onnx")
    print(f"[VOICE WS] Using voice: Piper {piper_voice}", flush=True)
    tts = DirectTTSProcessor(
        voice_path=piper_onnx,
        ws_send_bytes=ws_send_bytes,
        ws_send_json=ws_send_json,
    )

    whisper_for_session = None if effective_stt == "deepgram" else _whisper_model

    # Build pipeline using Whisper or Deepgram STT
    try:
        pipeline = VoicePipeline(
            session=session,
            ws_send_json=ws_send_json,
            tts=tts,
            silero_onnx_path=_resolve_silero_onnx_path(),
            whisper_model=whisper_for_session,
            lang=lang,
            stt_backend=effective_stt,
            deepgram_api_key=deepgram_key or None,
            deepgram_model=deepgram_model,
        )
        print(f"[VOICE WS] Ready: session={session_id} stt={effective_stt}", flush=True)
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

    # First prompt: await before receive loop so mic is armed when client gets voice_ready.
    # JCC flip1: skip speaking the long orchestrator line; still arm mic (was previously neither branch).
    question = (state.get("question") or "").strip()
    is_flip1_initial = bool(state.get("auto_flip") and state.get("jcc_flip") == "flip1")
    try:
        if question and not is_flip1_initial:
            if lang == "hi":
                from voice.pipeline import _translate_to_hindi
                question = _translate_to_hindi(question)
            await pipeline.speak_prompt_and_open_listening(question)
        else:
            await pipeline.open_listening_after_boot()
    except Exception as e:
        print(f"[VOICE WS] Initial prompt / boot failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        pipeline.force_open_listening()

    await ws_send_json({
        "type": "voice_ready",
        "tts_sample_rate": tts.sample_rate,
        "stt": effective_stt,
    })

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
