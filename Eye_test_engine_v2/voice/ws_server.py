"""FastAPI WebSocket server for voice pipeline.

Runs on a separate port (default 8766) alongside the Flask API server.
Streams audio between browser and the voice pipeline.

WebSocket protocol:
  Browser → Server: binary PCM audio (16-bit, 16kHz, mono)
  Server → Browser: binary (0x01 prefix + PCM int16 audio) or JSON text messages
"""

import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice.pipeline import build_pipeline, SAMPLE_RATE

fastapi_app = FastAPI(title="Eye Test Voice Pipeline")

_sessions_ref = None


def set_sessions_ref(sessions_dict):
    global _sessions_ref
    _sessions_ref = sessions_dict


@fastapi_app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str, lang: str = "en"):
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
    print(f"[VOICE WS] Connected: session={session_id}")

    async def ws_send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            print(f"[VOICE WS] Send JSON error: {e}")

    async def ws_send_bytes(data: bytes):
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            print(f"[VOICE WS] Send bytes error: {e}")

    # Build pipeline + TTS
    pipeline, tts = build_pipeline(
        session=session,
        ws_send_json=ws_send_json,
        ws_send_bytes=ws_send_bytes,
        lang=lang,
    )

    # Send initial state
    state = session._build_response()
    await ws_send_json({
        "type": "state_update",
        "data": {"status": "active", **state},
    })

    # Speak the first question
    question = state.get("question", "")
    if question and not (state.get("auto_flip") and state.get("jcc_flip") == "flip1"):
        asyncio.create_task(tts.speak(question))

    await ws_send_json({"type": "voice_ready", "tts_sample_rate": tts.sample_rate})

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data and data["bytes"]:
                # Feed mic audio directly into our pipeline
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
                        print(f"[VOICE WS] Stop requested: session={session_id}")
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        print(f"[VOICE WS] Disconnected: session={session_id}")
    except Exception as e:
        print(f"[VOICE WS] Error: {e}")
    finally:
        pipeline.stop()
        tts.stop()
        print(f"[VOICE WS] Cleaned up: session={session_id}")
