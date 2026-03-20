#!/usr/bin/env python3
"""Launch the Eye Test Engine v2 API server.

Set VOICE_ENABLED=true to also start the voice WebSocket server.

Modes:
  - SINGLE_PORT=true (default on Railway): Flask + WebSocket on one port via FastAPI+ASGI.
  - SINGLE_PORT=false (local dev): Flask on PORT, voice WS on VOICE_PORT (8766).

Models are downloaded automatically if missing on first run.
"""
import os
import sys
from pathlib import Path

# Ensure this package directory is on sys.path
pkg_dir = str(Path(__file__).resolve().parent)
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from api_server import app, sessions

VOICE_ENABLED = os.environ.get("VOICE_ENABLED", "false").lower() == "true"
# Railway/cloud: single port mode (RAILWAY_ENVIRONMENT or SINGLE_PORT=true)
SINGLE_PORT = (
    os.environ.get("SINGLE_PORT", "").lower() == "true"
    or os.environ.get("RAILWAY_ENVIRONMENT", "") != ""
)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") == "development"

    if VOICE_ENABLED:
        # Auto-download voice models if missing
        from voice.download_models import ensure_models
        ensure_models()

        # Share the sessions dict with the voice WebSocket server
        from voice.ws_server import set_sessions_ref, preload_models, fastapi_app
        set_sessions_ref(sessions)

        # Pre-load heavy models before starting servers
        preload_models()

        if SINGLE_PORT:
            # Single-port mode: mount Flask inside FastAPI on the same port.
            # WebSocket at /ws/voice/{session_id}, everything else → Flask.
            from asgiref.wsgi import WsgiToAsgi
            from starlette.routing import Mount

            flask_asgi = WsgiToAsgi(app)
            fastapi_app.mount("/", flask_asgi)

            import uvicorn
            print(f"[SINGLE PORT] Flask + Voice WS on port {port}")
            uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
        else:
            # Dual-port mode (local dev): Flask on PORT, voice WS on VOICE_PORT
            import threading
            from werkzeug.serving import make_server

            voice_port = int(os.environ.get("VOICE_PORT", 8766))

            flask_server = make_server(host, port, app)
            flask_thread = threading.Thread(
                target=flask_server.serve_forever,
                daemon=True,
            )
            flask_thread.start()
            print(f"Flask API server running on http://{host}:{port}")

            import uvicorn
            print(f"Voice WebSocket server starting on ws://{host}:{voice_port}/ws/voice/{{session_id}}")
            uvicorn.run(fastapi_app, host=host, port=voice_port, log_level="info")
    else:
        app.run(host=host, port=port, debug=debug)
