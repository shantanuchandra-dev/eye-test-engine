#!/usr/bin/env python3
"""Launch the Eye Test Engine v2 API server.

Set VOICE_ENABLED=true to also start the Pipecat voice WebSocket server on port 8766.
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

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") == "development"

    if VOICE_ENABLED:
        import threading
        from werkzeug.serving import make_server

        # Share the sessions dict with the voice WebSocket server
        from voice.ws_server import set_sessions_ref, fastapi_app
        set_sessions_ref(sessions)

        voice_port = int(os.environ.get("VOICE_PORT", 8766))

        # Start Flask in a background thread
        flask_server = make_server(host, port, app)
        flask_thread = threading.Thread(
            target=flask_server.serve_forever,
            daemon=True,
        )
        flask_thread.start()
        print(f"Flask API server running on http://{host}:{port}")

        # Start FastAPI/Pipecat on main thread (asyncio event loop)
        import uvicorn
        print(f"Voice WebSocket server starting on ws://{host}:{voice_port}/ws/voice/{{session_id}}")
        uvicorn.run(
            fastapi_app,
            host=host,
            port=voice_port,
            log_level="info",
        )
    else:
        app.run(host=host, port=port, debug=debug)
