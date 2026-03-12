#!/usr/bin/env python3
"""Launch the Eye Test Engine v2 API server."""
import os
import sys
from pathlib import Path

# Ensure this package directory is on sys.path
pkg_dir = str(Path(__file__).resolve().parent)
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from api_server import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host=host, port=port, debug=debug)
