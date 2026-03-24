"""Entry point for ETE_v2."""
import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_server import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    print(f"Starting ETE_v2 on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
