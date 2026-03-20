#!/usr/bin/env python3
"""Central server for multi-clinic audio aggregation.

Receives audio + manifests from clinic machines, stores centrally,
and serves retrained models back.

Usage:
    python -m voice.training.central_server [--port 9000]

This runs a standalone Flask server (separate from the main eye test server).
Deploy on your central machine.

Environment variables:
    CENTRAL_API_KEY=xxx           # shared secret for clinic auth
    CENTRAL_DATA_DIR=/data/audio  # where to store received data (default: ~/eye_test_central/)
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

CENTRAL_DATA_DIR = Path(os.environ.get("CENTRAL_DATA_DIR", str(Path.home() / "eye_test_central")))
CENTRAL_MODELS_DIR = CENTRAL_DATA_DIR / "models"
API_KEY = os.environ.get("CENTRAL_API_KEY", "changeme")


def _check_auth():
    """Check API key from request headers."""
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    return None


@app.route("/api/sync/upload", methods=["POST"])
def upload_file():
    """Receive an audio/manifest file from a clinic machine."""
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    clinic_id = request.headers.get("X-Clinic-ID", "unknown")
    file_path = request.headers.get("X-File-Path", "")

    if not file_path:
        return jsonify({"error": "X-File-Path header required"}), 400

    # Sanitize path
    file_path = file_path.replace("..", "").lstrip("/")
    dest = CENTRAL_DATA_DIR / "clinics" / clinic_id / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = request.get_data()
    with open(dest, "wb") as f:
        f.write(data)

    print(f"[CENTRAL] Received: {clinic_id}/{file_path} ({len(data)} bytes)")
    return jsonify({"status": "ok", "path": file_path, "size": len(data)})


@app.route("/api/sync/models/latest", methods=["GET"])
def get_latest_model():
    """Return metadata about the latest retrained model."""
    _check_auth()

    if not CENTRAL_MODELS_DIR.exists():
        return jsonify({"version": None, "download_url": None})

    versions = sorted(
        [d.name for d in CENTRAL_MODELS_DIR.iterdir()
         if d.is_dir() and d.name.startswith("v")],
        key=lambda v: int(v[1:]) if v[1:].isdigit() else 0,
    )

    if not versions:
        return jsonify({"version": None, "download_url": None})

    latest = versions[-1]
    meta_path = CENTRAL_MODELS_DIR / latest / "training_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Build download URL
    server_url = request.host_url.rstrip("/")
    download_url = f"{server_url}/api/sync/models/{latest}/download"

    return jsonify({
        "version": latest,
        "download_url": download_url,
        "meta": meta,
    })


@app.route("/api/sync/models/<version>/download", methods=["GET"])
def download_model(version):
    """Download a model version as tar.gz."""
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    model_dir = CENTRAL_MODELS_DIR / version
    if not model_dir.exists():
        return jsonify({"error": "Model not found"}), 404

    # Create tar.gz
    tar_path = CENTRAL_DATA_DIR / "tmp" / f"{version}.tar.gz"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(tar_path).replace(".tar.gz", ""), "gztar", str(model_dir))

    return send_file(str(tar_path), mimetype="application/gzip",
                     as_attachment=True, download_name=f"{version}.tar.gz")


@app.route("/api/sync/clinics", methods=["GET"])
def list_clinics():
    """List all clinics that have uploaded data."""
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    clinics_dir = CENTRAL_DATA_DIR / "clinics"
    if not clinics_dir.exists():
        return jsonify([])

    clinics = []
    for d in sorted(clinics_dir.iterdir()):
        if not d.is_dir():
            continue
        # Count files
        file_count = sum(1 for _ in d.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        clinics.append({
            "clinic_id": d.name,
            "file_count": file_count,
            "total_size_mb": round(total_size / 1024 / 1024, 1),
        })

    return jsonify(clinics)


@app.route("/api/sync/stats", methods=["GET"])
def sync_stats():
    """Global stats for the central server."""
    auth_err = _check_auth()
    if auth_err:
        return auth_err

    clinics_dir = CENTRAL_DATA_DIR / "clinics"
    total_files = 0
    total_size = 0
    total_utterances = 0
    clinic_count = 0

    if clinics_dir.exists():
        for clinic_dir in clinics_dir.iterdir():
            if not clinic_dir.is_dir():
                continue
            clinic_count += 1
            for f in clinic_dir.rglob("*"):
                if f.is_file():
                    total_files += 1
                    total_size += f.stat().st_size
                    if f.name == "manifest.jsonl":
                        with open(f) as mf:
                            total_utterances += sum(1 for line in mf if line.strip())

    model_count = 0
    if CENTRAL_MODELS_DIR.exists():
        model_count = sum(1 for d in CENTRAL_MODELS_DIR.iterdir() if d.is_dir())

    return jsonify({
        "clinics": clinic_count,
        "total_files": total_files,
        "total_size_mb": round(total_size / 1024 / 1024, 1),
        "total_utterances": total_utterances,
        "model_versions": model_count,
    })


def main():
    parser = argparse.ArgumentParser(description="Central server for multi-clinic sync")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    CENTRAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CENTRAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Central Server starting on {args.host}:{args.port}")
    print(f"Data dir: {CENTRAL_DATA_DIR}")
    print(f"API key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:]}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
