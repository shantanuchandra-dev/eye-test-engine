#!/usr/bin/env python3
"""Central server sync for multi-clinic audio data.

Pushes local audio recordings + manifests to a central server.
Pulls retrained models back to the clinic machine.

Designed for daily sync via cron:
    0 2 * * * cd /path/to && venv/bin/python -m voice.training.server_sync --push
    0 4 * * * cd /path/to && venv/bin/python -m voice.training.server_sync --pull

Sync protocol:
    - Push: rsync or HTTP POST of new audio files + manifests
    - Pull: HTTP GET of latest model versions

Configuration via environment variables:
    SYNC_SERVER_URL=https://central.example.com
    SYNC_API_KEY=xxx
    SYNC_CLINIC_ID=clinic_delhi_01
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AUDIO_BASE_DIR = Path.home() / ".eye_test_audio"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SYNC_STATE_FILE = AUDIO_BASE_DIR / ".sync_state.json"


def get_config():
    """Load sync configuration from environment."""
    return {
        "server_url": os.environ.get("SYNC_SERVER_URL", ""),
        "api_key": os.environ.get("SYNC_API_KEY", ""),
        "clinic_id": os.environ.get("SYNC_CLINIC_ID", "default_clinic"),
        "rsync_target": os.environ.get("SYNC_RSYNC_TARGET", ""),  # user@host:/path/
    }


def load_sync_state() -> dict:
    """Load the last sync state."""
    if SYNC_STATE_FILE.exists():
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    return {"last_push": None, "last_pull": None, "pushed_files": []}


def save_sync_state(state: dict):
    """Save the sync state."""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def find_new_files(since: str = None) -> list:
    """Find audio files that haven't been synced yet."""
    files = []
    if not AUDIO_BASE_DIR.exists():
        return files

    for date_dir in sorted(AUDIO_BASE_DIR.iterdir()):
        if not date_dir.is_dir() or date_dir.name.startswith(".") or date_dir.name.startswith("_"):
            continue
        if since and date_dir.name < since[:10]:
            continue
        for sess_dir in sorted(date_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            for f in sess_dir.iterdir():
                if f.suffix in (".flac", ".wav", ".jsonl", ".json"):
                    files.append(str(f.relative_to(AUDIO_BASE_DIR)))

    return files


def push_via_rsync(config: dict, state: dict):
    """Push audio data to central server via rsync."""
    target = config["rsync_target"]
    if not target:
        print("[SYNC] SYNC_RSYNC_TARGET not set. Cannot push via rsync.")
        return False

    clinic_id = config["clinic_id"]
    remote_path = f"{target}/{clinic_id}/"

    print(f"[SYNC] Pushing to {remote_path}...")

    # rsync with compression, only new files
    cmd = [
        "rsync", "-avz", "--progress",
        "--exclude", ".*",
        "--exclude", "_*",
        str(AUDIO_BASE_DIR) + "/",
        remote_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout[-500:] if result.stdout else "")
        if result.returncode == 0:
            print(f"[SYNC] Push complete")
            state["last_push"] = datetime.now().isoformat()
            save_sync_state(state)
            return True
        else:
            print(f"[SYNC] Push failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[SYNC] Push error: {e}")
        return False


def push_via_http(config: dict, state: dict):
    """Push audio data to central server via HTTP API."""
    server_url = config["server_url"]
    api_key = config["api_key"]
    clinic_id = config["clinic_id"]

    if not server_url:
        print("[SYNC] SYNC_SERVER_URL not set. Cannot push via HTTP.")
        return False

    import urllib.request
    import urllib.error

    # Find new files since last push
    since = state.get("last_push")
    new_files = find_new_files(since=since)
    pushed = state.get("pushed_files", [])
    to_push = [f for f in new_files if f not in pushed]

    if not to_push:
        print(f"[SYNC] No new files to push.")
        return True

    print(f"[SYNC] Pushing {len(to_push)} new files to {server_url}...")

    # Push manifests first, then audio files
    manifests = [f for f in to_push if f.endswith(".jsonl") or f.endswith(".json")]
    audio = [f for f in to_push if f.endswith(".flac") or f.endswith(".wav")]

    success_count = 0
    for file_list in [manifests, audio]:
        for rel_path in file_list:
            full_path = AUDIO_BASE_DIR / rel_path
            if not full_path.exists():
                continue

            url = f"{server_url}/api/sync/upload"
            try:
                with open(full_path, "rb") as f:
                    data = f.read()

                headers = {
                    "X-API-Key": api_key,
                    "X-Clinic-ID": clinic_id,
                    "X-File-Path": rel_path,
                    "Content-Type": "application/octet-stream",
                }
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        success_count += 1
                        pushed.append(rel_path)
            except Exception as e:
                print(f"[SYNC] Failed to push {rel_path}: {e}")

    state["last_push"] = datetime.now().isoformat()
    state["pushed_files"] = pushed[-10000:]  # keep last 10k entries
    save_sync_state(state)
    print(f"[SYNC] Pushed {success_count}/{len(to_push)} files")
    return True


def pull_models(config: dict, state: dict):
    """Pull retrained models from central server."""
    server_url = config["server_url"]
    api_key = config["api_key"]
    clinic_id = config["clinic_id"]

    if not server_url:
        # Try rsync
        target = config["rsync_target"]
        if target:
            return pull_models_rsync(config, state)
        print("[SYNC] No server configured for model pull.")
        return False

    import urllib.request
    import urllib.error

    url = f"{server_url}/api/sync/models/latest"
    headers = {"X-API-Key": api_key, "X-Clinic-ID": clinic_id}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta = json.loads(resp.read())

        available_version = meta.get("version")
        model_url = meta.get("download_url")

        if not available_version or not model_url:
            print("[SYNC] No models available on server.")
            return True

        # Check if we already have this version
        local_path = MODELS_DIR / "whisper-finetuned" / available_version
        if local_path.exists():
            print(f"[SYNC] Already have model {available_version}")
            return True

        print(f"[SYNC] Downloading model {available_version}...")
        # Download as tar.gz
        local_path.mkdir(parents=True, exist_ok=True)
        tar_path = local_path / "model.tar.gz"
        urllib.request.urlretrieve(model_url, str(tar_path))

        # Extract
        subprocess.run(["tar", "xzf", str(tar_path), "-C", str(local_path)], check=True)
        tar_path.unlink()

        state["last_pull"] = datetime.now().isoformat()
        save_sync_state(state)
        print(f"[SYNC] Model {available_version} downloaded to {local_path}")
        return True

    except Exception as e:
        print(f"[SYNC] Pull error: {e}")
        return False


def pull_models_rsync(config: dict, state: dict):
    """Pull models via rsync."""
    target = config["rsync_target"]
    clinic_id = config["clinic_id"]
    remote_models = f"{target}/models/"
    local_models = str(MODELS_DIR / "whisper-finetuned") + "/"

    Path(local_models).mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync", "-avz", "--progress",
        remote_models,
        local_models,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            state["last_pull"] = datetime.now().isoformat()
            save_sync_state(state)
            print(f"[SYNC] Models synced from {remote_models}")
            return True
        else:
            print(f"[SYNC] Model pull failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[SYNC] Pull error: {e}")
        return False


def show_status():
    """Show current sync status."""
    config = get_config()
    state = load_sync_state()

    print("=" * 50)
    print("SYNC STATUS")
    print("=" * 50)
    print(f"  Clinic ID:     {config['clinic_id']}")
    print(f"  Server URL:    {config['server_url'] or '(not set)'}")
    print(f"  Rsync Target:  {config['rsync_target'] or '(not set)'}")
    print(f"  Last Push:     {state.get('last_push', 'never')}")
    print(f"  Last Pull:     {state.get('last_pull', 'never')}")
    print(f"  Pushed Files:  {len(state.get('pushed_files', []))}")

    # Count local data
    new_files = find_new_files()
    pushed = state.get("pushed_files", [])
    pending = len([f for f in new_files if f not in pushed])
    print(f"  Local Files:   {len(new_files)}")
    print(f"  Pending Push:  {pending}")


def main():
    parser = argparse.ArgumentParser(description="Central server sync")
    parser.add_argument("--push", action="store_true", help="Push audio data to server")
    parser.add_argument("--pull", action="store_true", help="Pull retrained models from server")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    args = parser.parse_args()

    config = get_config()
    state = load_sync_state()

    if args.status or (not args.push and not args.pull):
        show_status()
        return

    if args.push:
        if config["rsync_target"]:
            push_via_rsync(config, state)
        elif config["server_url"]:
            push_via_http(config, state)
        else:
            print("[SYNC] Set SYNC_SERVER_URL or SYNC_RSYNC_TARGET to enable push.")

    if args.pull:
        pull_models(config, state)


if __name__ == "__main__":
    main()
