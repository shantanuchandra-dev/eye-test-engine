"""HITL Review API — Flask blueprint for audio annotation review.

Provides endpoints for the review web UI to list, annotate, and export
utterance data recorded by the audio recorder.

Mounted at /api/review/* in the main Flask app.
"""

import hashlib
import json
import os
import secrets
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Blueprint, request, jsonify, send_file

import sys
from pathlib import Path

# Add Eye_test_engine_v2 to path so we can import audio_recorder
_engine_dir = Path(__file__).resolve().parent.parent / "Eye_test_engine_v2"
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from voice.audio_recorder import (
    AUDIO_BASE_DIR,
    load_all_utterances,
    update_utterance,
    export_training_dataset,
)

review_bp = Blueprint("review", __name__)

# ── Simple auth (file-based user store) ─────────────────────────────────
_USERS_FILE = AUDIO_BASE_DIR / ".reviewers.json"
_TOKENS = {}  # token -> {email, name}


def _load_users() -> dict:
    if _USERS_FILE.exists():
        with open(_USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _require_auth(f):
    """Decorator to require a valid auth token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token or token not in _TOKENS:
            return jsonify({"error": "Unauthorized"}), 401
        request.reviewer = _TOKENS[token]
        return f(*args, **kwargs)
    return decorated


# ── Auth endpoints ──────────────────────────────────────────────────────

@review_bp.route("/auth/register", methods=["POST"])
def register():
    """Register a new reviewer account."""
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()

    if not email or not password or not name:
        return jsonify({"error": "email, password, and name required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    users = _load_users()
    if email in users:
        return jsonify({"error": "Email already registered"}), 409

    users[email] = {
        "name": name,
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)

    return jsonify({"status": "registered", "email": email, "name": name})


@review_bp.route("/auth/login", methods=["POST"])
def login():
    """Login and get an auth token."""
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    users = _load_users()
    user = users.get(email)
    if not user or user["password_hash"] != _hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = secrets.token_hex(32)
    _TOKENS[token] = {"email": email, "name": user["name"]}

    return jsonify({"token": token, "email": email, "name": user["name"]})


# ── Utterance endpoints ─────────────────────────────────────────────────

@review_bp.route("/utterances", methods=["GET"])
@_require_auth
def list_utterances():
    """List utterances with filters.

    Query params:
        date: YYYY-MM-DD
        session: session ID
        needs_review: true/false
        reviewed: true/false
        reviewer: email of reviewer
        page: page number (default 1)
        per_page: items per page (default 50)
    """
    date_filter = request.args.get("date")
    session_filter = request.args.get("session")
    needs_review = request.args.get("needs_review")
    reviewed = request.args.get("reviewed")
    reviewer_filter = request.args.get("reviewer")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    # Convert string booleans
    if needs_review is not None:
        needs_review = needs_review.lower() == "true"
    if reviewed is not None:
        reviewed = reviewed.lower() == "true"

    utterances = load_all_utterances(
        date_filter=date_filter,
        needs_review=needs_review,
        reviewed=reviewed,
        session_id=session_filter,
    )

    # Filter by reviewer
    if reviewer_filter:
        utterances = [u for u in utterances if u.get("reviewed_by") == reviewer_filter]

    # Sort by timestamp descending (newest first)
    utterances.sort(key=lambda u: u.get("timestamp", ""), reverse=True)

    # Paginate
    total = len(utterances)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = utterances[start:end]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "utterances": page_items,
    })


@review_bp.route("/utterances/<session_id>/<utt_id>", methods=["PUT"])
@_require_auth
def annotate_utterance(session_id, utt_id):
    """Annotate/correct a single utterance.

    Body:
        correct_option: the correct FSM option (e.g. "READABLE")
        review_notes: optional notes
        is_garbage: true if this is noise/non-speech
    """
    data = request.json or {}
    reviewer = request.reviewer

    updates = {
        "reviewed": True,
        "reviewed_by": reviewer["email"],
        "reviewed_at": datetime.now().isoformat(),
    }

    if "correct_option" in data:
        updates["correct_option"] = data["correct_option"]
    if "review_notes" in data:
        updates["review_notes"] = data["review_notes"]
    if "is_garbage" in data:
        updates["is_garbage"] = bool(data["is_garbage"])

    success = update_utterance(session_id, utt_id, updates)
    if not success:
        return jsonify({"error": "Utterance not found"}), 404

    return jsonify({"status": "updated", "id": utt_id})


@review_bp.route("/bulk-approve", methods=["POST"])
@_require_auth
def bulk_approve():
    """Bulk approve correctly-matched utterances.

    Body:
        utterance_ids: list of {"session_id": "...", "utt_id": "..."}
    """
    data = request.json or {}
    items = data.get("utterance_ids", [])
    reviewer = request.reviewer
    count = 0

    for item in items:
        sid = item.get("session_id")
        uid = item.get("utt_id")
        if sid and uid:
            success = update_utterance(sid, uid, {
                "reviewed": True,
                "reviewed_by": reviewer["email"],
                "reviewed_at": datetime.now().isoformat(),
                "correct_option": None,  # means original match was correct
            })
            if success:
                count += 1

    return jsonify({"status": "approved", "count": count})


@review_bp.route("/stats", methods=["GET"])
@_require_auth
def review_stats():
    """Get review statistics."""
    all_utts = load_all_utterances()

    total = len(all_utts)
    needs_review = sum(1 for u in all_utts if u.get("needs_review") and not u.get("reviewed"))
    reviewed = sum(1 for u in all_utts if u.get("reviewed"))
    garbage = sum(1 for u in all_utts if u.get("is_garbage"))
    understood = sum(1 for u in all_utts if u.get("was_understood"))
    corrected = sum(1 for u in all_utts if u.get("reviewed") and u.get("correct_option"))

    # Per-reviewer stats
    reviewer_counts = {}
    for u in all_utts:
        rb = u.get("reviewed_by")
        if rb:
            reviewer_counts[rb] = reviewer_counts.get(rb, 0) + 1

    # Per-date stats
    date_counts = {}
    for u in all_utts:
        d = u.get("_date", "unknown")
        date_counts[d] = date_counts.get(d, 0) + 1

    # Accuracy (of understood utterances that were reviewed)
    reviewed_understood = [u for u in all_utts if u.get("reviewed") and u.get("was_understood")]
    correctly_matched = sum(1 for u in reviewed_understood if not u.get("correct_option"))
    accuracy = (correctly_matched / len(reviewed_understood) * 100) if reviewed_understood else 0

    return jsonify({
        "total_utterances": total,
        "needs_review": needs_review,
        "reviewed": reviewed,
        "garbage": garbage,
        "understood": understood,
        "not_understood": total - understood,
        "corrected": corrected,
        "accuracy": round(accuracy, 1),
        "by_reviewer": reviewer_counts,
        "by_date": date_counts,
    })


@review_bp.route("/export", methods=["GET"])
@_require_auth
def export_dataset():
    """Export reviewed data as training dataset.

    Query params:
        format: "whisper" or "intent" (default: "whisper")
    """
    fmt = request.args.get("format", "whisper")
    output_dir = str(AUDIO_BASE_DIR / "_exports")
    stats = export_training_dataset(output_dir, format=fmt)
    return jsonify(stats)


@review_bp.route("/audio/<path:audio_path>", methods=["GET"])
@_require_auth
def serve_audio(audio_path):
    """Serve an audio file for playback in the review UI."""
    full_path = AUDIO_BASE_DIR / audio_path
    if not full_path.exists() or not str(full_path).startswith(str(AUDIO_BASE_DIR)):
        return jsonify({"error": "Audio file not found"}), 404

    suffix = full_path.suffix.lower()
    mime = {
        ".flac": "audio/flac",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
    }.get(suffix, "audio/wav")

    return send_file(str(full_path), mimetype=mime)


@review_bp.route("/dates", methods=["GET"])
@_require_auth
def list_dates():
    """List available dates with utterance counts."""
    dates = {}
    if AUDIO_BASE_DIR.exists():
        for d in sorted(AUDIO_BASE_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("_"):
                count = 0
                for sess in d.iterdir():
                    manifest = sess / "manifest.jsonl"
                    if manifest.exists():
                        with open(manifest) as f:
                            count += sum(1 for line in f if line.strip())
                if count > 0:
                    dates[d.name] = count
    return jsonify(dates)


@review_bp.route("/sessions", methods=["GET"])
@_require_auth
def list_sessions():
    """List sessions for a given date."""
    date = request.args.get("date")
    sessions = []
    if date and AUDIO_BASE_DIR.exists():
        date_dir = AUDIO_BASE_DIR / date
        if date_dir.exists():
            for sess in sorted(date_dir.iterdir()):
                if not sess.is_dir():
                    continue
                manifest = sess / "manifest.jsonl"
                summary = sess / "session_summary.json"
                info = {"session_id": sess.name, "utterance_count": 0}
                if manifest.exists():
                    with open(manifest) as f:
                        info["utterance_count"] = sum(1 for line in f if line.strip())
                if summary.exists():
                    with open(summary) as f:
                        info["summary"] = json.load(f)
                sessions.append(info)
    return jsonify(sessions)
