"""
ETE_v2 Flask API Server

Endpoints:
- Session lifecycle (intake, respond, status, end, discard)
- Phoropter proxy (devices, acquire, release, heartbeat, run-tests, reset, etc.)
- Dashboard (stats, R&R, export, config)
- Debug logs (conversation, curl commands, responses)
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

from fsm.config.calibration_loader import CalibrationLoader
from session_orchestrator import SessionOrchestrator

# ── IO imports ──
from ete_io.ist_time import ist_now
from ete_io.outputs import (
    build_session_metadata,
    session_csv_string,
    write_session_csv,
    write_session_metadata,
    write_voice_utterances_csv,
    append_to_combined_log,
    append_to_combined_metadata,
)
from ete_io.remote_storage import remote_supabase_remote_only, upload_session
from ete_io.dashboard_data import (
    load_metadata_rows,
    filter_rows,
    get_rr_aggregates,
    count_today,
    export_metadata_columns,
)

# ── App paths ──
APP_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = APP_ROOT / "frontend"


def _resolve_app_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return APP_ROOT / path


# ── App setup ──
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# ── Configuration ──
PHOROPTER_BASE_URL = os.environ.get(
    "PHOROPTER_BASE_URL",
    "https://rajasthan-royals.preprod.lenskart.com",
)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5050")
CALIBRATION_PATH = _resolve_app_path(
    os.environ.get("CALIBRATION_PATH", "config/calibration.csv")
)

# ── Log paths ──
LOG_BASE = _resolve_app_path(os.environ.get("LOG_DIR", "logs"))
SESSIONS_DIR = LOG_BASE / "sessions"
COMBINED_LOG_PATH = LOG_BASE / "combined_log.csv"
COMBINED_METADATA_PATH = LOG_BASE / "combined_metadata.csv"
DASHBOARD_CONFIG_PATH = LOG_BASE / "dashboard_config.json"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory session store ──
sessions: dict[str, SessionOrchestrator] = {}


# ── Helper ──
def _request_payload() -> dict:
    """Extract JSON payload from request, handling both JSON and form data."""
    if request.is_json:
        return request.get_json(silent=True) or {}
    try:
        return json.loads(request.data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _proxy_request(method: str, path: str, body: Optional[dict] = None) -> tuple:
    """Proxy a request to the phoropter broker."""
    import requests as req
    url = f"{PHOROPTER_BASE_URL}{path}"

    # Log the curl command
    session_id = request.args.get("session_id") or ""
    if session_id and session_id in sessions:
        sessions[session_id].log_curl_command(method, url, body)

    try:
        if method == "GET":
            resp = req.get(url, timeout=10)
        elif method == "POST":
            print(f"[proxy → lenskart] POST {url}  body={body}")
            resp = req.post(url, json=body, timeout=15)
        else:
            return jsonify({"error": f"Unsupported method: {method}"}), 400

        try:
            data = resp.json()
        except Exception:
            data = resp.text

        print(f"[proxy ← lenskart] {resp.status_code}  {data}")
        return jsonify(data), resp.status_code
    except req.exceptions.RequestException as e:
        print(f"[proxy error] {method} {url}  error={e}")
        return jsonify({"error": str(e)}), 502


# ═══════════════════════════════════════════════════════════════════
# FRONTEND SERVING
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
        return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/intake")
def serve_intake():
    return send_from_directory(FRONTEND_DIR, "intake.html")


@app.route("/dashboard")
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


@app.route("/cal")
def serve_calibration():
    return send_from_directory(FRONTEND_DIR, "calibration.html")


# Pre-rendered cloud TTS clips (SHA-256 hex of UTF-8 phrase text; same strings as fsm_tts_phrases).
TTS_CACHE_DIR = Path(__file__).resolve().parent / "tts_cache"
SARVAM_TTS_CACHE_DIR = Path(__file__).resolve().parent / "sarvam_tts_cache"


def _serve_phrase_mp3(cache_dir: Path, phrase_id: str):
    if (
        len(phrase_id) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in phrase_id)
    ):
        abort(400)
    safe = phrase_id.lower()
    path = cache_dir / f"{safe}.mp3"
    if not path.is_file():
        abort(404)
    return send_from_directory(cache_dir, f"{safe}.mp3", mimetype="audio/mpeg")


@app.route("/api/tts/<phrase_id>.mp3")
def serve_tts_audio(phrase_id: str):
    """ElevenLabs cache (elevenlabs_tts_cache.py generate)."""
    return _serve_phrase_mp3(TTS_CACHE_DIR, phrase_id)


@app.route("/api/tts-sarvam/<phrase_id>.mp3")
def serve_sarvam_tts_audio(phrase_id: str):
    """Sarvam AI cache (sarvam_tts_cache.py generate)."""
    return _serve_phrase_mp3(SARVAM_TTS_CACHE_DIR, phrase_id)


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/config")
def get_config():
    return jsonify({
        "backend_url": BACKEND_URL,
        "phoropter_base_url": PHOROPTER_BASE_URL,
    })


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION EDITOR
# ═══════════════════════════════════════════════════════════════════

CALIBRATION_PASSWORD = "SidShan"


@app.route("/api/calibration", methods=["GET"])
def get_calibration():
    try:
        rows = CalibrationLoader.read_full(CALIBRATION_PATH)
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/calibration", methods=["PUT"])
def update_calibration():
    data = _request_payload()
    password = data.get("password", "")
    if password != CALIBRATION_PASSWORD:
        return jsonify({"error": "Invalid password"}), 403

    updates = data.get("parameters", {})
    if not updates:
        return jsonify({"error": "No parameters provided"}), 400

    try:
        count = CalibrationLoader.write_values(CALIBRATION_PATH, updates)
        return jsonify({"status": "saved", "updated_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# DEVICE MANAGEMENT (PROXY)
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/devices")
def list_devices():
    all_flag = request.args.get("all", "false")
    path = f"/devices?all={all_flag}" if all_flag == "true" else "/devices"
    return _proxy_request("GET", path)


@app.route("/api/devices/available")
def list_available_devices():
    """Return only devices that are free to acquire."""
    return _proxy_request("GET", "/devices/available")


@app.route("/api/devices/<device_id>")
def get_device(device_id):
    return _proxy_request("GET", f"/devices/{device_id}")


@app.route("/api/devices/<device_id>/acquire", methods=["POST"])
def acquire_device(device_id):
    # Read body robustly — try every possible source so silent failures are visible
    body = {}
    if request.is_json:
        body = request.get_json(force=True, silent=True) or {}
    if not body:
        try:
            body = json.loads(request.data.decode("utf-8"))
        except Exception:
            body = {}

    # Ensure the two fields the Lenskart API requires are always present
    brain_id    = body.get("brain_id", "brain_01")
    name        = body.get("name") or body.get("operator_name", "")

    payload = {"brain_id": brain_id, "name": name}
    app.logger.info(f"[acquire] device={device_id}  sending → {payload}")
    print(f"[acquire] device={device_id}  raw_body={body}  sending → {payload}")

    return _proxy_request("POST", f"/devices/{device_id}/acquire", payload)


@app.route("/api/devices/<device_id>/release", methods=["POST"])
def release_device(device_id):
    body = _request_payload()
    return _proxy_request("POST", f"/devices/{device_id}/release", body)


@app.route("/api/devices/<device_id>/heartbeat", methods=["POST"])
def heartbeat(device_id):
    body = _request_payload()
    return _proxy_request("POST", f"/devices/{device_id}/heartbeat", body)


@app.route("/api/brains")
def list_brains():
    return _proxy_request("GET", "/brains")


@app.route("/api/events")
def list_events():
    limit = request.args.get("limit", "20")
    return _proxy_request("GET", f"/events?limit={limit}")


# ═══════════════════════════════════════════════════════════════════
# PHOROPTER CONTROL (PROXY)
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/phoropter/<device_id>/reset", methods=["POST"])
def phoropter_reset(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/reset")


@app.route("/api/phoropter/<device_id>/run-tests", methods=["POST"])
def phoropter_run_tests(device_id):
    body = _request_payload()
    return _proxy_request("POST", f"/phoropter/{device_id}/run-tests", body)


@app.route("/api/phoropter/<device_id>/pinhole", methods=["POST"])
def phoropter_pinhole(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/pinhole")


@app.route("/api/phoropter/<device_id>/occluder", methods=["POST"])
def phoropter_occluder(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/occluder")


@app.route("/api/phoropter/<device_id>/screenshot", methods=["POST"])
def phoropter_screenshot(device_id):
    return _proxy_request("POST", f"/phoropter/{device_id}/screenshot")


@app.route("/api/phoropter/<device_id>/sync-state", methods=["POST"])
def phoropter_sync_state(device_id):
    body = _request_payload()
    return _proxy_request("POST", f"/phoropter/{device_id}/sync-state", body)


# ═══════════════════════════════════════════════════════════════════
# SESSION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/session/intake", methods=["POST"])
def session_intake():
    """Accept patient data, derive variables, initialize FSM, return first question."""
    data = _request_payload()
    phoropter_id = data.get("phoropter_id", "")
    patient = data.get("patient", data)

    session_id = f"session_{int(time.time() * 1000)}"

    orchestrator = SessionOrchestrator(
        calibration_path=CALIBRATION_PATH,
        phoropter_base_url=PHOROPTER_BASE_URL,
    )
    sessions[session_id] = orchestrator

    result = orchestrator.initialize(patient, session_id, phoropter_id)
    return jsonify(result)


@app.route("/api/voice/transcribe", methods=["POST"])
def voice_transcribe():
    """Transcribe audio using faster-whisper (same as FSMv3.1_R2).

    Accepts: { audio: "<base64 PCM16 mono 16kHz>", language: "en"|"hi"|"auto" }
    Returns: { text, detected_language, language_probability, stt_seconds, backend }
    """
    data = _request_payload()
    audio_b64 = data.get("audio", "")
    language = data.get("language", "auto")

    if not audio_b64:
        return jsonify({"error": "No audio data", "text": ""}), 400

    audio_format = data.get("audio_format", "pcm16")
    from voice_endpoint import transcribe_audio
    result = transcribe_audio(audio_b64, language_hint=language if language != "auto" else None, audio_format=audio_format)
    return jsonify(result)


@app.route("/api/voice/transcribe-and-match", methods=["POST"])
def voice_transcribe_and_match():
    """Full pipeline: transcribe audio with faster-whisper, then match response.
    Matches FSMv3.1_R2's _voice_response() flow exactly.

    Accepts: { audio, state, options, language, stimulus_letters }
    Returns: { transcript, accepted, response_value, confidence, method, stt_seconds, ... }
    """
    data = _request_payload()
    audio_b64 = data.get("audio", "")
    state = data.get("state", "")
    options = data.get("options", [])
    language = data.get("language", "auto")
    stimulus_letters = data.get("stimulus_letters")

    if not audio_b64:
        return jsonify({"error": "No audio data", "text": "", "accepted": False}), 400

    audio_format = data.get("audio_format", "pcm16")

    # Step 1: Transcribe
    from voice_endpoint import transcribe_audio
    stt_result = transcribe_audio(
        audio_b64,
        language_hint=language if language != "auto" else None,
        audio_format=audio_format,
    )

    if stt_result.get("error"):
        return jsonify({
            "accepted": False,
            "text": "",
            "error": stt_result["error"],
            "backend": stt_result.get("backend", "none"),
        })

    transcript = stt_result.get("text", "").strip()
    if not transcript:
        return jsonify({
            "accepted": False,
            "text": "",
            "error": "No speech detected",
            "backend": stt_result.get("backend"),
            "stt_seconds": stt_result.get("stt_seconds"),
        })

    # Step 2: Match
    try:
        from fsm.audio.response_matching import match_response, infer_response_language
        match = match_response(
            transcript=transcript,
            state=state,
            available_options=options,
            stimulus_letters=stimulus_letters,
        )
        inferred_lang = infer_response_language(
            transcript=transcript,
            detected_language=stt_result.get("detected_language"),
            detected_language_probability=stt_result.get("language_probability"),
        )
        return jsonify({
            "accepted": match.accepted,
            "response_value": match.response_value,
            "transcript": transcript,
            "confidence": match.confidence,
            "method": match.method,
            "canonical_label": match.canonical_label,
            "reason": match.reason,
            "reprompt": match.reprompt_text,
            "detected_language": stt_result.get("detected_language"),
            "language_probability": stt_result.get("language_probability"),
            "inferred_language": inferred_lang,
            "stt_seconds": stt_result.get("stt_seconds"),
            "backend": stt_result.get("backend"),
        })
    except Exception as e:
        return jsonify({
            "accepted": False,
            "transcript": transcript,
            "error": str(e),
            "stt_seconds": stt_result.get("stt_seconds"),
            "backend": stt_result.get("backend"),
        })


@app.route("/api/voice/status")
def voice_status():
    """Check if faster-whisper backend is available."""
    from voice_endpoint import is_whisper_available
    return jsonify({"whisper_available": is_whisper_available()})


@app.route("/api/voice/match", methods=["POST"])
def voice_match():
    """Match a voice transcript to an FSM response option.
    Uses fsm/audio/response_matching.py — the same logic as FSMv3.1_R2.
    """
    data = _request_payload()
    transcript = data.get("transcript", "")
    state = data.get("state", "")
    options = data.get("options", [])
    stimulus_letters = data.get("stimulus_letters")  # Chart letters for B/D states

    if not transcript or not options:
        return jsonify({"accepted": False, "reason": "Missing transcript or options"}), 400

    try:
        from fsm.audio.response_matching import match_response
        match = match_response(
            transcript=transcript,
            state=state,
            available_options=options,
            stimulus_letters=stimulus_letters,
        )
        return jsonify({
            "accepted": match.accepted,
            "response_value": match.response_value,
            "confidence": match.confidence,
            "method": match.method,
            "canonical_label": match.canonical_label,
            "reason": match.reason,
            "reprompt": match.reprompt_text,
        })
    except Exception as e:
        return jsonify({"accepted": False, "reason": str(e)}), 500


@app.route("/api/voice/labels", methods=["POST"])
def voice_labels():
    """Get localized option labels and question text for a given state."""
    data = _request_payload()
    state = data.get("state", "")
    language = data.get("language", "en")
    question = data.get("question", "")
    options = data.get("options", [])

    try:
        from fsm.audio.response_matching import (
            interactive_option_labels,
            localized_voice_prompt,
            localized_option_label,
        )

        display_labels = options or interactive_option_labels(state)
        localized_labels = []
        for label in display_labels:
            loc = localized_option_label(label, language, state=state, question=question)
            localized_labels.append({
                "internal": label,
                "display": loc if loc != label else label,
                "localized": loc,
            })

        localized_question = localized_voice_prompt(
            state=state,
            language=language,
            retry=False,
            fallback_question=question,
        )

        return jsonify({
            "labels": localized_labels,
            "question": localized_question,
            "language": language,
        })
    except Exception as e:
        return jsonify({
            "labels": [],
            "question": question,
            "language": language,
            "error": str(e),
        })


@app.route("/api/session/<session_id>/respond", methods=["POST"])
def session_respond(session_id):
    """Process patient response, return next question."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    data = _request_payload()
    response_value = data.get("response", data.get("response_value", ""))

    if not response_value:
        return jsonify({"error": "Missing response value"}), 400

    voice_meta = data.get("voice_meta")
    input_method = data.get("input_method", "Button")
    # Store language in session state
    language = data.get("language")
    if language:
        sessions[session_id].session_language = language
    result = sessions[session_id].process_response(
        response_value, voice_meta=voice_meta, input_method=input_method
    )
    return jsonify(result)


@app.route("/api/session/<session_id>/status")
def session_status(session_id):
    """Return full session state for UI restore."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    status = sessions[session_id].get_status()
    status["language"] = getattr(sessions[session_id], "session_language", "en")
    return jsonify(status)


@app.route("/api/session/<session_id>/derived-variables")
def session_derived_variables(session_id):
    """Return computed derived variables + AR/Lenso for debug display."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    orch = sessions[session_id]
    result = orch.get_derived_variables()
    # Include AR and Lensometry powers for the DV drawer
    pi = orch.patient_input
    if pi:
        ar_re = pi.autorefractor_re
        ar_le = pi.autorefractor_le
        lo_re = pi.lenso_re
        lo_le = pi.lenso_le
        result["_ar"] = {
            "re": {"sph": ar_re.sphere, "cyl": ar_re.cylinder, "axis": ar_re.axis} if ar_re else None,
            "le": {"sph": ar_le.sphere, "cyl": ar_le.cylinder, "axis": ar_le.axis} if ar_le else None,
        }
        result["_lenso"] = {
            "re": {"sph": lo_re.sphere, "cyl": lo_re.cylinder, "axis": lo_re.axis} if lo_re else None,
            "le": {"sph": lo_le.sphere, "cyl": lo_le.cylinder, "axis": lo_le.axis} if lo_le else None,
        }
    return jsonify(result)


@app.route("/api/session/<session_id>/sync-power", methods=["POST"])
def session_sync_power(session_id):
    """Sync manual power changes from frontend to backend."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    data = _request_payload()
    result = sessions[session_id].sync_power(data)
    return jsonify(result)


@app.route("/api/session/<session_id>/phoropter-dispatch", methods=["POST"])
def session_phoropter_dispatch(session_id):
    """Enable/disable phoropter auto-dispatch for this session."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    data = _request_payload()
    enabled = data.get("enabled", True)
    sessions[session_id].phoropter_auto_dispatch = bool(enabled)
    auto_screenshot = data.get("auto_screenshot")
    if auto_screenshot is not None:
        sessions[session_id].auto_screenshot = bool(auto_screenshot)
    return jsonify({
        "phoropter_auto_dispatch": bool(enabled),
        "auto_screenshot": sessions[session_id].auto_screenshot,
    })


@app.route("/api/session/<session_id>/jcc-flip", methods=["POST"])
def session_jcc_flip(session_id):
    """Send JCC handle command (flip from position 1 to position 2).
    Called by frontend after 2s auto-flip delay."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    orch = sessions[session_id]
    if not orch.phoropter_auto_dispatch or not orch.phoropter_id:
        return jsonify({"skipped": True})
    result = orch._send_jcc("handle")
    return jsonify({"flip": "flip2", "result": result})


@app.route("/api/session/<session_id>/screenshot", methods=["POST"])
def session_screenshot(session_id):
    """Capture a screenshot from the phoropter."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    orch = sessions[session_id]
    img = orch._capture_screenshot()
    if img:
        return jsonify({"screenshot": img})
    return jsonify({"error": "Screenshot failed"}), 502


@app.route("/api/session/<session_id>/failed-voice-attempts", methods=["POST"])
def session_failed_voice_attempts(session_id):
    """Store failed voice attempts from frontend."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    data = _request_payload()
    attempts = data.get("attempts", [])
    if not hasattr(sessions[session_id], "failed_voice_attempts"):
        sessions[session_id].failed_voice_attempts = []
    sessions[session_id].failed_voice_attempts.extend(attempts)
    return jsonify({"stored": len(attempts)})


@app.route("/api/session/<session_id>/end", methods=["POST"])
def session_end(session_id):
    """End session: write logs under LOG_DIR, or Supabase-only when REMOTE_STORAGE=supabase."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    data = _request_payload()
    orch = sessions[session_id]
    end_time = ist_now()

    # Build metadata
    metadata = build_session_metadata(
        session_id=session_id,
        phoropter_id=orch.phoropter_id,
        session_start_time=orch.session_start_time or end_time,
        session_end_time=end_time,
        completion_status="completed",
        rows=orch.session_history,
        ar=data.get("ar"),
        lensometry=data.get("lensometry"),
        phase_jump_count=orch.phase_jump_count,
        unable_to_read_count=orch.unable_to_read_count,
        phases_completed=orch.phases_completed,
        duration_per_phase=orch.duration_per_phase,
        operator_name=data.get("operator_name", ""),
        customer_name=data.get("customer_name") or (orch.patient_input.patient_name if orch.patient_input else ""),
        customer_age=data.get("customer_age", ""),
        customer_gender=data.get("customer_gender", ""),
        qualitative_feedback=data.get("qualitative_feedback", ""),
        patient_input=orch.patient_input,
        derived_variables=orch.derived_variables,
        calibration_snapshot=orch.calibration.get_snapshot(),
    )

    # Enrich metadata with voice/language stats (matching FSMv3.1_R2 summary.json)
    duration_secs = metadata.get("session_duration_seconds", 0)
    mins, secs = divmod(int(duration_secs), 60)
    metadata["total_test_duration_display"] = f"{mins}m {secs}s"
    metadata["session_language"] = getattr(orch, "session_language", data.get("language", "en"))
    metadata["input_mode"] = "voice_browser_speech_recognition"
    metadata["failed_voice_attempt_count"] = len(getattr(orch, "failed_voice_attempts", []))
    metadata["total_steps"] = len(orch.session_history)
    metadata["prompt_instance_count"] = orch.prompt_instance_id

    remote_only = remote_supabase_remote_only()

    # Write files (skipped when REMOTE_STORAGE=supabase — logs only in Supabase Storage)
    if not remote_only:
        csv_path = SESSIONS_DIR / f"{session_id}.csv"
        meta_path = SESSIONS_DIR / f"{session_id}_metadata.json"
        write_session_csv(orch.session_history, csv_path)
        write_session_metadata(metadata, meta_path)
        append_to_combined_log(orch.session_history, session_id, COMBINED_LOG_PATH)
        append_to_combined_metadata(metadata, COMBINED_METADATA_PATH)
        # Voice utterance training CSV
        voice_csv_path = SESSIONS_DIR / f"{session_id}_voice_utterances.csv"
        write_voice_utterances_csv(
            orch.session_history,
            getattr(orch, "failed_voice_attempts", []),
            session_id,
            voice_csv_path,
        )

    combined_log_for_remote: Optional[str] = None
    if not remote_only and COMBINED_LOG_PATH.exists():
        combined_log_for_remote = COMBINED_LOG_PATH.read_text(encoding="utf-8")

    # Failed voice attempts CSV (in memory for upload; local file only if not remote-only)
    failed_attempts = getattr(orch, "failed_voice_attempts", [])
    failed_voice_csv_content: Optional[str] = None
    if failed_attempts:
        import csv as _csv
        import io as _io

        keys = sorted(set().union(*(a.keys() for a in failed_attempts)))
        buf = _io.StringIO()
        writer = _csv.DictWriter(buf, fieldnames=keys)
        writer.writeheader()
        for attempt in failed_attempts:
            writer.writerow(attempt)
        failed_voice_csv_content = buf.getvalue()
        if not remote_only:
            fva_path = SESSIONS_DIR / f"{session_id}_failed_voice_attempts.csv"
            fva_path.parent.mkdir(parents=True, exist_ok=True)
            with fva_path.open("w", newline="", encoding="utf-8") as f:
                f.write(failed_voice_csv_content)

    # Remote upload (Supabase Storage when REMOTE_STORAGE=supabase; see ete_io.remote_storage)
    csv_content = session_csv_string(orch.session_history)
    upload_err = upload_session(
        session_id,
        csv_content,
        metadata,
        failed_voice_csv_content=failed_voice_csv_content,
        combined_log_csv_content=combined_log_for_remote if not remote_only else None,
        combined_log_merge=(
            (session_id, orch.session_history)
            if remote_only and orch.session_history
            else None
        ),
        combined_metadata_merge=metadata if remote_only else None,
    )

    result = {"status": "stored", "session_id": session_id}
    if upload_err:
        result["remote_upload_error"] = upload_err

    return jsonify(result)


@app.route("/api/session/<session_id>/discard", methods=["POST"])
def session_discard(session_id):
    """Discard session without storing."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    del sessions[session_id]
    return jsonify({"status": "discarded", "session_id": session_id})


# ═══════════════════════════════════════════════════════════════════
# DEBUG LOGS (password-gated in frontend)
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/session/<session_id>/logs/conversation")
def session_conversation_log(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[session_id].conversation_log)


@app.route("/api/session/<session_id>/logs/curl")
def session_curl_log(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[session_id].curl_log)


@app.route("/api/session/<session_id>/logs/responses")
def session_response_log(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[session_id].session_history)


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/dashboard/config")
def dashboard_config_get():
    if DASHBOARD_CONFIG_PATH.exists():
        return jsonify(json.loads(DASHBOARD_CONFIG_PATH.read_text()))
    return jsonify({"new_tests_enabled": True, "daily_limit": 100, "scope": "global"})


@app.route("/api/dashboard/config", methods=["PUT"])
def dashboard_config_put():
    data = _request_payload()
    DASHBOARD_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_CONFIG_PATH.write_text(json.dumps(data, indent=2))
    return jsonify(data)


@app.route("/api/dashboard/stats")
def dashboard_stats():
    from_str = request.args.get("from")
    to_str = request.args.get("to")
    operator = request.args.get("operator")
    phoropter = request.args.get("phoropter")

    from_date = date.fromisoformat(from_str) if from_str else None
    to_date = date.fromisoformat(to_str) if to_str else None

    rows = load_metadata_rows(COMBINED_METADATA_PATH)
    filtered = filter_rows(rows, from_date, to_date, operator, phoropter)

    today_count = count_today(COMBINED_METADATA_PATH)

    recent = filtered[-20:]
    recent.reverse()

    return jsonify({
        "total_sessions": len(filtered),
        "today_count": today_count,
        "recent_sessions": recent,
    })


@app.route("/api/dashboard/rr")
def dashboard_rr():
    from_str = request.args.get("from")
    to_str = request.args.get("to")
    operator = request.args.get("operator")
    phoropter = request.args.get("phoropter")

    from_date = date.fromisoformat(from_str) if from_str else None
    to_date = date.fromisoformat(to_str) if to_str else None

    rows = load_metadata_rows(COMBINED_METADATA_PATH)
    filtered = filter_rows(rows, from_date, to_date, operator, phoropter)

    return jsonify(get_rr_aggregates(filtered))


@app.route("/api/dashboard/export")
def dashboard_export():
    import io as _io
    import csv

    from_str = request.args.get("from")
    to_str = request.args.get("to")
    operator = request.args.get("operator")
    phoropter = request.args.get("phoropter")

    from_date = date.fromisoformat(from_str) if from_str else None
    to_date = date.fromisoformat(to_str) if to_str else None

    rows = load_metadata_rows(COMBINED_METADATA_PATH)
    filtered = filter_rows(rows, from_date, to_date, operator, phoropter)

    cols = export_metadata_columns()
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in filtered:
        writer.writerow(row)

    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions_export.csv"},
    )


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    app.run(host=host, port=port, debug=debug)
