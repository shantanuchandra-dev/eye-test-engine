# Eye Test Engine v2

Automated refraction engine for TOPCON phoropters. Runs a finite state machine (FSMv3.0) that walks a patient through a complete eye test — coarse sphere, JCC axis/power, duochrome, binocular balance, and near add — while controlling the phoropter hardware in real time.

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run (Flask API on port 5050)
python run.py
```

Open `http://localhost:5050` in Chrome. Fill in the intake form, then follow the on-screen questions.

## Voice Mode (Optional)

Adds hands-free operation: the engine speaks questions aloud and listens to patient responses via microphone.

### Setup

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Install all dependencies (includes voice packages)
pip install -r requirements.txt

# 3. Install additional voice deps
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pathvalidate

# 4. Download models locally (~620MB total, one-time)
python -m voice.download_models
```

Models are stored in `voice/models/` (git-ignored):

| Model | Size | Path |
|-------|------|------|
| faster-whisper `small` (STT) | ~460MB | `voice/models/whisper-small/` |
| Piper TTS voices (English + Hindi) | ~120MB | `voice/models/piper/` |
| Silero VAD | ~34MB | `voice/models/silero/` |

Available TTS voices:
- `en_US-lessac-medium` — English (default)
- `hi_IN-pratham-medium` — Hindi

### Running with Voice

```bash
VOICE_ENABLED=true python run.py
```

This starts two servers:
- **Flask API** on `:5050` (existing — unchanged)
- **Voice WebSocket** on `:8766` (new — Pipecat pipeline)

In the test UI, click the **Voice** button in the top bar to activate the microphone. Both click and voice inputs work simultaneously.

For Hindi TTS, connect with `?lang=hi`: `ws://localhost:8766/ws/voice/{session_id}?lang=hi`

### How It Works

```
Patient speaks → mic → browser captures PCM audio
    → WebSocket (:8766) → Silero VAD (speech boundary detection)
    → faster-whisper (local STT) → transcript
    → fuzzy matcher (keyword + rapidfuzz → FSM option)
    → SessionOrchestrator.process_response()
    → next question text → Piper TTS (local)
    → audio → WebSocket → browser plays through speaker
```

The fuzzy matcher maps natural speech to the 3-5 constrained FSM options per question. Supports English and basic Hindi keywords. Confidence threshold at 60% — below that, asks the patient to repeat.

## Project Structure

```
Eye_test_engine_v2/
├── api_server.py              # Flask REST API (port 5050)
├── session_orchestrator.py    # Wraps FSM + phoropter hardware control
├── run.py                     # Entry point (Flask only, or Flask + Voice)
├── requirements.txt
│
├── fsm/                       # FSMv3.0 — Finite State Machine
│   ├── refraction_engine.py   # Core engine: states, questions, transitions
│   ├── delta_calculators.py   # Response → power change logic
│   ├── state_transitions.py   # State routing rules
│   ├── escalation_rules.py    # Safety guardrails
│   ├── derived_variables.py   # Patient-derived parameters
│   ├── derived_variables_engine.py
│   ├── patient.py             # Patient data model
│   ├── prescription.py        # Rx data model
│   ├── calibration.py         # Phoropter calibration loader
│   ├── fsm_runtime.py         # Runtime row structure
│   └── chart_scale.py         # Chart progression
│
├── voice/                     # Voice pipeline (optional)
│   ├── pipeline.py            # Pipecat pipeline: VAD → STT → match → TTS
│   ├── fuzzy_matcher.py       # Transcript → FSM option mapping
│   ├── ws_server.py           # FastAPI WebSocket server (:8766)
│   ├── download_models.py     # One-time model download script
│   └── models/                # Local model storage (git-ignored)
│       ├── whisper-small/     # faster-whisper STT model
│       ├── piper/             # Piper TTS voice files
│       └── silero/            # Silero VAD model
│
├── frontend/                  # Vanilla JS + HTML
│   ├── index.html             # Main test UI
│   ├── intake.html            # Patient intake form
│   ├── dashboard.html         # Analytics dashboard
│   └── app.js                 # Frontend logic + voice client
│
├── io/                        # Data I/O
│   ├── outputs.py             # CSV/JSON session logging
│   ├── remote_storage.py      # Supabase upload (optional)
│   └── dashboard_data.py      # Dashboard queries
│
├── config/
│   └── calibration.csv        # Phoropter calibration data
│
├── logs/sessions/             # Session data (local)
└── tests/                     # Test suite
```

## FSM States (v3.0)

| State | Phase | Eye | Response Type |
|-------|-------|-----|---------------|
| B | Coarse Sphere | RE | READABILITY |
| D | Coarse Sphere | LE | READABILITY |
| E | JCC Axis | RE | COMPARE_1_2 |
| F | JCC Power | RE | COMPARE_1_2 |
| G | Duochrome | RE | COLOR_CHOICE |
| H | JCC Axis | LE | COMPARE_1_2 |
| I | JCC Power | LE | COMPARE_1_2 |
| J | Duochrome | LE | COLOR_CHOICE |
| K | Binocular Balance | BIN | TOP_BOTTOM |
| P | Near Add | RE | NEAR_READABILITY |
| Q | Near Add | LE | NEAR_READABILITY |
| R | Near Binocular | BIN | NEAR_BINOC |

## API Endpoints

See [curl_API.md](curl_API.md) for phoropter device control.

### Session

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/session/intake` | Start session with patient data |
| POST | `/api/session/{id}/respond` | Submit patient response |
| GET | `/api/session/{id}/status` | Get current state/question |
| POST | `/api/session/{id}/jump` | Jump to specific FSM phase |
| POST | `/api/session/{id}/end` | End and save session |
| POST | `/api/session/{id}/discard` | Discard session |

### Device

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/devices` | List available phoropters |
| POST | `/api/devices/{id}/acquire` | Lock device to operator |
| POST | `/api/devices/{id}/release` | Release device |
| POST | `/api/phoropter/{id}/reset` | Reset to 0/0/180 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5050` | Flask API port |
| `HOST` | `0.0.0.0` | Bind address |
| `VOICE_ENABLED` | `false` | Enable voice pipeline |
| `VOICE_PORT` | `8766` | Voice WebSocket port |
| `REMOTE_STORAGE` | (none) | Set to `supabase` for remote logging |
| `PHOROPTER_BASE_URL` | preprod URL | Phoropter broker base URL |
