# ETE v2 — Eye Test Engine

Automated refraction engine that drives a complete eye examination through a phoropter, using an FSM (Finite State Machine) to navigate sphere, cylinder, axis, duochrome, binocular balance, and near vision testing.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS)               │
│  index.html  │  intake.html  │  dashboard.html  │ cal │
└──────────┬───────────────────────────────────────────┘
           │  REST API (Flask)
┌──────────▼───────────────────────────────────────────┐
│               api_server.py                            │
│  Session CRUD │ Voice │ Phoropter Proxy │ Calibration  │
└──────────┬───────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────┐
│           session_orchestrator.py                       │
│  Drives FSM │ Records History │ Phoropter Dispatch      │
└──────────┬───────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────┐
│              FSM Engine Layer                           │
│  refraction_fsm_engine.py  │  state_transitions.py     │
│  delta_calculators.py      │  escalation_rules.py      │
│  derived_variables_engine.py │  chart_scale.py          │
└──────────────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────┐
│           Configuration                                 │
│  calibration.csv (115 params, 13 sections)              │
│  CalibrationLoader → DerivedVariablesEngine             │
└──────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd ETE_v2
pip install -r requirements.txt
python run.py
# Open http://localhost:5050
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CALIBRATION_PATH` | `config/calibration.csv` | Path to calibration parameters |
| `PHOROPTER_BASE_URL` | preprod broker URL | Phoropter API broker endpoint |
| `BACKEND_URL` | `http://localhost:5050` | Self URL for frontend config |
| `LOG_DIR` | `logs` | Session log output directory |
| `FLASK_ENV` | `development` | Flask run mode |

## FSM State Flow (FSMv3.1)

```
B: Coarse Sphere RE ──► E: JCC Axis RE ──► F: JCC Power RE ──► G: Duochrome RE
       │                                                              │
       ▼                                                              ▼
D: Coarse Sphere LE ──► H: JCC Axis LE ──► I: JCC Power LE ──► J: Duochrome LE
                                                                      │
                                                                      ▼
                                                               K: Binocular Balance
                                                                      │
                                                            ┌─────────┴─────────┐
                                                            ▼                   ▼
                                                    P/Q: Near Add         END: Complete
                                                    R: Binoc Near
                                                            │
                                                            ▼
                                                       END: Complete
```

## Project Structure

```
ETE_v2/
├── api_server.py              # Flask API (all endpoints)
├── session_orchestrator.py    # Session lifecycle + phoropter dispatch
├── voice_endpoint.py          # Faster-whisper STT integration
├── run.py                     # Entry point
│
├── fsm/                       # Core refraction logic
│   ├── engines/
│   │   ├── refraction_fsm_engine.py    # Main FSM engine
│   │   ├── state_transitions.py        # State transition rules
│   │   ├── delta_calculators.py        # Prescription adjustments
│   │   ├── escalation_rules.py         # Safety guardrails
│   │   └── derived_variables_engine.py # Patient → test config
│   ├── models/
│   │   ├── fsm_runtime.py      # FSMRuntimeRow (step state)
│   │   ├── derived_variables.py # 63-field test config
│   │   ├── patient.py          # PatientInput model
│   │   └── prescription.py     # EyePrescription model
│   ├── charts/
│   │   └── chart_scale.py      # VA chart ladder
│   ├── audio/
│   │   ├── local_stt.py        # Whisper STT wrapper
│   │   └── response_matching.py # Voice → intent matching
│   └── config/
│       ├── calibration_loader.py # CSV config reader + writer
│       └── calibration_schema.py # Schema validation
│
├── config/
│   └── calibration.csv         # 115 tunable parameters
│
├── ete_io/                     # I/O and logging
│   ├── outputs.py              # CSV/JSON log writers
│   ├── remote_storage.py       # Supabase upload
│   ├── dashboard_data.py       # Dashboard stats
│   └── ist_time.py             # IST timezone helper
│
└── frontend/                   # Vanilla JS UI
    ├── index.html              # Main test screen
    ├── intake.html             # Patient intake form
    ├── dashboard.html          # R&R dashboard
    ├── calibration.html        # Calibration editor (/cal)
    ├── app.js                  # Main test screen JS
    └── favicon.svg             # Eye icon favicon
```

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Test Session | Main eye test interface with questions, Rx table, phase progress |
| `/intake` | Patient Intake | Patient form → starts session |
| `/dashboard` | Dashboard | Session stats, R&R metrics, export |
| `/cal` | Calibration Editor | Password-gated editor for all 115 calibration parameters |

## Logging

Each session produces:
- `{session_id}.csv` — Per-step log (23 columns including input method and voice data)
- `{session_id}_metadata.json` — Full session metadata (calibration snapshot, patient input, derived variables)
- `{session_id}_voice_utterances.csv` — Voice interaction training data (16 columns)
- `{session_id}_failed_voice_attempts.csv` — Failed STT attempts
- `logs/sessions/audio/` — Saved audio blobs (.webm) from whisper transcriptions

Combined logs across sessions:
- `combined_log.csv` — All session steps appended
- `combined_metadata.csv` — All session metadata flattened (99 columns)
