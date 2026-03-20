# High-Level Architecture — Eye Test Engine v2

## System Overview

The Eye Test Engine is an AI-powered automated refraction system that conducts eye tests by controlling a TOPCON phoropter, presenting visual stimuli, and collecting patient responses — via button clicks or voice.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BROWSER (Chrome)                              │
│                                                                            │
│  ┌──────────┐  ┌─────────────┐  ┌───────────┐  ┌─────────────────────────┐│
│  │ Intake   │→ │ Test UI     │  │ Dashboard │  │ HITL Review            ││
│  │ Form     │  │ (index.html)│  │           │  │ (review.html)          ││
│  └──────────┘  └──────┬──────┘  └───────────┘  └─────────────────────────┘│
│                  HTTP ─┤─ WebSocket                                        │
└────────────────────────┼──────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────────────┐
          │              │                      │
┌─────────▼────────┐ ┌──▼───────────────┐ ┌────▼──────────────────┐
│   Flask API      │ │  Voice Server    │ │  TOPCON Phoropter     │
│   :5050          │ │  :8766 (FastAPI) │ │  (Broker API)         │
│                  │ │                  │ │                        │
│  Session CRUD    │ │  WebSocket       │ │  Chart display         │
│  Device control  │ │  Audio streaming │ │  Power adjustment      │
│  Dashboard       │ │  TTS playback    │ │  Occluder switching    │
│  Review API      │ │                  │ │  JCC control           │
└────────┬─────────┘ └────────┬─────────┘ └────────────────────────┘
         │                    │
         │   shared sessions  │
         │   dict (in-process)│
         │                    │
┌────────▼────────────────────▼──────────────────────────────────────┐
│                    SESSION ORCHESTRATOR                             │
│                                                                    │
│  ┌──────────────────────┐   ┌────────────────────────────────┐    │
│  │  FSM v3.0 Engine     │   │  Phoropter Command Builder     │    │
│  │  (12 states, 6 types)│   │  (chart, power, occluder, JCC) │    │
│  └──────────────────────┘   └────────────────────────────────┘    │
│                                                                    │
│  ┌──────────────────────┐   ┌────────────────────────────────┐    │
│  │  Derived Variables   │   │  Calibration Engine             │    │
│  │  (patient → params)  │   │  (device-specific settings)    │    │
│  └──────────────────────┘   └────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

## Three Pillars

### 1. Refraction Engine (FSM)
The core test logic. A finite state machine walks through 12 phases of an eye test:

```
B (Coarse Sphere RE) → D (Coarse Sphere LE)
    → E (JCC Axis RE) → F (JCC Power RE) → G (Duochrome RE)
    → H (JCC Axis LE) → I (JCC Power LE) → J (Duochrome LE)
    → K (Binocular Balance)
    → P (Near Add RE) → Q (Near Add LE) → R (Near Binocular)
    → END
```

Each state generates a question, presents response options, and adjusts lens power based on the patient's answer. The FSM handles fogging, escalation, chart progression, and convergence detection.

### 2. Voice Pipeline
Hands-free operation: the engine speaks questions aloud and listens to patient responses.

```
Patient speaks → Mic (16kHz) → Silero VAD → faster-whisper STT
    → Fuzzy Matcher → FSM intent → Phoropter commands
    → Next question → Piper/Meta MMS TTS → Speaker

Supports 10 languages: EN, HI, TE, TA, ML, KN, MR, GU, BN, PA
12 TTS voice options in the UI dropdown
```

### 3. HITL Training Pipeline
Continuous improvement loop for the voice system:

```
Audio Recordings → HITL Review Tool → Annotated Dataset
    → Whisper Fine-Tuning → A/B Testing → Production Model
    → Confidence Optimizer → Fuzzy Matcher Expansion
```

## Data Flow Summary

| Data | Source | Storage | Format |
|------|--------|---------|--------|
| Patient intake | Intake form | In-memory session | JSON |
| Test responses | Click / Voice | Session history | SessionRow objects |
| Session logs | End of test | `logs/sessions/` | CSV + JSON |
| Combined logs | End of test | `logs/combined_*.csv` | CSV |
| Voice recordings | During test | `~/.eye_test_audio/` | FLAC + JSONL manifest |
| HITL annotations | Review tool | `~/.eye_test_audio/` (updated manifest) | JSONL |
| Trained models | Weekly retrain | `voice/models/whisper-finetuned/` | CTranslate2 |
| Dashboard stats | On-demand | Read from combined logs | JSON API |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla JS + HTML/CSS (no framework) |
| Backend API | Flask (Python) |
| Voice Server | FastAPI + WebSocket (Python) |
| FSM Engine | Pure Python |
| STT | faster-whisper (local, CPU) |
| TTS | Piper (local ONNX) + Meta MMS-TTS (local HuggingFace) |
| VAD | Silero VAD (local PyTorch) |
| Phoropter | HTTP API to TOPCON broker |
| Storage | Local filesystem + optional Supabase |
| Training | HuggingFace Transformers + PyTorch |

## Deployment Topology

```
                    CURRENT (Single Machine)
                    ========================
                    ┌──────────────────────┐
                    │  MacBook / Clinic PC  │
                    │                      │
                    │  Flask :5050         │
                    │  FastAPI :8766       │
                    │  ~/.eye_test_audio/  │
                    │  voice/models/       │
                    └──────────┬───────────┘
                               │
                               │ HTTP
                               ▼
                    ┌──────────────────────┐
                    │  TOPCON Phoropter    │
                    │  (via Broker API)    │
                    └──────────────────────┘


                    FUTURE (Multi-Clinic)
                    =====================
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Clinic 1 │  │ Clinic 2 │  │ Clinic N │
    │ :5050    │  │ :5050    │  │ :5050    │
    │ :8766    │  │ :8766    │  │ :8766    │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
         └──────┬──────┘─────────────┘
                │  daily sync (rsync/HTTP)
                ▼
    ┌────────────────────────┐
    │   Central Server       │
    │   :9000                │
    │                        │
    │   Aggregated audio     │
    │   HITL review (shared) │
    │   Weekly retraining    │
    │   Model distribution   │
    └────────────────────────┘
```

## Security & Privacy

- Patient audio stored in hidden directory (`~/.eye_test_audio/`)
- HITL review tool requires email/password authentication
- Central server sync uses API key authentication
- Patient consent checkbox required in intake form
- Audio stored forever in structured format for training
- No PHI transmitted to cloud services — all models run locally
