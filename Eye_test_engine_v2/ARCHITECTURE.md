# Architecture — Eye Test Engine v2

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser (Chrome)                         │
│                                                                  │
│  ┌────────────┐  ┌─────────────┐  ┌───────────────────────────┐ │
│  │ intake.html│→ │  index.html │  │      dashboard.html       │ │
│  │ (form)     │  │  + app.js   │  │      (analytics)          │ │
│  └────────────┘  └──────┬──────┘  └───────────────────────────┘ │
│                         │                                        │
│         HTTP REST ──────┤────── WebSocket (voice)                │
└─────────────────────────┼──────────────┼─────────────────────────┘
                          │              │
              ┌───────────▼──────┐  ┌────▼─────────────────────┐
              │  Flask API       │  │  FastAPI Voice Server     │
              │  :5050           │  │  :8766                    │
              │                  │  │                           │
              │  REST endpoints  │  │  WebSocket endpoint       │
              │  Session CRUD    │  │  /ws/voice/{session_id}   │
              └────────┬─────────┘  └────────┬──────────────────┘
                       │                     │
                       │    shared `sessions` dict (in-process)
                       │                     │
              ┌────────▼─────────────────────▼──────────────────┐
              │          SessionOrchestrator                     │
              │                                                  │
              │  ┌──────────────────┐  ┌──────────────────────┐ │
              │  │ FSMv3.0 Engine   │  │ Phoropter Commands   │ │
              │  │ (refraction_     │  │ (HTTP → TOPCON       │ │
              │  │  engine.py)      │  │  broker)             │ │
              │  └──────────────────┘  └──────────────────────┘ │
              └─────────────────────────────────────────────────┘
```

## Data Flow: Click-Based (Standard)

```
1. Patient intake form → POST /api/session/intake
2. SessionOrchestrator.initialize() → FSM row + phoropter commands
3. Frontend polls GET /api/session/{id}/status → displays question + buttons
4. Operator clicks response → POST /api/session/{id}/respond
5. SessionOrchestrator.process_response() →
   a. FSM applies response, calculates power delta
   b. Builds next FSM row (new question + options)
   c. Sends phoropter commands (chart + power + occluder)
6. Frontend renders next question → repeat from step 4
7. Terminal state → POST /api/session/{id}/end → saves CSV + JSON
```

## Data Flow: Voice Pipeline

```
1. Operator clicks "Voice" button in UI
2. Browser requests mic access, opens WebSocket to :8766
3. Browser streams raw PCM audio (16kHz, int16) over WebSocket

    ┌─────────────────── Pipecat Pipeline ───────────────────┐
    │                                                         │
    │  Audio In ─→ Silero VAD ─→ faster-whisper STT          │
    │                              │                          │
    │                              ▼                          │
    │                        FuzzyMatchProcessor               │
    │                        (transcript → FSM option)         │
    │                              │                          │
    │                              ▼                          │
    │                   SessionOrchestrator                    │
    │                   .process_response()                    │
    │                              │                          │
    │                              ▼                          │
    │                        Piper TTS                         │
    │                        (question → audio)                │
    │                              │                          │
    │                              ▼                          │
    │                      AudioSerializer                     │
    │                      (PCM → WebSocket binary)            │
    └─────────────────────────────────────────────────────────┘

4. Browser receives:
   - Binary (0x01 + PCM): plays TTS audio through speaker
   - JSON text: state updates → syncs UI (same as click path)
5. Click-based UI stays in sync — both input paths coexist
```

## Voice Pipeline Components

### Models (all local, stored in `voice/models/`)

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| VAD | Silero VAD | ~34MB | Detects when patient starts/stops speaking |
| STT | faster-whisper `small` | ~460MB | Transcribes speech to text (supports Hindi + English) |
| TTS | Piper `en_US-lessac-medium` | ~60MB | English questions |
| TTS | Piper `hi_IN-pratham-medium` | ~60MB | Hindi questions |

**Pipecat 0.0.106 note:** `SileroVADAnalyzer` is not a `FrameProcessor`. We wrap it
in a custom `VADFrameProcessor` that emits `VADUserStarted/StoppedSpeakingFrame` so
`WhisperSTTService` (a `SegmentedSTTService`) knows speech boundaries.

### Fuzzy Matching (`voice/fuzzy_matcher.py`)

The FSM has only 3-5 valid options per question. The matcher uses two passes:

1. **Exact keyword match** — substring search against a curated alias list per response type
2. **Fuzzy fallback** — `rapidfuzz.fuzz.partial_ratio` against all aliases, threshold 60%

If no match exceeds the threshold, TTS asks the patient to repeat.

Keyword maps include basic Hindi terms (e.g., "pehla" → BETTER_1, "upar" → TOP_CLEARER).

### JCC Auto-Flip

In JCC states (E/F/H/I), the voice pipeline owns the flip timer:
- Flip 1 displays for 2 seconds (no response expected)
- Pipeline sends AUTO_FLIP after timeout
- Flip 2 appears, patient responds verbally
- When voice is active, the frontend's JS timer is bypassed

### Thread Safety

The Flask API and Pipecat pipeline both call `SessionOrchestrator.process_response()`.
Both run in the same process — Flask in a thread, Pipecat on asyncio. Concurrent
voice + click responses to the same session could race. In practice this is unlikely
(one operator per session), but production hardening should add per-session locking.

## Deployment Modes

| Mode | Command | What runs |
|------|---------|-----------|
| Standard | `python run.py` | Flask only (:5050) |
| Voice | `VOICE_ENABLED=true python run.py` | Flask (:5050) + FastAPI (:8766) |
| Vercel | Deployed via `vercel.json` | Flask only (serverless, no voice) |

Voice mode requires local execution (not serverless) due to model size and mic access.
