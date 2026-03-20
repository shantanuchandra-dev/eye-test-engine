# Low-Level Architecture — Eye Test Engine v2

## 1. Codebase Structure

```
Eye_test_engine_v2/                     # 10,800+ lines
├── api_server.py                       # Flask API (30+ endpoints, 560 lines)
├── session_orchestrator.py             # FSM ↔ phoropter bridge (870 lines)
├── run.py                              # Entry point (dual server startup)
│
├── fsm/                                # Finite State Machine (2,500+ lines)
│   ├── refraction_engine.py            # Core FSM: states, questions, transitions (787 lines)
│   ├── delta_calculators.py            # Response → power change math
│   ├── state_transitions.py            # State routing rules
│   ├── escalation_rules.py             # Safety guardrails
│   ├── derived_variables.py            # Patient-derived parameters (dataclass)
│   ├── derived_variables_engine.py     # Calculates DVs from intake
│   ├── patient.py                      # PatientInput dataclass
│   ├── prescription.py                 # EyePrescription dataclass
│   ├── calibration.py                  # Device calibration loader
│   ├── fsm_runtime.py                  # FSMRuntimeRow (per-step state)
│   └── chart_scale.py                  # Chart progression logic
│
├── voice/                              # Voice pipeline (3,500+ lines)
│   ├── pipeline.py                     # Main pipeline: VAD → STT → match → TTS (953 lines)
│   ├── fuzzy_matcher.py                # Transcript → FSM option mapping (190 lines)
│   ├── ws_server.py                    # FastAPI WebSocket server (140 lines)
│   ├── audio_recorder.py               # FLAC recording + JSONL manifest (290 lines)
│   ├── review_api.py                   # HITL review Flask blueprint (250 lines)
│   ├── regional_languages.py           # 10-language translations + keywords (380 lines)
│   ├── download_models.py              # Model download script
│   ├── models/                         # Local model storage (git-ignored)
│   │   ├── whisper-small/              # faster-whisper STT (~460MB)
│   │   ├── piper/                      # Piper TTS voices (5 × ~60MB)
│   │   └── silero/                     # Silero VAD (~34MB)
│   └── training/                       # ML training pipeline
│       ├── whisper_finetune.py          # Whisper fine-tuning (240 lines)
│       ├── ab_testing.py                # A/B model comparison (220 lines)
│       ├── confidence_optimizer.py      # Threshold sweep analysis (180 lines)
│       ├── matcher_expansion.py         # Auto-expand fuzzy keywords (150 lines)
│       ├── intent_classifier.py         # Audio → intent MLP (320 lines)
│       ├── weekly_retrain.py            # Cron orchestrator (130 lines)
│       ├── server_sync.py              # Multi-clinic rsync/HTTP (200 lines)
│       ├── central_server.py           # Central aggregation server (170 lines)
│       └── setup_cron.sh               # Cron installation helper
│
├── frontend/                           # Browser UI (4,000+ lines)
│   ├── index.html                      # Main test UI (950 lines)
│   ├── app.js                          # Test logic + voice client (2,120 lines)
│   ├── intake.html                     # Patient intake form (800 lines)
│   ├── dashboard.html                  # Analytics dashboard
│   └── review.html                     # HITL review tool (450 lines)
│
├── io/                                 # Data I/O
│   ├── outputs.py                      # CSV/JSON session logging
│   ├── remote_storage.py               # Supabase upload
│   └── dashboard_data.py               # Dashboard queries
│
├── config/
│   └── calibration.csv                 # Phoropter calibration data
│
├── docs/                               # Documentation
├── tests/                              # Test suite
└── logs/sessions/                      # Session data (local)
```

## 2. API Endpoints

### Flask API (:5050)

#### Session Management
| Method | Endpoint | Handler | Description |
|--------|----------|---------|-------------|
| POST | `/api/session/intake` | `session_intake()` | Create session from intake data |
| POST | `/api/session/{id}/respond` | `respond()` | Submit patient response |
| GET | `/api/session/{id}/status` | `get_status()` | Get current FSM state |
| POST | `/api/session/{id}/jump` | `jump_to_phase()` | Jump to specific phase |
| POST | `/api/session/{id}/end` | `end_session()` | End and save session |
| POST | `/api/session/{id}/discard` | `discard_session()` | Discard session |
| POST | `/api/session/{id}/sync-power` | `sync_power()` | Sync manual power changes |
| POST | `/api/session/{id}/send-power` | `send_power()` | Re-send phoropter power |
| GET | `/api/session/{id}/derived-variables` | `get_derived_variables()` | Debug panel data |

#### Device Control (proxied to TOPCON broker)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/devices` | List available phoropters |
| GET | `/api/devices/{id}` | Get device details |
| POST | `/api/devices/{id}/acquire` | Lock device to operator |
| POST | `/api/devices/{id}/release` | Release device lock |
| POST | `/api/devices/{id}/heartbeat` | Keep lock alive |
| GET | `/api/brains` | List brain controllers |
| POST | `/api/phoropter/{id}/sync-state` | Sync power to device |
| POST | `/api/phoropter/{id}/reset` | Reset to 0/0/180 |
| POST | `/api/phoropter/{id}/pinhole` | Set pinhole |
| POST | `/api/phoropter/{id}/screenshot` | Capture live view |

#### HITL Review API
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/review/auth/register` | No | Self-registration |
| POST | `/api/review/auth/login` | No | Get auth token |
| GET | `/api/review/utterances` | Token | List with filters |
| PUT | `/api/review/utterances/{sid}/{uid}` | Token | Annotate/correct |
| POST | `/api/review/bulk-approve` | Token | Bulk approve |
| GET | `/api/review/stats` | Token | Review statistics |
| GET | `/api/review/export` | Token | Export training dataset |
| GET | `/api/review/audio/{path}` | Token | Serve audio file |
| GET | `/api/review/dates` | Token | Available dates |
| GET | `/api/review/sessions` | Token | Sessions per date |

#### Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/config` | Dashboard settings |
| PUT | `/api/dashboard/config` | Update settings |
| GET | `/api/dashboard/stats` | Session statistics |
| GET | `/api/dashboard/rr` | Refraction rate aggregates |
| GET | `/api/dashboard/export` | Export metadata CSV |

### Voice WebSocket (:8766)
| Endpoint | Params |
|----------|--------|
| `ws://host:8766/ws/voice/{session_id}?lang=en&voice=en_US-kusal-medium` | `lang`: en/hi/te/ta/kn/mr/ml/gu/bn/pa, `voice`: Piper name or `meta-mms-*` |

## 3. WebSocket Message Protocol

### Browser → Server
| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary (raw int16 PCM, 16kHz mono) | Mic audio chunks (~33ms intervals) |
| Stop | `{"type": "stop"}` | Close voice connection |

### Server → Browser
| Type | Format | Description |
|------|--------|-------------|
| `state_update` | `{"type":"state_update","data":{...}}` | FSM state change (same as `/respond` API response) |
| `voice_ready` | `{"type":"voice_ready","tts_sample_rate":22050}` | Pipeline ready for audio |
| `tts_start` | `{"type":"tts_start","text":"..."}` | TTS synthesis starting |
| `tts_end` | `{"type":"tts_end"}` | TTS synthesis complete |
| Audio | Binary: `0x01` + int16 PCM | TTS audio for playback |
| `vad` | `{"type":"vad","speaking":true/false}` | Voice activity detection |
| `transcript` | `{"type":"transcript","text":"..."}` | Whisper transcription result |
| `match` | `{"type":"match","option":"READABLE","confidence":95.0,"transcript":"clear"}` | Fuzzy match success |
| `no_match` | `{"type":"no_match","transcript":"mumble"}` | Fuzzy match failure |
| `test_complete` | `{"type":"test_complete"}` | Test ended (after TTS speaks completion) |
| `exit_confirmed` | `{"type":"exit_confirmed"}` | Patient requested stop + confirmed |
| `loading` | `{"type":"loading","message":"..."}` | Models loading |
| `error` | `{"type":"error","message":"..."}` | Error condition |

## 4. FSM State Machine

### State Definitions
| State | Phase | Type | Eye | Response Type | Options |
|-------|-------|------|-----|---------------|---------|
| B | Coarse Sphere | COARSE_SPHERE | RE | READABILITY | READABLE, NOT_READABLE, BLURRY |
| D | Coarse Sphere | COARSE_SPHERE | LE | READABILITY | READABLE, NOT_READABLE, BLURRY |
| E | JCC Axis | JCC_AXIS | RE | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| F | JCC Power | JCC_POWER | RE | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| G | Duochrome | DUOCHROME | RE | COLOR_CHOICE | RED_CLEARER, GREEN_CLEARER, EQUAL, CANT_TELL |
| H | JCC Axis | JCC_AXIS | LE | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| I | JCC Power | JCC_POWER | LE | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| J | Duochrome | DUOCHROME | LE | COLOR_CHOICE | RED_CLEARER, GREEN_CLEARER, EQUAL, CANT_TELL |
| K | Binocular Balance | BINOC_BALANCE | BIN | TOP_BOTTOM | TOP_CLEARER, BOTTOM_CLEARER, SAME, CANT_TELL |
| P | Near Add | NEAR_ADD_RE | RE | NEAR_READABILITY | READABLE, NOT_READABLE, BLURRY |
| Q | Near Add | NEAR_ADD_LE | LE | NEAR_READABILITY | READABLE, NOT_READABLE, BLURRY |
| R | Near Binocular | NEAR_BINOC | BIN | NEAR_BINOC | TARGET_OK, NOT_CLEAR |

### State Transition Graph
```
START → B
B → B (chart progression) | B → D (converged)
D → D (chart progression) | D → E (converged)
E ↔ E (JCC axis flip cycle) | E → F (axis converged)
F ↔ F (JCC power flip cycle) | F → G (power converged)
G ↔ G (duochrome iterations) | G → H (duochrome done)
H ↔ H (JCC axis flip cycle) | H → I (axis converged)
I ↔ I (JCC power flip cycle) | I → J (power converged)
J ↔ J (duochrome iterations) | J → K (duochrome done)
K → P (near test required) | K → END (no near test)
P → Q | Q → R | R → END

Any state → ESCALATE (safety guardrail triggered)
```

### JCC Flip Cycle (States E, F, H, I)
```
Enter JCC state
  ↓
Flip 1 displayed (2s countdown, no response expected)
  ↓ AUTO_FLIP
Handle sent to phoropter → Flip 2 displayed
  ↓
Patient responds (BETTER_1 / BETTER_2 / SAME / CANT_TELL)
  ↓
FSM adjusts power → back to Flip 1 (same state) or transition to next state
```

## 5. Voice Pipeline Internals

### VoicePipeline Class
```python
class VoicePipeline:
    # Components
    _vad_model: SileroVAD        # 512-sample chunks, threshold 0.6, RMS gate 0.01
    _whisper: WhisperModel        # Pre-loaded at startup, shared across sessions
    _recorder: AudioRecorder      # FLAC + JSONL per utterance
    _intent_classifier: MLP       # Fallback when fuzzy match fails (if model exists)

    # State
    _vad_speaking: bool           # Debounced: 3 chunks to start, 8 to stop
    _speech_buffer: np.ndarray    # Accumulated audio while speaking
    _silence_timer: asyncio.Task  # 3s timeout → rephrase question
    _awaiting_exit_confirm: bool  # Waiting for yes/no after exit keyword
    _prev_state: str              # Track for follow-up vs full question
    _pending_flip2_msg: str       # Paired flip terminology

    # Audio flow
    async process_audio(audio_int16: bytes)
        → _vad_buffer accumulate
        → Silero VAD per 512 samples (with RMS gate < 0.01 skip)
        → Debounced speaking start/stop
        → On stop: _transcribe_and_process()

    async _transcribe_and_process()
        → Whisper STT (in thread)
        → Check exit keywords → confirm flow
        → Fuzzy match (keyword + rapidfuzz)
        → Record utterance (FLAC + manifest)
        → If match: process_response → speak next question
        → If no match: try intent classifier fallback
        → If still no match: "please repeat"
```

### DirectTTSProcessor (Piper)
```
speak(text) → asyncio.to_thread(_synthesize)
    → PiperVoice.synthesize(text)
    → audio_int16_bytes chunks
    → WebSocket binary: 0x01 + chunk
    → tts_start/tts_end JSON messages
```

### MetaTTSProcessor (Meta MMS)
```
speak(text) → asyncio.to_thread(_synthesize)
    → VitsModel(**tokenizer(text))
    → waveform float32 → int16
    → WebSocket binary: 0x01 + all bytes
    → tts_start/tts_end JSON messages
```

### Fuzzy Matcher
```
match_transcript(transcript, response_type, threshold=60)
    Pass 1: Exact substring match against KEYWORD_MAP
            → longest matching phrase wins (100% confidence)
    Pass 2: rapidfuzz.fuzz.partial_ratio against all aliases
            → best score above threshold wins

KEYWORD_MAP: per response_type, per option
    English + romanized Hindi + Devanagari Hindi keywords
    Regional language keywords in REGIONAL_KEYWORDS
```

## 6. Audio Recording Schema

### Directory Structure
```
~/.eye_test_audio/
    .reviewers.json              # HITL user credentials
    .sync_state.json             # Last sync timestamps
    _analysis/                   # Training pipeline outputs
        confidence_analysis.json
        matcher_suggestions.json
        ab_test_results.json
        retrain_log.jsonl
    _exports/                    # Training dataset exports
    2026-03-20/
        session_xxx/
            manifest.jsonl       # One JSON line per utterance
            utt_0001.flac
            utt_0002.flac
            session_summary.json
```

### Utterance Manifest Entry
```json
{
    "id": "utt_0001",
    "session_id": "session_1773943129773",
    "timestamp": "2026-03-20T02:48:40.123",
    "audio_file": "utt_0001.flac",
    "duration_sec": 1.8,
    "sample_rate": 16000,
    "transcript_whisper": "the first one",
    "response_type": "COMPARE_1_2",
    "matched_option": "BETTER_1",
    "confidence": 100.0,
    "was_understood": true,
    "needs_review": false,
    "fsm_state": "E",
    "phase_name": "JCC Axis RE",
    "lang": "en",
    "mic_device": "MacBook Pro Microphone",
    "ambient_rms": 0.0234,
    "reviewed": false,
    "reviewed_by": null,
    "reviewed_at": null,
    "correct_option": null,
    "review_notes": null,
    "is_garbage": false
}
```

## 7. Phoropter Command Protocol

All commands sent as HTTP POST to `{PHOROPTER_BASE_URL}/phoropter/{id}/run-tests`:

### Power Command
```json
{
    "test_cases": [{
        "case_id": 1,
        "prev_aux_lens": "AuxLensL",
        "prev_right_eye": {"sph": -2.0, "cyl": -0.5, "axis": 180},
        "prev_left_eye": {"sph": -1.75, "cyl": -0.25, "axis": 175},
        "aux_lens": "AuxLensL",
        "right_eye": {"sph": -2.25, "cyl": -0.5, "axis": 180},
        "left_eye": {"sph": -1.75, "cyl": -0.25, "axis": 175}
    }]
}
```

### Chart Command
```json
{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_15"]}}]}
```

### JCC Command
```json
{"test_cases": [{"jcc": "handle"}]}        // flip
{"test_cases": [{"jcc": "power_axis_switch"}]} // toggle axis/power
{"test_cases": [{"jcc": "R"}]}             // right eye mode
```

### Occluder Mapping
| State | aux_lens | Occluded eye |
|-------|----------|-------------|
| B, E, F, G, P | AuxLensL | Left occluded → test right |
| D, H, I, J, Q | AuxLensR | Right occluded → test left |
| K, R | BINO | Both eyes open |

## 8. Training Pipeline

### Weekly Retrain Flow (`weekly_retrain.py`)
```
Monday 3am (cron)
    │
    ├─ Check: >50 new annotations since last train?
    │   ├─ No → skip training, run analysis only
    │   └─ Yes ↓
    │
    ├─ Step 1: Whisper Fine-Tuning
    │   ├─ Load reviewed (audio, transcript) pairs
    │   ├─ 90/10 train/eval split
    │   ├─ Seq2SeqTrainer, 3 epochs, WER metric
    │   ├─ Save HuggingFace model → convert to CTranslate2
    │   └─ Output: voice/models/whisper-finetuned/vN/
    │
    ├─ Step 2: A/B Test (new model vs current)
    │   ├─ Run both on all reviewed utterances
    │   ├─ Compare: accuracy, review rate, avg time
    │   └─ Recommend: promote or keep current
    │
    ├─ Step 3: Confidence Threshold Optimization
    │   ├─ Sweep 40-100% in 5% steps
    │   ├─ Per response_type analysis
    │   └─ Report: optimal threshold per type
    │
    └─ Step 4: Fuzzy Matcher Expansion
        ├─ Extract phrases from corrections
        ├─ Group by (response_type, option)
        └─ Generate code patch for review
```

### Intent Classifier Architecture
```
Audio file (FLAC/WAV)
    ↓
Feature extraction (no librosa dependency):
    - 100ms chunks
    - Per chunk: energy (RMS), zero crossing rate, spectral centroid
    - 50 chunks × 3 features = 150-dim vector
    ↓
MLP:
    Linear(150 → 64) → ReLU → Dropout(0.3)
    → Linear(64 → 32) → ReLU → Dropout(0.2)
    → Linear(32 → num_classes)
    ↓
Softmax → intent label + confidence
```

## 9. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5050` | Flask API port |
| `HOST` | `0.0.0.0` | Bind address |
| `VOICE_ENABLED` | `false` | Enable voice pipeline |
| `VOICE_PORT` | `8766` | Voice WebSocket port |
| `PHOROPTER_BASE_URL` | preprod URL | TOPCON broker |
| `REMOTE_STORAGE` | (none) | `supabase` for cloud logging |
| `FLASK_ENV` | (none) | `development` for debug |
| `SYNC_SERVER_URL` | (none) | Central server URL |
| `SYNC_RSYNC_TARGET` | (none) | rsync target for sync |
| `SYNC_CLINIC_ID` | `default_clinic` | Clinic identifier |
| `SYNC_API_KEY` | (none) | Central server API key |
| `CENTRAL_API_KEY` | `changeme` | Central server shared secret |
| `CENTRAL_DATA_DIR` | `~/eye_test_central/` | Central server data path |
