# HITL Audio Annotation & Retraining Roadmap

## Phase 1: Audio Recording + HITL Review Tool

### Audio Recording Pipeline
- [x] Record every patient utterance as audio file during voice sessions
- [x] Store in hidden directory `~/.eye_test_audio/` organized by date/session
- [x] Use FLAC format (lossless, ~40% of WAV size, training-friendly)
- [x] Generate JSONL manifest per session with full metadata
- [x] Tag unrecognized utterances (`needs_review: true`)
- [x] Capture: transcript, matched intent, confidence, FSM state, response_type
- [x] Capture: patient intake data (age, occupation, symptoms, AR/lenso)
- [x] Capture: mic device info, ambient noise level (RMS), voice language
- [x] Capture: reviewer assignments, timestamps

### HITL Review Web UI
- [x] Standalone page at `/review.html`
- [x] Login system (email/password, self-registration)
- [x] Utterance table with inline audio playback
- [x] Filters: date, needs_review, reviewed, reviewer, session, confidence
- [x] Correct-intent dropdown per utterance
- [x] Mark as garbage/noise
- [x] Bulk approve correctly-matched utterances
- [x] Review stats dashboard (total, reviewed, pending, accuracy)
- [x] Export annotations as training-ready dataset

### Review API
- [x] `GET /api/review/utterances` — list with filters
- [x] `PUT /api/review/utterances/{id}` — annotate/correct
- [x] `POST /api/review/bulk-approve` — bulk approve
- [x] `GET /api/review/stats` — review statistics
- [x] `GET /api/review/export` — export training dataset
- [x] `POST /api/review/auth/register` — self-registration
- [x] `POST /api/review/auth/login` — login
- [x] Audio file serving with auth

## Phase 2: Training & A/B Testing (Future)

### Whisper Fine-Tuning
- [ ] Export HITL annotations as Whisper training format (audio + transcript pairs)
- [ ] Fine-tuning script using `faster-whisper` or HuggingFace `transformers`
- [ ] Auto-retrain cron: Monday 3am, if >50 new annotations since last train
- [ ] Model versioning (v1, v2, ...) stored in `voice/models/whisper-finetuned/`

### Intent Classifier (Fallback)
- [ ] Train audio → intent classifier from HITL-labeled data
- [ ] Runs when fuzzy matcher confidence < threshold
- [ ] Architecture: small CNN or wav2vec2 fine-tuned on intent labels

### A/B Testing Framework
- [ ] Run old model vs new model on same audio
- [ ] Track metrics: accuracy, review rate, completion time
- [ ] Dashboard showing A/B comparison
- [ ] Auto-promote model if new version outperforms

### Confidence Threshold Optimizer
- [ ] Analyze HITL annotations to find optimal fuzzy match threshold
- [ ] Per-response-type thresholds (READABILITY may need different threshold than COMPARE_1_2)
- [ ] Report: false positive rate vs false negative rate at different thresholds

### Fuzzy Matcher Auto-Expansion
- [ ] Extract new keyword aliases from HITL corrections
- [ ] Auto-suggest additions to KEYWORD_MAP
- [ ] One-click apply to fuzzy_matcher.py

### Central Server Sync
- [ ] Clinic machines push audio + manifest to central server daily
- [ ] Central HITL tool aggregates all clinic data
- [ ] Retrained models pushed back to clinic machines

### Regional Language Expansion
- [ ] Tamil, Telugu, Kannada, Marathi voice support
- [ ] Per-language Whisper models
- [ ] Per-language fuzzy matcher keyword maps
- [ ] Per-language HITL annotation categories

## Storage Budget

| Scale | Format | Per Test | Daily (tests) | Daily Storage |
|-------|--------|----------|---------------|---------------|
| Pilot | FLAC | ~1.1MB | 10 | ~11MB |
| Growth | FLAC | ~1.1MB | 100 | ~110MB |
| Scale | Opus | ~0.36MB | 1000 | ~360MB |

Switch from FLAC to Opus when daily volume exceeds 500 tests.
