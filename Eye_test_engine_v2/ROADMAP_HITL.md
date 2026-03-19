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
- [x] Export HITL annotations as Whisper training format (audio + transcript pairs)
- [x] Fine-tuning script: `python -m voice.training.whisper_finetune`
- [x] Auto-retrain cron: Monday 3am, if >50 new annotations since last train
- [x] Model versioning (v1, v2, ...) stored in `voice/models/whisper-finetuned/`
- [x] CTranslate2 conversion for faster-whisper compatibility
- [ ] *Requires training deps:* `pip install datasets evaluate jiwer soundfile`

### Intent Classifier (Fallback)
- [x] Train audio → intent classifier: `python -m voice.training.intent_classifier --train`
- [x] Runs as fallback when fuzzy matcher confidence < threshold
- [x] Architecture: audio features (energy + ZCR + spectral centroid) → 2-layer MLP
- [x] Integrated into voice pipeline — loads lazily if model exists
- [x] Evaluate: `python -m voice.training.intent_classifier --eval`

### A/B Testing Framework
- [x] Run old model vs new model on same audio: `python -m voice.training.ab_testing`
- [x] Track metrics: accuracy, review rate, avg inference time
- [x] Detailed disagreement report with per-utterance comparison
- [x] Auto-promotion recommendation based on accuracy + review rate
- [x] Results saved to `~/.eye_test_audio/_analysis/ab_test_results.json`

### Confidence Threshold Optimizer
- [x] Analyze HITL annotations: `python -m voice.training.confidence_optimizer`
- [x] Per-response-type thresholds (sweeps 40-100% in 5% steps)
- [x] Report: precision, recall, F1, false positive/negative at each threshold
- [x] Results saved to `~/.eye_test_audio/_analysis/confidence_analysis.json`

### Fuzzy Matcher Auto-Expansion
- [x] Extract new keyword aliases from HITL corrections: `python -m voice.training.matcher_expansion`
- [x] Auto-suggest additions to KEYWORD_MAP grouped by response_type
- [x] Generate code patch for manual review
- [x] Results saved to `~/.eye_test_audio/_analysis/matcher_suggestions.json`

### Weekly Retrain Orchestrator
- [x] `python -m voice.training.weekly_retrain` — runs all steps in sequence
- [x] Checks if enough new data (>50 annotations) before training
- [x] Runs: fine-tune → A/B test → confidence optimizer → matcher expansion
- [x] Logs all runs to `~/.eye_test_audio/_analysis/retrain_log.jsonl`
- [ ] Cron setup: `0 3 * * 1 cd /path/to && venv/bin/python -m voice.training.weekly_retrain`

### Central Server Sync
- [x] Push audio data: `python -m voice.training.server_sync --push`
- [x] Pull retrained models: `python -m voice.training.server_sync --pull`
- [x] Supports rsync (SYNC_RSYNC_TARGET) and HTTP API (SYNC_SERVER_URL)
- [x] Tracks sync state, only pushes new files
- [x] Show status: `python -m voice.training.server_sync --status`
- [ ] Central server API implementation (receiver side)
- [ ] Cron setup: `0 2 * * * push` / `0 4 * * * pull`

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
