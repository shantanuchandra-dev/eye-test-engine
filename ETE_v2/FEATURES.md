# ETE v2 — Feature List

## FSM Engine (FSMv3.1)

### Refraction Phases
- **Coarse Sphere** (RE then LE) — Starts from the third-smallest chart (`70_60_50`), shows only the active working line, asks the patient to read that line, adds minus on blur, and uses a clearer-vs-more-blurry confirmation loop before ending the phase
- **JCC Axis Refinement** (RE then LE) — Adaptive lane-driven axis search with reversal-based step reduction
- **JCC Power Refinement** (RE then LE) — Cylinder power optimization with flip reversal tracking
- **Duochrome** (RE then LE) — Red/green balance test with configurable max flips and bias policies
- **Distance VA Confirmation** (RE then LE) — After duochrome convergence, starts from `20_20_20`, steps to larger charts if needed, and records the last confirmed line read for each eye
- **Binocular Balance** — Equalize both eyes with OU comparison
- **Near Vision Workflow** — Monocular ADD (RE/LE) then binocular near refinement
- **Escalation** — Safety exit to optometrist review on anomaly detection

### Accommodation-Driven Fogging (v3.1)
- Fogging amount based on age/accommodation level (child → strong, adult → standard, presbyope → low fog)
- Configurable via calibration: `strong_fog_amount`, `standard_fog_amount`, `low_fog_amount`
- Applied once at coarse sphere entry as an initial plus shift; the coarse loop then refines using last-line readability and clearer-vs-more-blurry confirmation

### Derived Variables Engine
- Derived variables computed from patient intake data + calibration parameters
- Covers: age bucket, risk levels, mismatch detection, start Rx policy, adaptive axis lane policy, fogging policy, step sizes, tolerances, and phase timeouts
- Fully driven by `calibration.csv` — no hardcoded clinical values

### Safety Guardrails
- Max delta from start Rx (configurable per step policy)
- Max delta from AR sphere
- Phase timeout limits (per convergence speed)
- Automatic optometrist review flags (high medical risk, anomaly watch)
- Cylinder zero hard stop (v3.1)

---

## Input Methods

### Voice Input
- **Browser Speech Recognition** — Web Speech API with interim quick-matching for single digits
- **Faster-Whisper** — Backend STT with VAD (voice activity detection), configurable silence thresholds
- Server-side response matching with fuzzy/exact/letter-reading modes
- Retry logic with spoken prompts on failed matches
- Multi-language: English and Hindi

### Gamepad Input
- Xbox controller support via Chrome Gamepad API
- B/A/X → option 1/2/3, Y → REPEAT
- Configurable polling with debounce

### Keyboard Input
- Number keys 1-9 map to option buttons

### Input Method Tracking
- Every interaction tagged: `Voice_Whisper`, `Voice_Browser`, `Gamepad`, `Keyboard`, `Button`, `Manual_Adjustment`
- Exported in session CSV (`Input_Method` column)

---

## Voice Data & Training Pipeline

### Per-Interaction Capture
- Raw transcript (what the patient said)
- Alternative transcripts (browser speech recognition alternatives)
- Match confidence score (0-1)
- Match method (fuzzy, exact, letter_reading, server_side)
- Canonical label matched
- Stimulus letters shown (for chart-reading states)

### Audio Blob Storage
- Whisper recordings saved as `.webm` files in `logs/sessions/audio/`
- Filename format: `{session_id}_step{N}_{timestamp}.webm`
- Correlated to session CSV by timestamp and step number

### Voice Utterance Training CSV
- Per-session file: `{session_id}_voice_utterances.csv`
- 16 columns: timestamp, session_id, step, state, phase_name, transcript, alternatives, intent_matched, canonical_label, confidence, match_method, input_method, language, stimulus_letters, audio_file, accepted
- Merges successful matches AND failed attempts chronologically
- Ready for STT/NLU model training

---

## Phoropter Integration

- Auto-dispatch: phoropter commands sent automatically on each FSM step
- Occluder control per phase (RE occlude LE, LE occlude RE, binocular)
- JCC flip automation (handle → flip1 → observation → flip2)
- Chart switching (distance/near, chart groups)
- Power sync endpoint for manual adjustments
- Screenshot capture after each command batch
- PIP (picture-in-picture) panel for live phoropter view

---

## Calibration System

### Calibration Categories
- SYSTEM, PATIENT_TO_DV, MISMATCH_THRESHOLDS, START_POLICY
- DISTANCE_TARGET, ENDPOINT_BIAS, STEP_POLICY, AXIS_POLICY
- CYL_POLICY, FOGGING, DUOCHROME, BINOC_BALANCE
- NEAR_WORKFLOW, ESCALATION, PHASE_TIMEOUTS, CHARTS

### Live Editor (`/cal`)
- Password-gated standalone page (dark instrument panel UI)
- All parameters grouped by section, searchable
- Inline value editing with modified-state highlighting
- Save writes directly to `calibration.csv`
- Changes effective on next session (no restart needed)
- 24-hour unlock persistence

### Calibration Snapshot in Metadata
- Full calibration state (section, key, value, unit_or_type) saved in each session's metadata JSON
- Enables post-hoc analysis of which calibration was active during any session

---

## Session Logging

### Per-Session Files
| File | Format | Content |
|------|--------|---------|
| `{id}.csv` | CSV | Step-by-step log: Rx values, input method, voice transcript, phase, chart, lane metadata, and VA confirmation state |
| `{id}_metadata.json` | JSON | Full session metadata: patient input, derived variables, calibration snapshot, quality metrics |
| `{id}_voice_utterances.csv` | CSV (16 cols) | Voice interactions for training (successful + failed) |
| `{id}_failed_voice_attempts.csv` | CSV | Failed STT attempts with alternatives |

### Metadata JSON Contents
- Session info: ID, phoropter, operator, customer, start/end times, duration
- AR and lensometry input values
- Final prescription (RE/LE with ADD) plus final confirmed distance VA for each eye
- Phase completion and skip lists
- Quality metrics: manual adjustments, QnA count, phase jumps, unable-to-read count, duration per phase
- Full patient input (32 fields from intake form)
- All derived variables
- Calibration snapshot with section, key, value, unit

### Combined Logs
- `combined_log.csv` — All sessions appended (step-level)
- `combined_metadata.csv` — All sessions flattened, including patient input, derived variables, and final distance VA fields

---

## Frontend

### Test Screen (`/`)
- Question card with chart display (letter charts, red-green)
- Option buttons with keyboard shortcut hints (1-9)
- Real-time Rx table (SPH/CYL/AXIS/ADD for RE and LE)
- Phase progress tracker with completion dots
- Blink/time-left motivation prompt after major milestones
- Fog active indicator
- Conversation log panel
- Voice status bar with STT feedback
- TTS (text-to-speech) with voice selection dropdown
- Phoropter screenshot PIP with zoom controls
- DV drawer with AR/Lenso powers, color-coded risk badges, collapsible sections
- Session logs drawer (password-gated: conversation, CURL, responses tabs)
- iPad responsive (sidebar as overlay via floating Rx button)

### Intake Form (`/intake`)
- Patient demographics (age, gender, occupation)
- Symptoms and visual complaints
- Medical history flags (diabetes, keratoconus, amblyopia, prior surgery)
- AR and lensometry values (with stepper inputs)
- Screen time, driving hours, comfort priority, near priority
- Satisfaction with current Rx
- Axis-lane validation presets for live testing
- iPad responsive (2-column grid, touch-friendly targets)

### Dashboard (`/dashboard`)
- Session count (today/total, per-phoropter)
- Export session data
- R&R (refraction result) summary

### Calibration Editor (`/cal`)
- Dark instrument panel aesthetic
- Password-gated (server-side validation)
- Search across calibration parameters
- Collapsible section cards
- Modified value highlighting (amber glow)
- Discard and save controls
- Version badge from active_profile

---

## Multi-Language Support
- English and Hindi for all patient-facing prompts
- Language selection at session start (voice or button)
- TTS speaks questions in selected language
- Voice recognition handles both languages

---

## Deployment
- Flask server on any Python 3.9+ environment
- Railway deployment supported
- Supabase remote storage optional (session logs, metadata)
- Environment variable configuration (no hardcoded secrets)
