# User Flow — Eye Test Engine v2

## Overview

Three user personas interact with the system:

| Persona | Interface | Purpose |
|---------|-----------|---------|
| **Operator** | Test UI (`index.html`) | Conducts eye tests with patients |
| **Patient** | Voice / Button responses | Answers questions during the test |
| **Reviewer** | HITL Review (`review.html`) | Annotates voice recordings for training |

---

## Flow 1: Conducting an Eye Test

### 1.1 Patient Intake

```
Operator opens intake.html
    │
    ├─ Enters patient demographics
    │   ├─ Name, Age, Gender
    │   ├─ Occupation, Screen time, Driving hours
    │   └─ Chief complaint, Symptoms
    │
    ├─ Enters clinical measurements
    │   ├─ Autorefractor (AR): SPH/CYL/AXIS for RE and LE
    │   └─ Lensometry (Lenso): SPH/CYL/AXIS/ADD for RE and LE
    │
    ├─ Selects phoropter device
    │
    └─ Clicks "Start Test"
         │
         └─ POST /api/session/intake
              │
              ├─ Creates SessionOrchestrator
              ├─ Derives patient variables (age bucket, risk, fogging, etc.)
              ├─ Initializes FSM at State B
              ├─ Sends initial power to phoropter
              └─ Redirects to index.html?session={id}
```

### 1.2 Test Execution (Click Mode)

```
Test UI loads
    │
    ├─ Acquires phoropter device (heartbeat every 15s)
    ├─ Resets phoropter to 0/0/180
    ├─ Fetches session status → displays first question
    │
    ▼
┌─────────────────────── QUESTION LOOP ─────────────────────────┐
│                                                                │
│  UI shows:                                                     │
│    ├─ Phase badge (e.g. "Phase B: Coarse Sphere RE")          │
│    ├─ Eye indicator (RIGHT EYE / LEFT EYE / BOTH EYES)        │
│    ├─ Question text                                            │
│    └─ Response buttons (3-4 options)                           │
│                                                                │
│  Operator clicks a button (or patient presses 1/2/3/4)        │
│    │                                                           │
│    └─ POST /api/session/{id}/respond                           │
│         │                                                      │
│         ├─ FSM applies response                                │
│         │   ├─ Calculates power delta (SPH/CYL/AXIS/ADD)      │
│         │   └─ Determines next state                           │
│         │                                                      │
│         ├─ Phoropter commands sent                             │
│         │   ├─ Chart update (Snellen / JCC / Duochrome / etc.) │
│         │   ├─ Power adjustment (delta from previous)          │
│         │   └─ Occluder switch (if eye changes)                │
│         │                                                      │
│         └─ Returns next question + options                     │
│              │                                                 │
│              └─ UI updates → back to top of loop               │
│                                                                │
│  Special: JCC Flip Cycle (States E/F/H/I)                     │
│    ├─ Flip 1: countdown 2s (no response buttons)               │
│    ├─ AUTO_FLIP sent automatically                             │
│    ├─ Flip 2: response buttons appear                          │
│    └─ Patient responds → power adjusted → next flip or exit    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
Test Complete (FSM → END)
    │
    ├─ Shows final prescription (RE + LE: SPH/CYL/AXIS/ADD)
    ├─ Phoropter shows 20/20/20 chart
    ├─ Operator clicks "Save & End" or "Discard"
    │   ├─ Save: writes CSV + JSON to logs/, uploads to Supabase
    │   └─ Discard: deletes session, no logs
    └─ Releases phoropter device
```

### 1.3 Test Execution (Voice Mode)

```
Test UI loads
    │
    ├─ Voice mode auto-activates (or operator clicks Voice button)
    ├─ Browser requests microphone permission
    ├─ WebSocket connects to ws://localhost:8766/ws/voice/{session_id}
    │
    ▼
Voice Pipeline Starts
    │
    ├─ Models loaded (Whisper pre-loaded at startup)
    ├─ Silero VAD initialized
    ├─ Piper/Meta TTS voice loaded for selected language
    │
    ├─ AI speaks first question through speaker
    │   e.g. "Looking at the letters, can you read them clearly?"
    │
    ▼
┌─────────────────── VOICE QUESTION LOOP ───────────────────────┐
│                                                                │
│  AI speaks question → Speaker plays audio                      │
│    │                                                           │
│    ├─ Speaking animation (blue icon + sweep on question text)  │
│    ├─ Voice chip shows "Ready" (bars static)                   │
│    │                                                           │
│    ▼                                                           │
│  Silence timer starts (3 seconds)                              │
│    │                                                           │
│    ├─ If 3s silence → AI rephrases with shorter question       │
│    │   e.g. "Can you read the letters?"                        │
│    │   (only rephrases once per question)                      │
│    │                                                           │
│    ▼                                                           │
│  Patient speaks into microphone                                │
│    │                                                           │
│    ├─ VAD detects speech start                                 │
│    │   ├─ Voice chip: bars animate, shows "Listening..."       │
│    │   └─ Audio accumulated in speech buffer                   │
│    │                                                           │
│    ├─ VAD detects speech end (8 consecutive silence chunks)    │
│    │   └─ Voice chip: bars stop                                │
│    │                                                           │
│    ▼                                                           │
│  Whisper STT transcribes speech                                │
│    │                                                           │
│    ├─ Voice log: "clear" (user bubble, right-aligned)          │
│    │                                                           │
│    ├─ Check exit keywords ("stop", "help", "रुको", etc.)      │
│    │   └─ If exit: "Do you want to stop? Say yes or no."      │
│    │       ├─ Yes → "Okay, stopping the test." → escalation   │
│    │       └─ No → "Okay, let's continue." → resume           │
│    │                                                           │
│    ▼                                                           │
│  Fuzzy Matcher maps transcript → FSM option                    │
│    │                                                           │
│    ├─ MATCH (confidence ≥ 60%)                                 │
│    │   ├─ Voice log: "clear" → ✓ READABLE (green bubble)      │
│    │   ├─ Voice chip: green "heard" state, 2s display          │
│    │   │                                                       │
│    │   ├─ FSM processes response                               │
│    │   │   ├─ Phoropter commands sent                          │
│    │   │   └─ Next question generated                          │
│    │   │                                                       │
│    │   ├─ Same state? → short follow-up (random variation)     │
│    │   │   e.g. "How about now?" / "And this line?"            │
│    │   │                                                       │
│    │   ├─ New state? → full question (translated if non-EN)    │
│    │   │   e.g. "Which view makes the dots look sharper?"      │
│    │   │                                                       │
│    │   └─ AI speaks next question → back to top of loop        │
│    │                                                           │
│    ├─ NO MATCH → try intent classifier fallback                │
│    │   ├─ If classifier confidence > 70% → treat as match      │
│    │   └─ If still no match:                                   │
│    │       ├─ Voice log: "mumble" → ? (red bubble)             │
│    │       ├─ Voice chip: red "no-match", 2s                   │
│    │       └─ AI says: "I didn't catch that. Please repeat."   │
│    │                                                           │
│    └─ Audio recorded as FLAC to ~/.eye_test_audio/             │
│        ├─ Manifest entry with full metadata                    │
│        └─ Flagged as needs_review if no match or low conf      │
│                                                                │
│  Special: JCC Flip Cycle (Voice)                               │
│    ├─ AI: "This is one." (random variation)                    │
│    ├─ 2s pause                                                 │
│    ├─ AI: "This is two. Which is better, one or two?"          │
│    ├─ Patient responds                                         │
│    ├─ Repeat cycle or transition                               │
│    └─ Follow-ups: "And this time?" / "Any difference?"         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
Test Complete
    │
    ├─ AI speaks: "The eye test is now complete. Thank you."
    ├─ Phoropter shows 20/20/20 chart
    ├─ Voice pipeline stops (mic released, WebSocket closed)
    ├─ Prescription displayed on screen
    └─ Operator clicks Save or Discard
```

### 1.4 Voice Language Selection

```
Operator selects language from dropdown (top bar)
    │
    ├─ English (Indian) → Piper kusal voice, English STT
    ├─ English (US) → Piper lessac voice, English STT
    ├─ Hindi (Piper) → Piper pratham voice, Hindi STT, Hindi questions
    ├─ Hindi (Meta) → Meta MMS Hindi, Hindi STT, Hindi questions
    ├─ Telugu → Piper venkatesh voice, Telugu STT, Telugu questions
    ├─ Tamil → Meta MMS Tamil, Tamil STT, Tamil questions
    ├─ Kannada → Meta MMS Kannada, Kannada STT, Kannada questions
    ├─ Marathi → Meta MMS Marathi, Marathi STT, Marathi questions
    ├─ Malayalam → Piper arjun voice, Malayalam STT, Malayalam questions
    ├─ Gujarati → Meta MMS Gujarati, Gujarati STT, Gujarati questions
    ├─ Bengali → Meta MMS Bengali, Bengali STT, Bengali questions
    └─ Punjabi → Meta MMS Punjabi, Punjabi STT, Punjabi questions

Switching language mid-test:
    ├─ Voice pipeline disconnects
    ├─ Reconnects with new language/voice
    └─ Next question spoken in new language
```

---

## Flow 2: HITL Voice Review

### 2.1 Reviewer Onboarding

```
Reviewer opens review.html
    │
    ├─ First time: clicks "Register"
    │   ├─ Enters name, email, password (min 6 chars)
    │   ├─ POST /api/review/auth/register
    │   └─ Account created → sign in
    │
    └─ Returning: enters email + password
        ├─ POST /api/review/auth/login
        ├─ Receives auth token (stored in localStorage)
        └─ Dashboard loads
```

### 2.2 Review Workflow

```
Dashboard shows stats bar:
    ├─ Total utterances
    ├─ Needs review (red)
    ├─ Reviewed (green)
    ├─ Accuracy %
    └─ Garbage count

Reviewer sets filters:
    ├─ Date (calendar picker)
    ├─ Status: All / Needs Review / Reviewed / Not Understood
    ├─ Session ID (optional)
    └─ Clicks "Filter"

Utterance table loads:
    ┌────┬──────┬──────────────┬──────────┬──────┬───────────┬────────┬──────────┐
    │ ☐  │ ▶    │ Transcript   │ Matched  │ Conf │ Correct   │ Status │ Reviewer │
    ├────┼──────┼──────────────┼──────────┼──────┼───────────┼────────┼──────────┤
    │ ☐  │ [▶]  │ "the first"  │ BETTER_1 │ 100% │ [dropdown]│ Match  │          │
    │ ☐  │ [▶]  │ "mumble"     │ —        │ 0%   │ [dropdown]│ NoMatch│          │
    │ ☐  │ [▶]  │ "clear yes"  │ READABLE │ 85%  │ [dropdown]│ Match  │ raj@     │
    └────┴──────┴──────────────┴──────────┴──────┴───────────┴────────┴──────────┘

For each utterance, reviewer can:
    │
    ├─ Click ▶ to play audio (inline, with auth)
    │
    ├─ Listen and verify:
    │   ├─ Correct match? → leave dropdown as-is, click Save
    │   ├─ Wrong match? → select correct option from dropdown, click Save
    │   ├─ Garbage/noise? → select GARBAGE from dropdown, click Save
    │   └─ Bulk: check multiple rows → click "Approve Selected"
    │
    └─ Annotation saved:
        ├─ PUT /api/review/utterances/{session_id}/{utt_id}
        ├─ Updates manifest.jsonl: reviewed=true, reviewed_by, correct_option
        └─ Stats bar refreshes
```

### 2.3 Export Training Data

```
Reviewer clicks "Export Whisper" or "Export Intent"
    │
    ├─ GET /api/review/export?format=whisper
    │   └─ Generates dataset_whisper.jsonl:
    │       {"audio": "/path/to/utt_0001.flac", "transcript": "clear", "language": "en"}
    │
    └─ GET /api/review/export?format=intent
        └─ Generates dataset_intent.jsonl:
            {"audio": "/path/to/utt_0001.flac", "intent": "READABLE", "response_type": "READABILITY"}
```

---

## Flow 3: Training & Model Improvement

### 3.1 Automated Weekly Retrain

```
Monday 3:00 AM (cron)
    │
    ├─ weekly_retrain.py starts
    │
    ├─ Check: >50 new reviewed utterances?
    │   ├─ No → skip training
    │   └─ Yes ↓
    │
    ├─ Step 1: Fine-tune Whisper
    │   ├─ Load reviewed (audio, transcript) pairs
    │   ├─ Train with HuggingFace Seq2SeqTrainer
    │   ├─ Evaluate WER on 10% holdout
    │   └─ Save model: voice/models/whisper-finetuned/vN/
    │
    ├─ Step 2: A/B Test
    │   ├─ Run current model + new model on same data
    │   ├─ Compare accuracy, review rate, inference time
    │   └─ Log recommendation: promote or keep
    │
    ├─ Step 3: Optimize thresholds
    │   ├─ Sweep confidence 40-100% per response_type
    │   └─ Report optimal threshold per type
    │
    ├─ Step 4: Expand fuzzy matcher
    │   ├─ Extract new phrases from corrections
    │   └─ Generate code patch for review
    │
    └─ Log run: ~/.eye_test_audio/_analysis/retrain_log.jsonl
```

### 3.2 Manual Training Commands

```
# Fine-tune Whisper
python -m voice.training.whisper_finetune --epochs 3

# A/B test models
python -m voice.training.ab_testing --model-a small --model-b v1

# Optimize confidence thresholds
python -m voice.training.confidence_optimizer

# Auto-expand fuzzy matcher keywords
python -m voice.training.matcher_expansion

# Train intent classifier
python -m voice.training.intent_classifier --train

# Sync to central server
python -m voice.training.server_sync --push
python -m voice.training.server_sync --pull
```

---

## Flow 4: Multi-Clinic Sync

```
                    Clinic Machine                    Central Server
                    ──────────────                    ──────────────
During the day:
    Tests conducted
    Audio recorded to
    ~/.eye_test_audio/

2:00 AM (cron):
    server_sync --push ──────────────→ /api/sync/upload
    (rsync or HTTP)                    Stores under
                                       clinics/{clinic_id}/

3:00 AM:                              Weekly retrain runs
                                       on aggregated data
                                       New model: vN

4:00 AM (cron):
    server_sync --pull ←──────────── /api/sync/models/latest
    Downloads model                   Returns tar.gz of
    to whisper-finetuned/vN           latest version

Next day:
    Voice pipeline loads
    latest local model
    automatically
```

---

## UI Layout Reference

### Test UI (index.html)
```
┌─────────────────────────────────────────────────────────────────┐
│ [status] Eye Test v2 [device▼] [Live] [Reset] [🎤Voice] [▼EN] │ ← top bar
├────────────┬────────────────────────────────────────────────────┤
│ Progress   │                                                    │
│ ▸ B ●      │   ┌────────────────────────────────────────────┐  │
│   D ○      │   │ Phase B: Coarse Sphere RE    [RIGHT EYE]  │  │
│   E ○      │   │                                            │  │
│   F ○      │   │ Can you read the letters clearly?          │  │
│   ...      │   │                                            │  │
│            │   │ [1. READABLE] [2. NOT_READABLE] [3. BLURRY]│  │
│ Voice Log  │   │                                            │  │
│ ┌────────┐ │   │ "clear"                                    │  │ ← voice transcript
│ │AI: ... │ │   └────────────────────────────────────────────┘  │
│ │ "clear"│ │                                                    │
│ │  ✓READ │ │   [Status: Active] [Phase: B] [Step: 3]          │
│ └────────┘ │   [RE: -2.25/-0.50×180] [LE: -1.75/-0.25×175]   │
│            │                                                    │
│ History    │                                                    │
│ ▸ ...      │                                                    │
│            │                                                    │
│ [New][End] │                                                    │
└────────────┴────────────────────────────────────────────────────┘
```

### HITL Review (review.html)
```
┌─────────────────────────────────────────────────────────────────┐
│ HITL Voice Review                    [Export▼] [user@] [Logout] │
├─────────────────────────────────────────────────────────────────┤
│ [Total: 142] [Needs Review: 23] [Reviewed: 119] [Accuracy: 94%]│
├─────────────────────────────────────────────────────────────────┤
│ Date [____] Status [All▼] Session [____]  [Filter] [Clear]     │
├─────────────────────────────────────────────────────────────────┤
│ ☐ [Approve Selected]                                            │
├────┬──────┬────────────┬──────────┬──────┬─────────┬───────────┤
│ ☐  │ ▶    │ "clear"    │ READABLE │ 100% │ [▼    ] │ ✓Reviewed │
│ ☐  │ ▶    │ "mumble"   │ —        │  0%  │ [▼    ] │ ✗NoMatch  │
│ ☐  │ ▶    │ "pehla"    │ BETTER_1 │ 100% │ [▼    ] │ ✓Match    │
├────┴──────┴────────────┴──────────┴──────┴─────────┴───────────┤
│                    [< Prev] [1] [2] [3] [Next >]                │
└─────────────────────────────────────────────────────────────────┘
```
