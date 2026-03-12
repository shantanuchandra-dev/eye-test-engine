# Eye Test Engine v2 — Implementation Plan

## Executive Summary

This document outlines the plan to create `Eye_test_engine_v2` — a new version of the eye test engine that:

1. **Replaces** the old hand-coded state machine (`interactive_session.py` + `core/state_machine.py`) with the **FSMv2.2 refraction engine** logic
2. **Retains** the existing Flask API layer, phoropter integration, frontend UI, logging, and dashboard
3. **Adds** a proper patient intake flow (the FSMv2.2 `DerivedVariables` system requires patient data to configure the test)

---

## Analysis: What Exists Today

### eye_test_engine (v1) — Current System
| Component | Status | Notes |
|---|---|---|
| **Frontend UI** (`index.html`) | Keep & Update | Full-featured: session management, power display, response buttons, chart switching, phoropter controls, request logs |
| **Dashboard** (`dashboard.html`) | Keep As-Is | R&R metrics, session history, operator stats |
| **Flask API** (`api_server.py`) | Keep & Refactor | Proxy to phoropter broker, session CRUD, logging endpoints |
| **Interactive Session** (`interactive_session.py`) | **Replace** | 1400+ lines of hand-coded phase logic — this is what FSMv2.2 replaces |
| **State Machine** (`core/state_machine.py`) | **Replace** | Old phase detection logic — replaced by FSMv2.2 `state_transitions.py` |
| **Context/RowContext** (`core/context.py`) | **Replace** | Replaced by `FSMRuntimeRow` |
| **Config YAMLs** (`protocol.yaml`, `thresholds.yaml`) | **Replace** | FSMv2.2 uses `calibration.csv` + `DerivedVariables` instead |
| **Phoropter API** (`curl_API.md`) | Keep As-Is | Reference doc for hardware integration |
| **IO/Logging** (`io/outputs.py`, `io/remote_storage.py`) | Keep & Adapt | CSV/metadata logging, Supabase upload |
| **Frontend JS** (`app.js`) | Keep & Update | Needs updates for new response types and intake form |

### FSMv2.2 — New Logic Source
| Component | Status | Notes |
|---|---|---|
| **RefractionFSMEngine** (`engines/refraction_fsm_engine.py`) | **Core** | The main decision engine — `initialize_row()`, `apply_response()`, `_build_next_row()` |
| **State Transitions** (`engines/state_transitions.py`) | **Core** | `compute_next_state()` — all transition logic in one function |
| **Delta Calculators** (`engines/delta_calculators.py`) | **Core** | How prescription values change per response |
| **Escalation Rules** (`engines/escalation_rules.py`) | **Core** | Safety guardrails (anomaly detection, timeouts) |
| **DerivedVariables** (`models/derived_variables.py`) | **Core** | Clinical configuration computed from patient input |
| **DerivedVariablesEngine** (`engines/derived_variables_engine.py`) | **Core** | Derives all test parameters from patient data |
| **FSMRuntimeRow** (`models/fsm_runtime.py`) | **Core** | Single FSM step data structure |
| **PatientInput** (`models/patient.py`) | **Core** | Patient data model (age, symptoms, AR/Lenso, medical history) |
| **EyePrescription** (`models/prescription.py`) | **Core** | SPH/CYL/AXIS data structure |
| **Chart Scale** (`charts/chart_scale.py`) | **Core** | VA chart progression logic |
| **CalibrationLoader** (`config/calibration_loader.py`) | **Core** | Loads calibration.csv tuning parameters |
| **Simulation modules** | **Exclude** | Not needed for production |

---

## FSM State Flow (FSMv2.2)

The FSM defines these states (single-letter IDs) with clear transitions:

```
A: Distance Baseline (BINO) ──────────────────────────────┐
│  Q: "Can you read this clearly?"                         │
│  Responses: READABLE / NOT_READABLE / BLURRY             │
│  If READABLE and chart >= target → B                     │
│  If NOT_READABLE/BLURRY → B                              │
│                                                           │
B: Coarse Sphere RE (Right Eye) ───────────────────────────┤
│  Q: "Can you read this clearly?"                         │
│  Adjusts: R_SPH (fogged, then stepped down)              │
│  If READABLE and chart >= target → E                     │
│  Timeout/Escalation → ESCALATE                           │
│                                                           │
E: JCC Axis RE ────────────────────────────────────────────┤
│  Q: "Which is clearer: 1 or 2?"                         │
│  Adjusts: R_AXIS (±5°/3°/1° reversal stepping)          │
│  If SAME/CANT_TELL → F                                   │
│                                                           │
F: JCC Power RE ───────────────────────────────────────────┤
│  Q: "Which is clearer: 1 or 2?"                         │
│  Adjusts: R_CYL + SPH compensation                      │
│  If SAME streak or flip limit → G                        │
│                                                           │
G: Duochrome RE ───────────────────────────────────────────┤
│  Q: "Which is clearer: RED or GREEN?"                    │
│  Adjusts: R_SPH (red=minus, green=plus)                  │
│  If EQUAL streak or flip limit → D                       │
│                                                           │
D: Coarse Sphere LE (Left Eye) ────────────────────────────┤
│  (Same as B but for left eye)                            │
│  If READABLE and chart >= target → H                     │
│                                                           │
H: JCC Axis LE ────────────────────────────────────────────┤
│  (Same as E but for left eye)                            │
│  If SAME/CANT_TELL → I                                   │
│                                                           │
I: JCC Power LE ───────────────────────────────────────────┤
│  (Same as F but for left eye)                            │
│  If SAME streak or flip limit → J                        │
│                                                           │
J: Duochrome LE ───────────────────────────────────────────┤
│  (Same as G but for left eye)                            │
│  If done → K (or skip to P/END)                          │
│                                                           │
K: Binocular Balance ──────────────────────────────────────┤
│  Q: "Which is clearer: TOP or BOTTOM?"                   │
│  Adjusts: R_SPH or L_SPH                                │
│  Skip if |R_SPH - L_SPH| ≤ 0.25                         │
│  If SAME or flip limit → P/END                           │
│                                                           │
P: Near Add RE ────────────────────────────────────────────┤
│  Q: "Can you read the near target clearly?"              │
│  Adjusts: R_ADD                                          │
│  If READABLE → Q                                         │
│                                                           │
Q: Near Add LE ────────────────────────────────────────────┤
│  Adjusts: L_ADD                                          │
│  If READABLE → R                                         │
│                                                           │
R: Near Binocular ─────────────────────────────────────────┤
│  Adjusts: both ADD                                       │
│  If TARGET_OK → END                                      │
│                                                           │
END ── Test Complete                                        │
ESCALATE ── Requires Optometrist Review                    │
```

---

## User Journey & Screens

### Screen 1: Login / Device Setup
- Select/scan phoropter device ID
- Acquire device (brain lock)
- Auto-heartbeat starts
- **Existing:** This is partially in the current UI (phoropter ID input + device acquisition buttons)

### Screen 2: Patient Intake Form (NEW)
This is **required by FSMv2.2** to compute `DerivedVariables` which configure the entire test.

**Fields needed (from `PatientInput` model):**

| Section | Fields |
|---|---|
| **Demographics** | Age, Occupation, Screen time (hrs/day), Driving hours |
| **Chief Complaint** | Primary reason (dropdown: Blurred distance / Blurred near / Headaches / Routine check / New glasses), Symptoms text (multi-select: Sudden vision change, Double vision, Glare/halos, Night driving difficulty, Fluctuating vision) |
| **Current Glasses** | Satisfaction (Satisfied / Not satisfied / No current Rx), Wear type (Single vision / Progressive / Bifocal / None), Distance target preference (6/6 target / Accept 6/9 if needed), Priority (Standard / Comfort-first), Near priority (High / Medium / Low) |
| **History** | Last eye test (months ago), Rx change was large (Y/N), Fluctuating vision reported (Y/N) |
| **Medical** | Diabetes (Y/N), Prior eye surgery (None / Cataract / LASIK / Other), Keratoconus (Y/N), Amblyopia (Y/N), Infection (Y/N), Optom review flag (Y/N) |
| **AR Readings** | Right eye: SPH, CYL, AXIS; Left eye: SPH, CYL, AXIS |
| **Lensometry** | Right eye: SPH, CYL, AXIS; Left eye: SPH, CYL, AXIS; ADD_R, ADD_L |

**On Submit:**
1. Backend creates `PatientInput` object
2. `DerivedVariablesEngine.derive()` computes all test parameters
3. `RefractionFSMEngine.initialize_row()` creates the first FSM step
4. UI transitions to the test screen

### Screen 3: Eye Test Session (Updated Existing UI)
The core testing screen — largely exists already but needs updates:

| Sub-section | Current Status | Changes Needed |
|---|---|---|
| **Phase indicator** | Shows phase name | Map to FSMv2.2 states (A/B/D/E/F/G/H/I/J/K/P/Q/R) |
| **Question display** | Shows optometrist question | Populated from `FSMRuntimeRow.question` |
| **Response buttons** | Dynamic intents | Populated from `FSMRuntimeRow.opt_1..opt_6` |
| **Power display** | RE/LE SPH/CYL/AXIS/ADD | Same — values come from FSMRuntimeRow |
| **Chart display** | Chart name indicator | Map `chart_param` to chart commands |
| **Phoropter controls** | Reset, JCC, manual power | Keep as-is |
| **Eye indicator** | Which eye being tested | From `FSMRuntimeRow.eye` (RE/LE/BIN) |
| **Progress tracker** | Phase list sidebar | Update to show FSM state progression |

**Response flow per state:**

| State | Response Type | Button Options |
|---|---|---|
| A (Distance Baseline) | READABILITY | READABLE, NOT_READABLE, BLURRY |
| B (Coarse Sphere RE) | READABILITY | READABLE, NOT_READABLE, BLURRY |
| D (Coarse Sphere LE) | READABILITY | READABLE, NOT_READABLE, BLURRY |
| E (JCC Axis RE) | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| F (JCC Power RE) | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| G (Duochrome RE) | COLOR_CHOICE | RED_CLEARER, GREEN_CLEARER, EQUAL, CANT_TELL |
| H (JCC Axis LE) | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| I (JCC Power LE) | COMPARE_1_2 | BETTER_1, BETTER_2, SAME, CANT_TELL |
| J (Duochrome LE) | COLOR_CHOICE | RED_CLEARER, GREEN_CLEARER, EQUAL, CANT_TELL |
| K (Binocular Balance) | TOP_BOTTOM | TOP_CLEARER, BOTTOM_CLEARER, SAME, CANT_TELL |
| P (Near Add RE) | NEAR_READABILITY | READABLE, NOT_READABLE, BLURRY |
| Q (Near Add LE) | NEAR_READABILITY | READABLE, NOT_READABLE, BLURRY |
| R (Near Binocular) | NEAR_BINOC | TARGET_OK, NOT_CLEAR |

### Screen 4: Test Complete / Prescription Summary
- Final prescription display (RE & LE: SPH, CYL, AXIS, ADD)
- Comparison with AR/Lenso input values
- Option to accept/discard/redo
- Operator name, customer details entry
- Qualitative feedback
- **Existing:** End-session flow already exists

### Screen 5: Escalation Screen (NEW)
When FSM returns `ESCALATE`:
- Show reason (timeout, safety threshold, or anomaly detection)
- Show current prescription state
- Option for remote optometrist to take over manually
- Option to override and continue

### Screen 6: Dashboard (Keep As-Is)
- `dashboard.html` — no changes needed

---

## Implementation Phases

### Phase 1: Project Setup & FSM Integration (Backend)
**Goal:** Create v2 folder, port FSM engine, create new session orchestrator

1. Create `/Eye_test_engine_v2/` folder structure:
   ```
   Eye_test_engine_v2/
   ├── api_server.py              # Updated Flask API
   ├── session_orchestrator.py    # New: replaces interactive_session.py
   ├── fsm/                       # FSMv2.2 engine (copied & adapted)
   │   ├── __init__.py
   │   ├── refraction_engine.py   # RefractionFSMEngine
   │   ├── state_transitions.py   # compute_next_state
   │   ├── delta_calculators.py   # All delta functions
   │   ├── escalation_rules.py    # Safety checks
   │   ├── derived_variables.py   # DerivedVariables dataclass
   │   ├── derived_variables_engine.py  # Compute DV from patient
   │   ├── fsm_runtime.py         # FSMRuntimeRow
   │   ├── patient.py             # PatientInput
   │   ├── prescription.py        # EyePrescription
   │   └── chart_scale.py         # Chart progression
   ├── config/
   │   └── calibration.csv        # Tuning parameters (new)
   ├── io/
   │   ├── outputs.py             # CSV/metadata logging (from v1)
   │   ├── remote_storage.py      # Supabase upload (from v1)
   │   └── dashboard_data.py      # Dashboard data (from v1)
   ├── frontend/
   │   ├── index.html             # Updated main UI
   │   ├── intake.html            # NEW: Patient intake form
   │   ├── app.js                 # Updated frontend JS
   │   ├── dashboard.html         # Kept from v1
   │   └── favicon.svg
   ├── curl_API.md                # Phoropter API reference
   ├── requirements.txt
   ├── run.py
   └── vercel.json
   ```

2. Port FSMv2.2 engine files (remove pandas dependency from CalibrationLoader — use stdlib csv)
3. Create `calibration.csv` with default values (extract from FSMv2.2 code defaults)
4. Build `session_orchestrator.py`:
   - Wraps `RefractionFSMEngine` + `DerivedVariablesEngine`
   - Maps FSM states to phoropter API commands (chart selection, occluder, JCC)
   - Tracks session history for logging
   - Provides `start_session(patient_input)` → first stimulus
   - Provides `process_response(response_value)` → next stimulus or END/ESCALATE

### Phase 2: API Layer Update
**Goal:** Update Flask API to support intake flow and new FSM responses

1. New endpoints:
   - `POST /api/session/intake` — Accept patient data, derive variables, initialize FSM
   - `GET /api/session/<id>/derived-variables` — Return computed DV for display
   - Response format changes to include FSMRuntimeRow fields

2. Updated endpoints:
   - `POST /api/session/<id>/respond` — Now accepts FSMv2.2 response values (READABLE, BETTER_1, RED_CLEARER, etc.)
   - Response includes: `state`, `phase_name`, `question`, `options[]`, `eye`, `chart_param`, `prescription`, `next_state`

3. Keep existing:
   - All phoropter proxy endpoints
   - Device management (acquire/release/heartbeat)
   - Session end/discard
   - Dashboard endpoints
   - Sync-power endpoint (for manual overrides)

### Phase 3: Phoropter Command Mapping
**Goal:** Map FSM states to correct phoropter API commands

| FSM State | Occluder (aux_lens) | Chart | JCC Action |
|---|---|---|---|
| A (Distance Baseline) | BINO | Snellen (by chart_param) | None |
| B (Coarse Sphere RE) | AuxLensL (occlude left) | Snellen (by chart_param) | None |
| D (Coarse Sphere LE) | AuxLensR (occlude right) | Snellen (by chart_param) | None |
| E (JCC Axis RE) | AuxLensL | JCC chart (chart_19) | power_axis_switch → Axis mode |
| F (JCC Power RE) | AuxLensL | JCC chart (chart_19) | power_axis_switch → Power mode |
| G (Duochrome RE) | AuxLensL | Duochrome (chart_17) | None |
| H (JCC Axis LE) | AuxLensR | JCC chart (chart_19) | power_axis_switch → Axis mode |
| I (JCC Power LE) | AuxLensR | JCC chart (chart_19) | power_axis_switch → Power mode |
| J (Duochrome LE) | AuxLensR | Duochrome (chart_17) | None |
| K (Binocular Balance) | BINO | BINO chart (chart_20) | None |
| P (Near Add RE) | AuxLensL | Near chart (Chart5) | None |
| Q (Near Add LE) | AuxLensR | Near chart (Chart5) | None |
| R (Near Binocular) | BINO | Near chart (Chart5) | None |

Chart param to phoropter chart_items mapping:
| chart_param | chart_items |
|---|---|
| 400 | ["chart_9"] (E-chart) |
| 200_150 | ["chart_10"] |
| 100_80 | ["chart_11"] |
| 70_60_50 | ["chart_12"] |
| 40_30_25 | ["chart_13"] |
| 20_20_20 | ["chart_15"] |

### Phase 4: Frontend — Intake Form
**Goal:** Build patient intake screen

1. Create `intake.html` with form sections matching `PatientInput` fields
2. AR/Lenso values entry with +/- steppers (0.25D steps for SPH/CYL, 1° steps for AXIS)
3. On submit → `POST /api/session/intake` → receive derived variables summary → start test
4. Show derived variables summary (confidence level, step sizes, fogging policy, etc.) before starting

### Phase 5: Frontend — Test Screen Updates
**Goal:** Update main test UI for FSMv2.2 response types

1. Dynamic response buttons from `opt_1..opt_6` (already partially supported)
2. State-aware question display
3. Updated phase sidebar with FSMv2.2 state names
4. Escalation handling (show alert, options to override or stop)
5. Color-coded response buttons:
   - READABILITY: Green (readable), Orange (blurry), Red (not readable)
   - COMPARE: Blue (1), Blue (2), Gray (same), Gray (can't tell)
   - DUOCHROME: Red, Green, Gray (equal)
   - BINOCULAR: Top, Bottom, Gray (same)
6. Keep all existing features: manual power controls, chart switching, request logs, prev-state sync

### Phase 6: Calibration & Testing
**Goal:** Create default calibration, test end-to-end

1. Create `calibration.csv` with sensible defaults extracted from FSMv2.2 code
2. End-to-end test: intake → distance baseline → coarse sphere → JCC → duochrome → binocular balance → near → END
3. Test escalation paths (timeout, safety thresholds)
4. Test skip binocular balance condition
5. Test with actual phoropter API (or mock)

---

## Key Differences: v1 vs v2

| Aspect | v1 (Current) | v2 (New) |
|---|---|---|
| **State machine** | Hand-coded phases in `interactive_session.py` (~1400 lines) | FSMv2.2 engine (~650 lines, well-structured) |
| **Phase flow** | A→B→E→F→G→H→I→J→K→L→M→N→O→P→Q→R (17 phases including validation) | A→B→E→F→G→D→H→I→J→K→P→Q→R→END (13 states, cleaner) |
| **Configuration** | YAML files (protocol.yaml, thresholds.yaml) | calibration.csv (single tuning file) |
| **Patient intake** | None — starts with 0/0/180 | Full intake form → DerivedVariables (fogging, step sizes, safety thresholds) |
| **Fogging** | Not implemented | Full fogging support (Strong/Standard/None) |
| **Escalation** | Not implemented | Built-in (anomaly detection, timeout, optom review) |
| **Sphere compensation** | Manual tracking in session | Automatic via `jcc_power_sphere_compensation()` |
| **Duochrome endpoint bias** | Not implemented | Configurable (Undercorrect/Overcorrect/Neutral) |
| **Axis step refinement** | Fixed 5° steps | Adaptive: 5° → 3° → 1° on reversals |
| **Validation phases** | H, M, N (20/20 validation) | Removed — duochrome serves as endpoint |
| **Binocular balance skip** | Always runs | Auto-skip when |RE_SPH - LE_SPH| ≤ 0.25 |

---

## Risk & Mitigation

| Risk | Mitigation |
|---|---|
| CalibrationLoader needs pandas | Rewrite with stdlib `csv` module |
| No calibration.csv exists | Create default from code defaults |
| Phoropter command mapping errors | Keep manual override controls |
| FSMv2.2 doesn't have validation phases | The validation was redundant with duochrome — acceptable to remove |
| Patient intake adds friction | Make fields optional with sensible defaults |

---

## Sign-off Checklist

Please review and confirm:

- [ ] **Phase flow:** A→B→E→F→G→D→H→I→J→K→P→Q→R→END is acceptable
- [ ] **Intake form:** Required fields are correct and complete
- [ ] **No validation phases:** Removing 20/20 validation (H, M, N) is acceptable (duochrome serves as endpoint)
- [ ] **Escalation behavior:** Show alert + allow manual override is acceptable
- [ ] **Folder structure:** `Eye_test_engine_v2/` in root directory
- [ ] **Implementation order:** Phase 1-6 as described above
- [ ] **Any additional screens or features needed?**

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Browser)                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │
│  │ intake.html   │  │ index.html   │  │ dashboard │  │  app.js   │ │
│  │ Patient Form  │→ │ Test Screen  │  │  .html    │  │ (shared)  │ │
│  │ Demographics  │  │ Questions    │  │ R&R Stats │  │           │ │
│  │ AR/Lenso Rx   │  │ Responses    │  │ Sessions  │  │ API calls │ │
│  │ Medical Hx    │  │ Power Table  │  │ Export    │  │ Phoropter │ │
│  │ Symptoms      │  │ Debug Panel  │  │           │  │ Controls  │ │
│  └──────┬───────┘  │ Live View    │  └───────────┘  │ UI State  │ │
│         │          │ Logs Panel   │                  └───────────┘ │
│         │          └──────┬───────┘                                 │
└─────────┼─────────────────┼─────────────────────────────────────────┘
          │ POST /intake    │ POST /respond, GET /status, GET /derived-variables
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FLASK API SERVER (api_server.py)                 │
│                                                                     │
│  Session CRUD       Phoropter Proxy      Dashboard        Frontend  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │
│  │ POST /intake │  │ GET /devices │  │ GET /stats   │  │ Static │ │
│  │ POST /respond│  │ POST /acquire│  │ GET /rr      │  │ Files  │ │
│  │ GET /status  │  │ POST /release│  │ PUT /config  │  │        │ │
│  │ GET /derived │  │ POST /hbeat  │  │ GET /export  │  │        │ │
│  │ POST /sync   │  │ POST /reset  │  └──────────────┘  └────────┘ │
│  │ POST /end    │  │ POST /pinhole│                                │
│  │ POST /discard│  │ POST /screen │                                │
│  └──────┬───────┘  └──────┬───────┘                                │
└─────────┼─────────────────┼─────────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│  SESSION LAYER   │  │         PHOROPTER BROKER                     │
│                  │  │  rajasthan-royals.preprod.lenskart.com       │
│ SessionOrchest-  │  │                                              │
│ rator            │  │  ┌──────────────────────────────────┐       │
│ ┌──────────────┐ │  │  │  TOPCON Digital Phoropter        │       │
│ │ Initialize   │ │  │  │  - Power control (SPH/CYL/AXIS) │       │
│ │ Process Resp │ │  │  │  - Aux lens (occluder)           │       │
│ │ Sync Power   │ │  │  │  - Chart display                 │       │
│ │ End Session  │ │  │  │  - JCC cross-cylinder            │       │
│ └──────┬───────┘ │  │  │  - Near ADD                      │       │
│        │         │  │  │  - Screenshot capture             │       │
│        ▼         │  │  └──────────────────────────────────┘       │
│ ┌──────────────┐ │  └──────────────────────────────────────────────┘
│ │ FSMv2.2      │ │
│ │ Engine       │ │
│ └──────┬───────┘ │
│        │         │
│        ▼         │
│ ┌──────────────┐ │
│ │ IO / Logging │ │
│ │ CSV + Meta   │ │
│ │ Supabase     │ │
│ └──────────────┘ │
└──────────────────┘
```

### Component Architecture

```
Eye_test_engine_v2/
│
├── api_server.py                 ← Flask HTTP layer
│   Responsibilities:
│   - Route requests to SessionOrchestrator
│   - Proxy phoropter API calls (avoids CORS)
│   - Serve frontend static files
│   - Dashboard data endpoints
│
├── session_orchestrator.py       ← Orchestration layer
│   Responsibilities:
│   - Build PatientInput from form data
│   - Invoke DerivedVariablesEngine
│   - Drive RefractionFSMEngine step-by-step
│   - Map FSM states → phoropter commands
│   - Track session history for logging
│   - Provide derived/working variables for debug
│
├── fsm/                          ← Clinical logic (from FSMv2.2)
│   ├── refraction_engine.py      ← Core FSM: initialize, apply_response, build_next
│   ├── derived_variables_engine.py ← 30+ derivation functions
│   ├── state_transitions.py      ← compute_next_state, compute_phase_max
│   ├── delta_calculators.py      ← Power change calculations per state
│   ├── escalation_rules.py       ← Safety: anomaly detection, timeouts
│   ├── calibration.py            ← Load calibration.csv (stdlib csv)
│   ├── chart_scale.py            ← VA chart progression logic
│   ├── derived_variables.py      ← DerivedVariables dataclass (46 fields)
│   ├── fsm_runtime.py            ← FSMRuntimeRow dataclass (78 fields)
│   ├── patient.py                ← PatientInput dataclass
│   └── prescription.py           ← EyePrescription dataclass
│
├── config/
│   └── calibration.csv           ← 98 tuning parameters (14 sections)
│
├── io/                           ← Logging & storage (from v1)
│   ├── outputs.py                ← CSV session logs, metadata JSON
│   ├── remote_storage.py         ← Supabase upload
│   └── dashboard_data.py         ← Dashboard aggregations
│
└── frontend/                     ← Browser UI
    ├── index.html                ← Main test screen
    ├── intake.html               ← Patient intake form (NEW)
    ├── app.js                    ← Shared JavaScript
    ├── dashboard.html            ← R&R dashboard
    └── favicon.svg
```

### Data Flow

#### 1. Session Initialization
```
User fills intake form
  → POST /api/session/intake { patient: {...}, phoropter_id: "..." }
    → SessionOrchestrator.initialize(patient_data)
      → _build_patient_input(data) → PatientInput
      → DerivedVariablesEngine.derive(patient_input) → DerivedVariables (46 fields)
      → RefractionFSMEngine.initialize_row(visit_id, dv, ar_re, ar_le) → FSMRuntimeRow
      → _send_phoropter_commands(row)  [chart + power + occluder]
    ← { state: "A", question, options, power, ... }
  ← UI transitions to test screen
```

#### 2. Response Processing Loop
```
User clicks response button (e.g., "READABLE")
  → POST /api/session/<id>/respond { response: "READABLE" }
    → SessionOrchestrator.process_response("READABLE")
      → RefractionFSMEngine.apply_response(current, "READABLE", dv, ar_re, ar_le)
        → delta_calculators compute power changes
        → escalation_rules check safety
        → state_transitions.compute_next_state() → next state
      → _record_row(finalized)  [→ session_history]
      → RefractionFSMEngine._build_next_row(finalized, dv) → next FSMRuntimeRow
      → _send_phoropter_commands(next_row)  [chart + power + occluder]
    ← { state: "B", question, options, power, step_info, ... }
  ← UI updates question, buttons, power table
```

#### 3. Phoropter Command Pipeline
```
_send_phoropter_commands(row):
  1. Determine chart type from STATE_CHART_MAP[state]
     → Send chart command (snellen/jcc/duochrome/bino/near)
  2. Build run-tests payload with prev_state for delta calculation:
     { test_cases: [{ prev_aux_lens, prev_right_eye, prev_left_eye,
                      aux_lens, right_eye, left_eye }] }
     → POST to phoropter broker
  3. For JCC states (E,F,H,I): send power_axis_switch command
  4. Update _prev_re, _prev_le, _prev_aux_lens for next iteration
```

#### 4. DerivedVariables Pipeline
```
PatientInput (user form data)
  → DerivedVariablesEngine.derive()
    → _derive_age_bucket(age)
    → _derive_symptom_risk(symptoms, sudden_change, double_vision, ...)
    → _derive_medical_risk(diabetes, surgery, keratoconus, ...)
    → _derive_stability(last_test_months, rx_change_large, fluctuating)
    → _derive_ar_lenso_mismatch(ar_rx, lenso_rx)  [per eye]
    → _derive_start_source_policy(mismatch_level, stability)
    → _derive_start_rx(policy, ar, lenso, current)  [per eye]
    → _derive_fogging_policy(age_bucket, medical_risk, stability)
    → _derive_step_sizes(confidence_req, stability)
    → _derive_safety_thresholds(medical_risk, symptom_risk)
    → _derive_near_test_required(age, add_expected, near_priority)
    → ... (30+ derivation functions)
  → DerivedVariables (46 fields configuring entire test)
```

### State Machine Flow

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         ▼
                    ┌─────────┐
                    │  A: Dist │  Both eyes open (BINO)
                    │ Baseline │  "Can you read this?"
                    └────┬────┘
                         ▼
              ┌──────────────────────┐
              │  RIGHT EYE (RE)      │  Left eye occluded
              │                      │
              │  B: Coarse Sphere    │  Fog → step down SPH
              │  E: JCC Axis         │  ±5°/3°/1° reversals
              │  F: JCC Power        │  CYL + SPH compensation
              │  G: Duochrome        │  Red/Green endpoint
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  LEFT EYE (LE)       │  Right eye occluded
              │                      │
              │  D: Coarse Sphere    │
              │  H: JCC Axis         │
              │  I: JCC Power        │
              │  J: Duochrome        │
              └──────────┬───────────┘
                         ▼
                    ┌─────────┐
                    │ K: Bino  │  Skip if |RE_SPH-LE_SPH| ≤ 0.25
                    │ Balance  │  Both eyes, Top/Bottom
                    └────┬────┘
                         ▼
              ┌──────────────────────┐
              │  NEAR VISION         │  (if age/add requires)
              │                      │
              │  P: Near Add RE      │
              │  Q: Near Add LE      │
              │  R: Near Binocular   │
              └──────────┬───────────┘
                         ▼
                ┌────────┴────────┐
                │      END        │
                │  or ESCALATE    │
                └─────────────────┘
```

### Key Technical Decisions

| Decision | Rationale |
|---|---|
| **Prev-state phoropter commands** | Phoropter broker needs previous power state to calculate click deltas accurately. JCC increase/decrease commands move the phoropter without updating the broker's tracker. |
| **Backend-driven phoropter control** | All phoropter commands are sent from `session_orchestrator.py`, not the frontend. This ensures FSM state and phoropter state stay in sync. Frontend only sends manual overrides. |
| **stdlib csv instead of pandas** | CalibrationLoader rewritten to avoid pandas dependency — keeps the deployment lightweight. |
| **importlib for io/ modules** | Python's built-in `io` module conflicts with the project's `io/` package. Using `importlib.util.spec_from_file_location` avoids the clash. |
| **SessionRow bridge dataclass** | FSMRuntimeRow has 78 fields; CSV logging needs 19. SessionRow maps between them. |
| **DerivedVariables debug panel** | Exposed via `GET /api/session/<id>/derived-variables` — shows all 46 derived variables and current working state for clinical debugging. |

---

*Please review this plan and provide your sign-off or feedback before implementation begins.*
