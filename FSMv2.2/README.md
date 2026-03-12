Below is the README.md version of the documentation formatted properly for GitHub / Markdown.

You can copy this entire block into README.md in the project root.

⸻


# AI Optometry FSM System

This repository contains the **Finite State Machine (FSM)** used to perform automated subjective refraction.

The project is divided into two independent systems:

1. **FSM Engine** — the production decision engine used during eye tests.
2. **FSM Simulation** — a validation framework used to test and benchmark the FSM using virtual patients and large-scale simulations.

The simulation system is used only for **R&D validation** and is **not required in production integration**.

---

# System Architecture

                 AI OPTOMETRY SYSTEM
                         │
                         │
             ┌───────────┴───────────┐
             │                       │
      FSM ENGINE                FSM SIMULATION
    (Production Logic)        (Testing Framework)
             │                       │
   Real patient interaction     Virtual patient models
   Phoropter control            Population testing
   Clinical decisions           Performance validation

---

# 1. FSM Engine (Production Decision Engine)

The FSM Engine is the **core system that runs the eye test**.

It determines:

- What stimulus to show
- How to adjust lens power
- When to move to the next phase
- When to escalate to an optometrist
- When to terminate the test

---

# FSM Engine Flow

Patient History / AR / Lenso
│
▼
Derived Variables Engine
│
▼
DerivedVariables Object
│
▼
RefractionFSMEngine
│
├── state_transitions.py
├── delta_calculators.py
├── escalation_rules.py
└── chart_scale.py
│
▼
FSMRuntimeRow (next stimulus)
│
▼
UI / Phoropter / Patient

---

# FSM Engine Directory Structure

refraction_engine
│
├── engines
│   ├── refraction_fsm_engine.py
│   ├── state_transitions.py
│   ├── delta_calculators.py
│   └── escalation_rules.py
│
├── models
│   ├── derived_variables.py
│   ├── fsm_runtime.py
│   ├── patient.py
│   └── prescription.py
│
├── charts
│   └── chart_scale.py
│
├── config
│   └── calibration_loader.py
│
└── data
└── calibration.csv

---

# FSM Engine Components

## refraction_fsm_engine.py

Core orchestration engine.

Responsibilities:

- Initialize FSM
- Apply patient responses
- Update prescription
- Generate next stimulus
- Track convergence

Main functions:

initialize_row()
apply_response()
_build_next_row()
run_visit()

This is the **primary interface used by the phoropter system**.

---

## state_transitions.py

Defines **FSM logic**.

Determines:

- Next FSM state
- Escalation conditions
- Test termination

Example transitions:

Distance Baseline → Coarse Sphere
Coarse Sphere → JCC Axis
JCC Power → Duochrome
Duochrome → Next Eye
Binocular Balance → Near

---

## delta_calculators.py

Defines **how prescription values change**.

Examples:

coarse_sphere_delta()
jcc_axis_delta()
jcc_power_cyl_delta()
duochrome_sphere_delta()
binocular_balance_delta()
near_add_delta()

---

## escalation_rules.py

Defines safety checks.

Escalation occurs when:

- Power deviates beyond thresholds
- Convergence fails
- Clinical anomalies are detected

---

## chart_scale.py

Defines visual acuity progression.

Example chart progression:

400 → 200 → 100 → 70 → 40 → 20

---

## derived_variables.py

Represents **clinical configuration variables**.

Controls:

confidence thresholds
step sizes
endpoint bias
maximum deltas
fogging policy
duochrome limits

Derived from:

patient history
autorefractor
lenso data
clinical policies

---

## fsm_runtime.py

Represents a **single FSM step**.

Each row contains:

state
power
chart
question
response
delta applied
next state

These rows form the **complete trace of the eye test**.

---

## calibration_loader.py

Loads parameters from:

data/calibration.csv

---

## calibration.csv

Primary tuning file for the FSM.

Controls:

sphere step sizes
cylinder step sizes
axis reversal step
duochrome confirmation count
JCC flip limits
binocular balance thresholds
timeouts

This allows **clinical tuning without code changes**.

---

# 2. FSM Simulation System

The FSM Simulation system is used to **test and validate the FSM engine**.

It enables:

- Virtual patient simulations
- Interactive testing
- Population-scale validation
- Accuracy benchmarking

---

# Simulation Flow

Truth Rx
│
▼
Virtual Patient Model
│
▼
FSM Engine
│
▼
Response generation
│
▼
Prescription convergence
│
▼
Accuracy analysis

---

# Simulation Directory Structure

refraction_engine
│
├── simulation
│   ├── population_runner.py
│   ├── virtual_patient.py
│   ├── behavior_models.py
│   ├── profile_library.py
│
├── run_single_patient_simulation.py
├── run_interactive_simulation.py
└── run_population_simulation.py

---

# Simulation Components

## virtual_patient.py

Simulates patient responses based on:

truth prescription
stimulus
current FSM state
behavior model

---

## behavior_models.py

Defines response behaviors:

ideal
noisy
inconsistent
hesitant
accommodative

Used to simulate realistic patient responses.

---

## profile_library.py

Defines clinical patient profiles:

Examples:

adult_hyperope
young_myope
presbyope
anisometrope
screen_heavy_user

Profiles control:

truth Rx distribution
clinical parameters
starting prescriptions

---

## population_runner.py

Runs large-scale simulations.

Steps:

Generate truth Rx
Generate patient profile
Generate patient behavior
Run FSM
Evaluate convergence
Record metrics

Outputs include:

accuracy
completion rate
escalation rate
step counts
prescription errors

---

# Simulation Entry Points

## Single Patient Simulation

python -m refraction_engine.run_single_patient_simulation

Purpose:

Debug FSM behavior
Trace a single eye test
Analyze convergence

Output:

results/single_patient//
trace.csv
summary.json

---

## Interactive Simulation

python -m refraction_engine.run_interactive_simulation

Purpose:

Manually simulate an eye test
Validate FSM transitions
Debug question flow

Output:

results/interactive//
trace.csv
summary.json

---

## Population Simulation

python -m refraction_engine.run_population_simulation

Purpose:

Validate FSM performance
Measure clinical accuracy
Stress test decision logic

Output:

results/population//
population_simulation_results.csv
leaderboard_profiles.csv
leaderboard_behaviors.csv
fsm_noncompletion_distribution.csv
rx_accuracy_metrics.csv
summary.json

---

# FSM Configuration

FSM behavior can be tuned at three levels.

---

## 1. Calibration (Low-Level Tuning)

File:

data/calibration.csv

Controls:

step sizes
flip limits
timeouts
confirmation thresholds

---

## 2. Derived Variables (Clinical Policy)

Defined in:

models/derived_variables.py

Examples:

endpoint_bias_policy
confidence_requirement
branching_guardrails
fogging_policy
duochrome_max_flips

---

## 3. Patient Profiles (Simulation Only)

Defined in:

simulation/profile_library.py

Controls:

truth Rx distributions
patient demographics
clinical scenarios

---

# Production Integration

The phoropter system interacts only with the **FSM Engine**.

Typical integration flow:

```python
engine = RefractionFSMEngine(calibration)

row = engine.initialize_row()

while True:

    display(row.question)

    response = get_patient_response()

    row = engine.apply_response(row, response)

    if row.next_state in ("END", "ESCALATE"):
        break


⸻

Result Storage

All simulation results are stored in:

results/

Structure:

results
│
├── single_patient
├── interactive
└── population

Each run creates a timestamped folder.

⸻

Summary

This repository contains two distinct systems.

FSM Engine

Clinical decision engine responsible for:

eye test logic
lens adjustments
state transitions
clinical rules


⸻

FSM Simulation

Testing framework responsible for:

virtual patients
population validation
accuracy analysis
performance benchmarking

Only the FSM Engine is required for production integration with the phoropter system.

---

If you'd like, I can also add **two extremely useful diagrams to this README**:

1. **Detailed FSM state transition diagram**
2. **End-to-end AI Optometry system architecture (phoropter + FSM + AI layers)**

Those make this documentation **much clearer for leadership and engineering teams**.