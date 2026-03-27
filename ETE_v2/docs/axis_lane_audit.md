# Axis Lane Audit

- Objective inputs enter through the intake/session orchestrator and are normalized into `PatientInput` plus AR/lensometry `EyePrescription` objects before the DV layer runs.
- Starting Rx source selection already lives in `DerivedVariablesEngine` via `dv_start_source_policy` and `_derive_start_rx`; the implemented axis lane selection reuses that policy rather than choosing a parallel source.
- Axis refinement execution lives in `RefractionFSMEngine` state `E/H`, with lane metadata coming from the DV layer and convergence/timeouts enforced in `state_transitions.py`.
- Axis hardware dispatch is handled in `SessionOrchestrator` by converting the finalized axis delta into repeated 5-degree JCC clicks.
- Session logging captures full runtime rows, lane metadata, and the selected step sequence for each eye, so axis behavior is auditable in both per-row logs and session metadata.
- The implemented refactor separates reversal-driven lane progression from the final safety flip stop, so longer lanes can still reach 5-degree refinement without exiting early.
