from __future__ import annotations
from dataclasses import dataclass

@dataclass
class DerivedVariables:
    visit_id: str
    dv_age_bucket: str
    dv_distance_priority: str
    dv_near_priority: str
    dv_symptom_risk_level: str
    dv_medical_risk_level: str
    dv_stability_level: str
    dv_ar_lenso_mismatch_level_RE: str
    dv_ar_lenso_mismatch_level_LE: str
    dv_start_source_policy: str
    dv_start_rx_RE_sph: float | None
    dv_start_rx_RE_cyl: float | None
    dv_start_rx_RE_axis: float | None
    dv_start_rx_LE_sph: float | None
    dv_start_rx_LE_cyl: float | None
    dv_start_rx_LE_axis: float | None
    dv_add_expected: str
    dv_target_distance_va: str
    dv_endpoint_bias_policy: str
    dv_step_size_policy: str
    dv_max_delta_from_start_sph: float
    dv_max_delta_from_ar_sph: float
    dv_axis_tolerance_deg: float
    dv_cyl_tolerance_D: float
    dv_requires_optom_review: bool
    dv_anomaly_watch: bool
    dv_expected_convergence_time: str
    dv_branching_guardrails: str
    dv_confidence_requirement: str
    dv_fogging_policy: str
    dv_fogging_amount_D: float
    dv_fogging_clearance_mode: str
    dv_fogging_required_confirmation: str
    dv_axis_step_policy: str
    dv_duochrome_max_flips: int
    dv_near_test_required: bool

    # FSM v2.3 additions (defaults prevent breaking existing scripts)

    dv_jcc_axis_same_required: int = 2
    dv_jcc_axis_max_flips: int = 3

    dv_near_binoc_step_D: float = 0.25
    dv_near_binoc_max_plus_steps: int = 4
    dv_near_binoc_max_minus_steps: int = 4
