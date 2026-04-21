from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FSMRuntimeRow:
    step: int
    visit_id: str
    state: str

    phase_name: str
    phase_type: str
    stimulus_type: str
    chart_type: str
    response_type: str

    response_value: str
    opt_1: str = ""
    opt_2: str = ""
    opt_3: str = ""
    opt_4: str = ""
    opt_5: str = ""
    opt_6: str = ""
    question: str = ""
    eye: str = ""

    chart_param: str = ""

    re_sph: Optional[float] = None
    re_cyl: Optional[float] = None
    re_axis: Optional[float] = None

    le_sph: Optional[float] = None
    le_cyl: Optional[float] = None
    le_axis: Optional[float] = None

    add_r: Optional[float] = None
    add_l: Optional[float] = None

    target_line: str = ""

    sph_step: float = 0.25
    cyl_step: float = 0.25
    axis_step: float = 5.0

    fog_policy: str = ""
    fog_amount: float = 0.0
    fog_clearance_mode: str = ""
    fog_confirm_required: int = 1

    dv_confidence_requirement: str = ""
    dv_requires_optom_review: bool = False
    dv_expected_convergence_time: str = ""
    dv_branching_guardrails: str = ""

    # DV / debugging visibility
    dv_near_test_required: bool = False
    dv_add_expected: str = ""
    skip_bino_balance: bool = False

    # FSM v3.1 fogging visibility
    dv_accommodation_level: str = "Unknown"
    dv_fogging_required: bool = False
    dv_fogging_stop_at_target_va: bool = True
    fog_active: bool = False

    ds_re: float = 0.0
    dc_re: float = 0.0
    da_re: float = 0.0

    ds_le: float = 0.0
    dc_le: float = 0.0
    da_le: float = 0.0

    dadd_r: float = 0.0
    dadd_l: float = 0.0

    next_chart_param: str = ""
    chart_idx: int = 0
    target_chart_idx: int = 0

    same_streak: int = 0
    phase_step_count: int = 0
    prev_axis_response: str = ""
    prompt_memory: dict[str, int] = field(default_factory=dict)
    duo_iter: int = 0
    duo_flip: int = 0
    duo_same_anchor_response: str = ""
    coarse_compare_mode: bool = False
    coarse_recheck_mode: bool = False
    coarse_last_confirmed_chart_re: str = ""
    coarse_last_confirmed_chart_le: str = ""
    distance_va_re_chart: str = ""
    distance_va_le_chart: str = ""
    distance_va_re_line: str = ""
    distance_va_le_line: str = ""
    va_confirm_ceiling_chart: str = ""
    preface_prompt: str = ""
    final_compare_enabled: bool = False
    final_compare_round: int = 0
    final_compare_option_source: str = ""
    final_compare_current_source: str = ""
    final_compare_choice_round_1: str = ""
    final_compare_choice_round_2: str = ""
    patient_accepted_achieved_over_current_rx: str = ""

    final_compare_current_re_sph: Optional[float] = None
    final_compare_current_re_cyl: Optional[float] = None
    final_compare_current_re_axis: Optional[float] = None
    final_compare_current_le_sph: Optional[float] = None
    final_compare_current_le_cyl: Optional[float] = None
    final_compare_current_le_axis: Optional[float] = None
    final_compare_current_add_r: Optional[float] = None
    final_compare_current_add_l: Optional[float] = None

    final_compare_achieved_re_sph: Optional[float] = None
    final_compare_achieved_re_cyl: Optional[float] = None
    final_compare_achieved_re_axis: Optional[float] = None
    final_compare_achieved_le_sph: Optional[float] = None
    final_compare_achieved_le_cyl: Optional[float] = None
    final_compare_achieved_le_axis: Optional[float] = None
    final_compare_achieved_add_r: Optional[float] = None
    final_compare_achieved_add_l: Optional[float] = None

    next_state: str = ""
    row_active: bool = True

    # ---------------------------------------
    # FSM v2.3 additions
    # ---------------------------------------

    # JCC Axis tracking
    axis_flip_count: int = 0
    axis_quick_search_active: bool = False
    axis_quick_phase: str = ""
    axis_last_directional_response: str = ""
    axis_reversal_count: int = 0
    axis_step_index: int = 0
    axis_step_sequence: str = ""
    axis_lane_id: str = ""
    axis_lane_name: str = ""
    axis_confidence_label: str = ""
    axis_source_used: str = ""
    axis_selection_reason: str = ""
    axis_is_near_cardinal: bool = False
    axis_cyl_magnitude_for_lane: float = 0.0

    # JCC Power entry cylinder (used for relative cylinder displacement compensation)
    jcc_power_start_re_cyl: Optional[float] = None
    jcc_power_start_le_cyl: Optional[float] = None

    # Near binocular search tracking
    near_bino_start_add_r: Optional[float] = None
    near_bino_start_add_l: Optional[float] = None
    near_bino_direction: str = ""   # "PLUS" or "MINUS"
    near_bino_reversed: bool = False

    # FSM v3.1 fogging lifecycle tracking
    fog_start_re_sph: Optional[float] = None
    fog_start_le_sph: Optional[float] = None
