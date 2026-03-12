from dataclasses import dataclass
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
    duo_iter: int = 0
    duo_flip: int = 0
    next_state: str = ""
    row_active: bool = True
