from dataclasses import asdict
from typing import List, Optional

from fsm.charts.chart_scale import (
    get_chart_index,
    get_next_chart,
    target_line_to_chart,
)
from fsm.engines.delta_calculators import (
    binocular_balance_delta,
    clamp_cyl_delta_at_zero,
    coarse_sphere_delta,
    duochrome_sphere_delta,
    jcc_axis_delta,
    jcc_power_cyl_delta,
    jcc_power_sphere_compensation,
    near_add_delta,
    near_binoc_ou_delta,
    wrap_axis,
)
from fsm.engines.escalation_rules import (
    phase_timeout_reached,
    should_escalate_le,
    should_escalate_re,
)
from fsm.engines.state_transitions import (
    compute_next_state,
    compute_phase_max,
)
from fsm.models.fsm_runtime import FSMRuntimeRow


COMPACT_PROMPT_CONFIG = {
    "B": {
        "response_type": "clarity_3way",
        "question": "Distance letter chart. Read the letters, or say clear, blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "D": {
        "response_type": "clarity_3way",
        "question": "Distance letter chart. Read the letters, or say clear, blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "E": {
        "response_type": "comparison_4way",
        "question": "Dot chart for axis. Say first, second, same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "F": {
        "response_type": "comparison_4way",
        "question": "Dot chart for power. Say first, second, same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "G": {
        "response_type": "duochrome_4way",
        "question": "Red-green chart. Say red, green, same, or repeat.",
        "options": ("RED", "GREEN", "SAME", "REPEAT"),
    },
    "H": {
        "response_type": "comparison_4way",
        "question": "Dot chart for axis. Say first, second, same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "I": {
        "response_type": "comparison_4way",
        "question": "Dot chart for power. Say first, second, same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "J": {
        "response_type": "duochrome_4way",
        "question": "Red-green chart. Say red, green, same, or repeat.",
        "options": ("RED", "GREEN", "SAME", "REPEAT"),
    },
    "K": {
        "response_type": "distance_bino_4way",
        "question": "Top-bottom balance chart. Say top, bottom, same, or repeat.",
        "options": ("TOP", "BOTTOM", "SAME", "REPEAT"),
    },
    "P": {
        "response_type": "clarity_3way",
        "question": "Near text chart. Say clear, blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "Q": {
        "response_type": "clarity_3way",
        "question": "Near text chart. Say clear, blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "R": {
        "response_type": "clarity_3way",
        "question": "Near text with both eyes. Say clear, blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
}


class RefractionFSMEngine:
    def __init__(self, calibration):
        self.cal = calibration
        self._re_coarse_entry_chart = {}

    def _fog_confirm_required(self, conf: str) -> int:
        if conf == "Low":
            return int(self.cal.get("fog_confirm_low", 1))
        if conf == "Medium":
            return int(self.cal.get("fog_confirm_medium", 2))
        return int(self.cal.get("fog_confirm_high", 3))

    def _power_same_required(self, conf: str) -> int:
        if conf == "Low":
            return int(self.cal.get("jcc_power_same_low", 1))
        if conf == "Medium":
            return int(self.cal.get("jcc_power_same_medium", 2))
        return int(self.cal.get("jcc_power_same_high", 2))

    def _axis_fixed_step(self) -> float:
        return float(self.cal.get("axis_fixed_step", 5))

    def _normalize_response_value(self, state: str, response_value: str) -> str:
        value = str(response_value or "").strip().upper()
        if not value:
            return value

        if state in ("B", "D", "P", "Q"):
            mapping = {
                "CLEAR": "CLEAR",
                "READABLE": "CLEAR",
                "BLURRY": "BLURRY",
                "REPEAT": "REPEAT",
                "NOT_READABLE": "REPEAT",
                "NOT_VISIBLE": "REPEAT",
                "CANT_TELL": "REPEAT",
            }
            return mapping.get(value, value)

        if state in ("E", "F", "H", "I"):
            mapping = {
                "ONE": "ONE",
                "BETTER_1": "ONE",
                "TWO": "TWO",
                "BETTER_2": "TWO",
                "SAME": "SAME",
                "REPEAT": "REPEAT",
                "CANT_TELL": "REPEAT",
            }
            return mapping.get(value, value)

        if state in ("G", "J"):
            mapping = {
                "RED": "RED",
                "RED_CLEARER": "RED",
                "GREEN": "GREEN",
                "GREEN_CLEARER": "GREEN",
                "SAME": "SAME",
                "EQUAL": "SAME",
                "REPEAT": "REPEAT",
                "CANT_TELL": "REPEAT",
            }
            return mapping.get(value, value)

        if state == "K":
            mapping = {
                "TOP": "TOP",
                "TOP_CLEARER": "TOP",
                "BOTTOM": "BOTTOM",
                "BOTTOM_CLEARER": "BOTTOM",
                "SAME": "SAME",
                "REPEAT": "REPEAT",
                "CANT_TELL": "REPEAT",
            }
            return mapping.get(value, value)

        if state == "R":
            mapping = {
                "CLEAR": "CLEAR",
                "TARGET_OK": "CLEAR",
                "SAME": "CLEAR",
                "BLURRY": "BLURRY",
                "NOT_CLEAR": "BLURRY",
                "NEED_ADJUSTMENT": "BLURRY",
                "NOT_COMFORTABLE": "BLURRY",
                "REPEAT": "REPEAT",
                "CANT_TELL": "REPEAT",
            }
            return mapping.get(value, value)

        return value

    def _row_for_state(
        self,
        step: int,
        visit_id: str,
        state: str,
        dv,
        re_sph: Optional[float],
        re_cyl: Optional[float],
        re_axis: Optional[float],
        le_sph: Optional[float],
        le_cyl: Optional[float],
        le_axis: Optional[float],
        add_r: Optional[float],
        add_l: Optional[float],
        chart_param: str,
        phase_step_count: int,
        same_streak: int,
        prev_axis_response: str,
        duo_iter: int,
        duo_flip: int,
        axis_step: float,
        axis_flip_count: int = 0,
        jcc_power_start_re_cyl: Optional[float] = None,
        jcc_power_start_le_cyl: Optional[float] = None,
        near_bino_start_add_r: Optional[float] = None,
        near_bino_start_add_l: Optional[float] = None,
        near_bino_direction: str = "",
        near_bino_reversed: bool = False,
        fog_active: bool = False,
        fog_start_re_sph: Optional[float] = None,
        fog_start_le_sph: Optional[float] = None,
    ) -> FSMRuntimeRow:
        target_chart = target_line_to_chart(dv.dv_target_distance_va, self.cal)
        target_chart_idx = get_chart_index(target_chart)
        if target_chart_idx == -1:
            target_chart = "20_20_20" if str(dv.dv_target_distance_va).strip() == "6/6_target" else "40_30_25"
            target_chart_idx = get_chart_index(target_chart)

        row = FSMRuntimeRow(
            step=step,
            visit_id=visit_id,
            state=state,
            phase_name="",
            phase_type="",
            stimulus_type="",
            chart_type="",
            response_type="",
            response_value="",
            opt_1="",
            opt_2="",
            opt_3="",
            opt_4="",
            opt_5="",
            opt_6="",
            question="",
            eye="",
            chart_param=str(chart_param),
            re_sph=re_sph,
            re_cyl=re_cyl,
            re_axis=re_axis,
            le_sph=le_sph,
            le_cyl=le_cyl,
            le_axis=le_axis,
            add_r=add_r,
            add_l=add_l,
            target_line=dv.dv_target_distance_va,
            sph_step=float(self.cal.get(f"{str(dv.dv_step_size_policy).lower()}_sph_step", 0.25)),
            cyl_step=float(self.cal.get(f"{str(dv.dv_step_size_policy).lower()}_cyl_step", 0.25)),
            axis_step=axis_step,
            fog_policy=dv.dv_fogging_policy,
            fog_amount=float(dv.dv_fogging_amount_D),
            fog_clearance_mode=dv.dv_fogging_clearance_mode,
            fog_confirm_required=self._fog_confirm_required(dv.dv_confidence_requirement),
            dv_confidence_requirement=dv.dv_confidence_requirement,
            dv_requires_optom_review=bool(dv.dv_requires_optom_review),
            dv_expected_convergence_time=dv.dv_expected_convergence_time,
            dv_branching_guardrails=dv.dv_branching_guardrails,
            dv_near_test_required=bool(getattr(dv, "dv_near_test_required", False)),
            dv_add_expected=getattr(dv, "dv_add_expected", None),
            dv_accommodation_level=getattr(dv, "dv_accommodation_level", "Unknown"),
            dv_fogging_required=bool(getattr(dv, "dv_fogging_required", False)),
            dv_fogging_stop_at_target_va=bool(getattr(dv, "dv_fogging_stop_at_target_va", True)),
            skip_bino_balance=False,
            fog_active=fog_active,
            fog_start_re_sph=fog_start_re_sph,
            fog_start_le_sph=fog_start_le_sph,
            ds_re=0.0,
            dc_re=0.0,
            da_re=0.0,
            ds_le=0.0,
            dc_le=0.0,
            da_le=0.0,
            dadd_r=0.0,
            dadd_l=0.0,
            next_chart_param=get_next_chart(str(chart_param)),
            chart_idx=get_chart_index(str(chart_param)),
            target_chart_idx=target_chart_idx,
            same_streak=same_streak,
            phase_step_count=phase_step_count,
            prev_axis_response=prev_axis_response,
            duo_iter=duo_iter,
            duo_flip=duo_flip,
            next_state=state,
            row_active=True,
            axis_flip_count=axis_flip_count,
            jcc_power_start_re_cyl=jcc_power_start_re_cyl,
            jcc_power_start_le_cyl=jcc_power_start_le_cyl,
            near_bino_start_add_r=near_bino_start_add_r,
            near_bino_start_add_l=near_bino_start_add_l,
            near_bino_direction=near_bino_direction,
            near_bino_reversed=near_bino_reversed,
        )
        
        if state == "B":
            row.phase_name = "Coarse Sphere RE"
            row.phase_type = "COARSE_SPHERE"
            row.stimulus_type = "COARSE_SPH"
            row.chart_type = "SNELLEN_FEET"
            row.response_type = COMPACT_PROMPT_CONFIG["B"]["response_type"]
            row.eye = "RE"
            row.question = COMPACT_PROMPT_CONFIG["B"]["question"]
            row.opt_1, row.opt_2, row.opt_3 = COMPACT_PROMPT_CONFIG["B"]["options"]

        elif state == "D":
            row.phase_name = "Coarse Sphere LE"
            row.phase_type = "COARSE_SPHERE"
            row.stimulus_type = "COARSE_SPH"
            row.chart_type = "SNELLEN_FEET"
            row.response_type = COMPACT_PROMPT_CONFIG["D"]["response_type"]
            row.eye = "LE"
            row.question = COMPACT_PROMPT_CONFIG["D"]["question"]
            row.opt_1, row.opt_2, row.opt_3 = COMPACT_PROMPT_CONFIG["D"]["options"]

        elif state == "E":
            row.phase_name = "JCC Axis RE"
            row.phase_type = "JCC_AXIS"
            row.stimulus_type = "JCC_AXIS"
            row.chart_type = "DOT_CHART_JCC"
            row.response_type = COMPACT_PROMPT_CONFIG["E"]["response_type"]
            row.eye = "RE"
            row.question = COMPACT_PROMPT_CONFIG["E"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["E"]["options"]

        elif state == "F":
            row.phase_name = "JCC Power RE"
            row.phase_type = "JCC_POWER"
            row.stimulus_type = "JCC_POWER"
            row.chart_type = "DOT_CHART_JCC"
            row.response_type = COMPACT_PROMPT_CONFIG["F"]["response_type"]
            row.eye = "RE"
            row.question = COMPACT_PROMPT_CONFIG["F"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["F"]["options"]

        elif state == "G":
            row.phase_name = "Duochrome RE"
            row.phase_type = "DUOCHROME"
            row.stimulus_type = "DUOCHROME"
            row.chart_type = "RED_GREEN_DUOCHROME"
            row.response_type = COMPACT_PROMPT_CONFIG["G"]["response_type"]
            row.eye = "RE"
            row.question = COMPACT_PROMPT_CONFIG["G"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["G"]["options"]

        elif state == "H":
            row.phase_name = "JCC Axis LE"
            row.phase_type = "JCC_AXIS"
            row.stimulus_type = "JCC_AXIS"
            row.chart_type = "DOT_CHART_JCC"
            row.response_type = COMPACT_PROMPT_CONFIG["H"]["response_type"]
            row.eye = "LE"
            row.question = COMPACT_PROMPT_CONFIG["H"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["H"]["options"]

        elif state == "I":
            row.phase_name = "JCC Power LE"
            row.phase_type = "JCC_POWER"
            row.stimulus_type = "JCC_POWER"
            row.chart_type = "DOT_CHART_JCC"
            row.response_type = COMPACT_PROMPT_CONFIG["I"]["response_type"]
            row.eye = "LE"
            row.question = COMPACT_PROMPT_CONFIG["I"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["I"]["options"]

        elif state == "J":
            row.phase_name = "Duochrome LE"
            row.phase_type = "DUOCHROME"
            row.stimulus_type = "DUOCHROME"
            row.chart_type = "RED_GREEN_DUOCHROME"
            row.response_type = COMPACT_PROMPT_CONFIG["J"]["response_type"]
            row.eye = "LE"
            row.question = COMPACT_PROMPT_CONFIG["J"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["J"]["options"]

        elif state == "K":
            row.phase_name = "Binocular Balance"
            row.phase_type = "BINOC_BALANCE"
            row.stimulus_type = "BINOC_BALANCE"
            row.chart_type = "POLARIZED_BALANCE"
            row.response_type = COMPACT_PROMPT_CONFIG["K"]["response_type"]
            row.eye = "BIN"
            row.question = COMPACT_PROMPT_CONFIG["K"]["question"]
            row.opt_1, row.opt_2, row.opt_3, row.opt_4 = COMPACT_PROMPT_CONFIG["K"]["options"]

        elif state == "P":
            row.phase_name = "Near Add RE"
            row.phase_type = "NEAR_ADD_RE"
            row.stimulus_type = "NEAR_ADD"
            row.chart_type = "NEAR_CHART"
            row.response_type = COMPACT_PROMPT_CONFIG["P"]["response_type"]
            row.eye = "RE"
            row.question = COMPACT_PROMPT_CONFIG["P"]["question"]
            row.opt_1, row.opt_2, row.opt_3 = COMPACT_PROMPT_CONFIG["P"]["options"]

        elif state == "Q":
            row.phase_name = "Near Add LE"
            row.phase_type = "NEAR_ADD_LE"
            row.stimulus_type = "NEAR_ADD"
            row.chart_type = "NEAR_CHART"
            row.response_type = COMPACT_PROMPT_CONFIG["Q"]["response_type"]
            row.eye = "LE"
            row.question = COMPACT_PROMPT_CONFIG["Q"]["question"]
            row.opt_1, row.opt_2, row.opt_3 = COMPACT_PROMPT_CONFIG["Q"]["options"]

        elif state == "R":
            row.phase_name = "Near Binocular"
            row.phase_type = "NEAR_BINOC"
            row.stimulus_type = "NEAR_BINOC"
            row.chart_type = "NEAR_CHART"
            row.response_type = COMPACT_PROMPT_CONFIG["R"]["response_type"]
            row.eye = "BIN"
            row.question = COMPACT_PROMPT_CONFIG["R"]["question"]
            row.opt_1, row.opt_2, row.opt_3 = COMPACT_PROMPT_CONFIG["R"]["options"]

        return row

    def initialize_row(self, visit_id: str, dv, ar_re=None, ar_le=None) -> FSMRuntimeRow:
        coarse_start_chart = str(self.cal.get("coarse_start_chart", self.cal.get("dist_chart_2", "200_150")))
        self._re_coarse_entry_chart[visit_id] = coarse_start_chart

        re_start_sph = dv.dv_start_rx_RE_sph
        if getattr(dv, "dv_fogging_required", False) and re_start_sph is not None:
            re_start_sph = float(re_start_sph) + float(dv.dv_fogging_amount_D or 0.0)

        fog_active = False
        fog_start_re_sph = None
        if getattr(dv, "dv_fogging_required", False) and re_start_sph is not None:
            fog_active = True
            fog_start_re_sph = re_start_sph

        return self._row_for_state(
            step=1,
            visit_id=visit_id,
            state="B",
            dv=dv,
            re_sph=re_start_sph,
            re_cyl=dv.dv_start_rx_RE_cyl,
            re_axis=dv.dv_start_rx_RE_axis,
            le_sph=dv.dv_start_rx_LE_sph,
            le_cyl=dv.dv_start_rx_LE_cyl,
            le_axis=dv.dv_start_rx_LE_axis,
            add_r=0.0,
            add_l=0.0,
            chart_param=coarse_start_chart,
            phase_step_count=0,
            same_streak=0,
            prev_axis_response="",
            duo_iter=0,
            duo_flip=0,
            axis_step=self._axis_fixed_step(),
            axis_flip_count=0,
            jcc_power_start_re_cyl=None,
            jcc_power_start_le_cyl=None,
            near_bino_start_add_r=None,
            near_bino_start_add_l=None,
            near_bino_direction="",
            near_bino_reversed=False,
            fog_active=fog_active,
            fog_start_re_sph=fog_start_re_sph,
        )

    def _next_phase_step_count(self, current_row: FSMRuntimeRow, next_state: str) -> int:
        if next_state != current_row.state:
            return 1
        if current_row.phase_step_count == 0:
            return 2
        return current_row.phase_step_count + 1

    def _next_chart_param_from_row(self, row: FSMRuntimeRow) -> str:

        if row.state == "B":
            if row.response_value == "CLEAR" and row.next_state == "B":
                return row.next_chart_param
            return row.chart_param

        if row.state == "D":
            if row.response_value == "CLEAR" and row.next_state == "D":
                return row.next_chart_param
            return row.chart_param

        return row.chart_param

    def _build_next_row(self, row: FSMRuntimeRow, dv) -> Optional[FSMRuntimeRow]:
        if row.next_state in ("END", "ESCALATE"):
            return None

        next_chart_param = self._next_chart_param_from_row(row)
        next_same_streak = row.same_streak if row.next_state == row.state else 0

        if row.next_state == row.state:
            if row.state in ("E", "H"):
                next_prev_axis_response = row.response_value
            else:
                next_prev_axis_response = row.prev_axis_response
        else:
            next_prev_axis_response = ""

        next_duo_iter = row.duo_iter if row.next_state == row.state else 0
        next_duo_flip = row.duo_flip if row.next_state == row.state else 0
        next_axis_step = row.axis_step if row.next_state == row.state else self._axis_fixed_step()
        next_axis_flip_count = row.axis_flip_count if row.next_state == row.state else 0
        next_jcc_power_start_re_cyl = row.jcc_power_start_re_cyl if row.next_state == row.state else None
        next_jcc_power_start_le_cyl = row.jcc_power_start_le_cyl if row.next_state == row.state else None

        next_re_sph = (row.re_sph or 0.0) + (row.ds_re or 0.0)
        next_re_cyl = (row.re_cyl or 0.0) + (row.dc_re or 0.0)
        next_re_axis = wrap_axis((row.re_axis or 0.0) + (row.da_re or 0.0)) if row.re_axis is not None else row.re_axis

        next_le_sph = (row.le_sph or 0.0) + (row.ds_le or 0.0)
        next_le_cyl = (row.le_cyl or 0.0) + (row.dc_le or 0.0)
        next_le_axis = wrap_axis((row.le_axis or 0.0) + (row.da_le or 0.0)) if row.le_axis is not None else row.le_axis

        next_add_r = (row.add_r or 0.0) + (row.dadd_r or 0.0)
        next_add_l = (row.add_l or 0.0) + (row.dadd_l or 0.0)
        next_fog_active = row.fog_active
        next_fog_start_re_sph = row.fog_start_re_sph
        next_fog_start_le_sph = row.fog_start_le_sph

        next_near_bino_start_add_r = row.near_bino_start_add_r if row.next_state == row.state else None
        next_near_bino_start_add_l = row.near_bino_start_add_l if row.next_state == row.state else None
        next_near_bino_direction = row.near_bino_direction if row.next_state == row.state else ""
        next_near_bino_reversed = row.near_bino_reversed if row.next_state == row.state else False

        if row.next_state == "B" and row.state != "B":
            next_chart_param = str(self.cal.get("coarse_start_chart", self.cal.get("dist_chart_2", "200_150")))
            self._re_coarse_entry_chart[row.visit_id] = next_chart_param

            base_sph = dv.dv_start_rx_RE_sph or 0.0
            if getattr(dv, "dv_fogging_required", False):
                base_sph += float(dv.dv_fogging_amount_D or 0.0)

            next_re_sph = base_sph
            next_re_cyl = dv.dv_start_rx_RE_cyl
            next_re_axis = dv.dv_start_rx_RE_axis

        if row.next_state == "D" and row.state != "D":
            entry_chart = self._re_coarse_entry_chart.get(row.visit_id)
            if not entry_chart:
                entry_chart = str(self.cal.get("coarse_start_chart", self.cal.get("dist_chart_2", "200_150")))

            base_sph = dv.dv_start_rx_LE_sph or 0.0
            if getattr(dv, "dv_fogging_required", False):
                base_sph += float(dv.dv_fogging_amount_D or 0.0)
                next_fog_active = True
                next_fog_start_le_sph = base_sph

            next_le_sph = base_sph
            next_le_cyl = dv.dv_start_rx_LE_cyl
            next_le_axis = dv.dv_start_rx_LE_axis
            next_chart_param = str(entry_chart)

        if row.next_state == "R" and row.state != "R":
            next_near_bino_start_add_r = next_add_r
            next_near_bino_start_add_l = next_add_l
            next_near_bino_direction = ""
            next_near_bino_reversed = False

        if row.next_state == "F" and row.state != "F":
            next_jcc_power_start_re_cyl = next_re_cyl

        if row.next_state == "I" and row.state != "I":
            next_jcc_power_start_le_cyl = next_le_cyl

        next_row = self._row_for_state(
            step=row.step + 1,
            visit_id=row.visit_id,
            state=row.next_state,
            dv=dv,
            re_sph=next_re_sph,
            re_cyl=next_re_cyl,
            re_axis=next_re_axis,
            le_sph=next_le_sph,
            le_cyl=next_le_cyl,
            le_axis=next_le_axis,
            add_r=next_add_r,
            add_l=next_add_l,
            chart_param=next_chart_param,
            phase_step_count=self._next_phase_step_count(row, row.next_state),
            same_streak=next_same_streak,
            prev_axis_response=next_prev_axis_response,
            duo_iter=next_duo_iter,
            duo_flip=next_duo_flip,
            axis_step=next_axis_step,
            axis_flip_count=next_axis_flip_count,
            jcc_power_start_re_cyl=next_jcc_power_start_re_cyl,
            jcc_power_start_le_cyl=next_jcc_power_start_le_cyl,
            near_bino_start_add_r=next_near_bino_start_add_r,
            near_bino_start_add_l=next_near_bino_start_add_l,
            near_bino_direction=next_near_bino_direction,
            near_bino_reversed=next_near_bino_reversed,
            fog_active=next_fog_active,
            fog_start_re_sph=next_fog_start_re_sph,
            fog_start_le_sph=next_fog_start_le_sph,
        )

        if next_row.state in ("F", "I", "G", "J", "K") and next_row.state == row.state:
            next_row.response_value = row.response_value

        return next_row

    def apply_response(self, current: FSMRuntimeRow, response_value: str, dv, ar_re=None, ar_le=None) -> FSMRuntimeRow:
        row = FSMRuntimeRow(**asdict(current))
        normalized_response = self._normalize_response_value(current.state, response_value)
        row.response_value = normalized_response

        row.ds_re = 0.0
        row.dc_re = 0.0
        row.da_re = 0.0
        row.ds_le = 0.0
        row.dc_le = 0.0
        row.da_le = 0.0
        row.dadd_r = 0.0
        row.dadd_l = 0.0

        row.chart_idx = get_chart_index(str(current.chart_param))
        row.target_chart_idx = get_chart_index(target_line_to_chart(dv.dv_target_distance_va, self.cal))
        row.next_chart_param = get_next_chart(str(current.chart_param))
        coarse_endpoint_reached = False
        coarse_oscillation_detected = False

        if current.state == "B":
            row.ds_re = coarse_sphere_delta(normalized_response, current.sph_step)
            if normalized_response == "CLEAR":
                row.next_chart_param = get_next_chart(str(current.chart_param))
            else:
                row.next_chart_param = str(current.chart_param)
            if current.chart_idx >= row.target_chart_idx and normalized_response == "CLEAR":
                coarse_endpoint_reached = True

        elif current.state == "D":
            row.ds_le = coarse_sphere_delta(normalized_response, current.sph_step)
            if normalized_response == "CLEAR":
                row.next_chart_param = get_next_chart(str(current.chart_param))
            else:
                row.next_chart_param = str(current.chart_param)
            if current.chart_idx >= row.target_chart_idx and normalized_response == "CLEAR":
                coarse_endpoint_reached = True

        elif current.state == "E":
            row.da_re = jcc_axis_delta(normalized_response, current.axis_step, positive_for_better_1=True)

        elif current.state == "H":
            row.da_le = jcc_axis_delta(normalized_response, current.axis_step, positive_for_better_1=True)

        elif current.state == "F":
            proposed_dc = jcc_power_cyl_delta(normalized_response, current.cyl_step)
            row.dc_re = clamp_cyl_delta_at_zero(current.re_cyl, proposed_dc)
            row.ds_re = jcc_power_sphere_compensation(
                current_cyl=current.re_cyl,
                proposed_cyl_delta=row.dc_re,
                start_cyl=current.jcc_power_start_re_cyl,
            )

        elif current.state == "I":
            proposed_dc = jcc_power_cyl_delta(normalized_response, current.cyl_step)
            row.dc_le = clamp_cyl_delta_at_zero(current.le_cyl, proposed_dc)
            row.ds_le = jcc_power_sphere_compensation(
                current_cyl=current.le_cyl,
                proposed_cyl_delta=row.dc_le,
                start_cyl=current.jcc_power_start_le_cyl,
            )

        elif current.state == "G":
            equal_reached = False
            if normalized_response == "SAME":
                equal_reached = (current.same_streak + 1) >= int(self.cal.get("duochrome_equal_confirmations", 2))

            row.ds_re = duochrome_sphere_delta(
                response_value=normalized_response,
                endpoint_bias_policy=dv.dv_endpoint_bias_policy,
                equal_confirmation_reached=equal_reached,
                calibration=self.cal,
            )

        elif current.state == "J":
            equal_reached = False
            if normalized_response == "SAME":
                equal_reached = (current.same_streak + 1) >= int(self.cal.get("duochrome_equal_confirmations", 2))

            row.ds_le = duochrome_sphere_delta(
                response_value=normalized_response,
                endpoint_bias_policy=dv.dv_endpoint_bias_policy,
                equal_confirmation_reached=equal_reached,
                calibration=self.cal,
            )

        elif current.state == "K":
            row.ds_re = binocular_balance_delta(normalized_response, self.cal, top_eye=True)
            row.ds_le = binocular_balance_delta(normalized_response, self.cal, top_eye=False)

        elif current.state == "P":
            row.dadd_r = near_add_delta(normalized_response, self.cal)

        elif current.state == "Q":
            row.dadd_l = near_add_delta(normalized_response, self.cal)

        elif current.state == "R":
            step = float(dv.dv_near_binoc_step_D)

            if normalized_response == "CLEAR":
                row.dadd_r = 0.0
                row.dadd_l = 0.0

            elif current.near_bino_direction == "":
                # First failure at binocular near baseline: try +0.25 OU
                row.near_bino_start_add_r = current.add_r if current.add_r is not None else 0.0
                row.near_bino_start_add_l = current.add_l if current.add_l is not None else 0.0
                row.near_bino_direction = "PLUS"
                row.near_bino_reversed = False
                row.dadd_r, row.dadd_l = near_binoc_ou_delta("PLUS", step)

            elif current.near_bino_direction == "PLUS" and normalized_response == "BLURRY":
                # PLUS trial failed: revert to baseline and start MINUS search
                start_r = current.near_bino_start_add_r if current.near_bino_start_add_r is not None else 0.0
                start_l = current.near_bino_start_add_l if current.near_bino_start_add_l is not None else 0.0

                target_r = start_r - step
                target_l = start_l - step

                row.dadd_r = target_r - (current.add_r if current.add_r is not None else 0.0)
                row.dadd_l = target_l - (current.add_l if current.add_l is not None else 0.0)

                row.near_bino_direction = "MINUS"
                row.near_bino_reversed = True

            elif current.near_bino_direction == "MINUS" and normalized_response == "BLURRY":
                # Continue reducing binocularly until target is OK
                row.dadd_r, row.dadd_l = near_binoc_ou_delta("MINUS", step)
                row.near_bino_direction = "MINUS"
                row.near_bino_reversed = True

        if current.state in ("G", "J"):
            if normalized_response == "SAME":
                row.same_streak = current.same_streak + 1 if current.same_streak > 0 else 1
            else:
                row.same_streak = 0
        else:
            if normalized_response == "SAME":
                row.same_streak = current.same_streak + 1 if current.same_streak > 0 else 1
            else:
                row.same_streak = 0

        row.prev_axis_response = current.prev_axis_response

        if current.state in ("G", "J"):
            row.duo_iter = current.duo_iter + 1
        else:
            row.duo_iter = 0

        if current.state in ("F", "I", "G", "J", "K"):
            flip = False

            if current.state in ("F", "I"):
                flip = (
                    (current.response_value == "ONE" and normalized_response == "TWO")
                    or (current.response_value == "TWO" and normalized_response == "ONE")
                )
            elif current.state in ("G", "J"):
                flip = (
                    (current.response_value == "RED" and normalized_response == "GREEN")
                    or (current.response_value == "GREEN" and normalized_response == "RED")
                )
            elif current.state == "K":
                flip = (
                    (current.response_value == "TOP" and normalized_response == "BOTTOM")
                    or (current.response_value == "BOTTOM" and normalized_response == "TOP")
                )

            row.duo_flip = current.duo_flip + 1 if flip else current.duo_flip
        else:
            row.duo_flip = 0

        if current.state in ("E", "H"):
            row.axis_step = self._axis_fixed_step()

            axis_flip = (
                current.prev_axis_response in ("ONE", "TWO")
                and normalized_response in ("ONE", "TWO")
                and current.prev_axis_response != normalized_response
            )
            row.axis_flip_count = current.axis_flip_count + 1 if axis_flip else current.axis_flip_count
        else:
            row.axis_flip_count = 0

        re_escalate = should_escalate_re(
            anomaly_watch=bool(dv.dv_anomaly_watch),
            current_sph=float(current.re_sph or 0.0),
            start_sph=float(dv.dv_start_rx_RE_sph or 0.0),
            ar_sph=float(ar_re.sphere if ar_re and ar_re.sphere is not None else dv.dv_start_rx_RE_sph or 0.0),
            max_delta_from_start=float(dv.dv_max_delta_from_start_sph),
            max_delta_from_ar=float(dv.dv_max_delta_from_ar_sph),
            phase_step_count=int(current.phase_step_count or 0),
        )

        le_escalate = should_escalate_le(
            anomaly_watch=bool(dv.dv_anomaly_watch),
            current_sph=float(current.le_sph or 0.0),
            start_sph=float(dv.dv_start_rx_LE_sph or 0.0),
            ar_sph=float(ar_le.sphere if ar_le and ar_le.sphere is not None else dv.dv_start_rx_LE_sph or 0.0),
            max_delta_from_start=float(dv.dv_max_delta_from_start_sph),
            max_delta_from_ar=float(dv.dv_max_delta_from_ar_sph),
            phase_step_count=int(current.phase_step_count or 0),
        )

        phase_max = compute_phase_max(current.state, dv.dv_expected_convergence_time, self.cal)
        timeout = phase_timeout_reached(current.phase_step_count, phase_max)

        power_same_n = self._power_same_required(dv.dv_confidence_requirement)
        duo_equal_n = int(self.cal.get("duochrome_equal_confirmations", 2))
        bino_same_n = int(self.cal.get("bino_balance_same_required", 1))
        near_target = self._normalize_response_value("R", str(self.cal.get("near_binoc_target_response", "CLEAR")))

        # Existing FSMv2.2 logic retained
        bino_flip_limit = 1
        skip_bino_balance = abs((current.re_sph or 0.0) - (current.le_sph or 0.0)) <= 0.25
        row.skip_bino_balance = skip_bino_balance

        context = {
            "state": current.state,
            "response": normalized_response,
            "timeout": timeout,
            "re_escalate": re_escalate,
            "le_escalate": le_escalate,
            "axis_tolerance": float(dv.dv_axis_tolerance_deg),
            "axis_step": float(row.axis_step),
            "same_streak": int(row.same_streak),
            "power_same_n": int(power_same_n),
            "duo_equal_n": int(duo_equal_n),
            "duo_flip": int(row.duo_flip),
            "duo_max": int(dv.dv_duochrome_max_flips),
            "bino_same_n": int(bino_same_n),
            "bino_flip": int(row.duo_flip) if current.state == "K" else 0,
            "bino_flip_limit": int(bino_flip_limit),
            "skip_bino_balance": bool(skip_bino_balance),
            "near_target": near_target,
            "near_required": bool(dv.dv_near_test_required),
            "immediate_review": bool(dv.dv_requires_optom_review),
            "chart_param": current.chart_param,
            "target_chart_param": target_line_to_chart(dv.dv_target_distance_va, self.cal),
            "chart_idx": get_chart_index(str(current.chart_param)),
            "target_chart_idx": get_chart_index(target_line_to_chart(dv.dv_target_distance_va, self.cal)),
            "coarse_endpoint_reached": coarse_endpoint_reached,
            "coarse_oscillation_detected": coarse_oscillation_detected,
            "jcc_power_flip_limit_hit": row.duo_flip >= int(self.cal.get("jcc_power_max_flips", 4)),
            "jcc_cyl_at_zero": (
                (current.state == "F" and abs(((current.re_cyl or 0.0) + (row.dc_re or 0.0))) < 1e-9)
                or (current.state == "I" and abs(((current.le_cyl or 0.0) + (row.dc_le or 0.0))) < 1e-9)
            ),

            # FSM v2.3 additions
            "axis_same_required": int(dv.dv_jcc_axis_same_required),
            "axis_flip_count": int(row.axis_flip_count),
            "axis_flip_max": int(dv.dv_jcc_axis_max_flips),

            "near_binoc_direction": row.near_bino_direction,
            "near_binoc_reversed": bool(row.near_bino_reversed),
            "near_binoc_max_plus_steps": int(dv.dv_near_binoc_max_plus_steps),
            "near_binoc_max_minus_steps": int(dv.dv_near_binoc_max_minus_steps),
        }

        row.next_state = compute_next_state(context)
        row.row_active = True

        return row

    def run_visit(
        self,
        visit_id: str,
        dv,
        responses: List[str],
        ar_re=None,
        ar_le=None,
        max_steps: int = 200,
    ) -> List[dict]:
        rows = []
        current = self.initialize_row(
            visit_id=visit_id,
            dv=dv,
            ar_re=ar_re,
            ar_le=ar_le,
        )

        idx = 0
        while idx < len(responses) and len(rows) < max_steps:
            response = responses[idx]

            finalized = self.apply_response(
                current=current,
                response_value=response,
                dv=dv,
                ar_re=ar_re,
                ar_le=ar_le,
            )

            rows.append(asdict(finalized))

            if finalized.next_state in ("END", "ESCALATE"):
                break

            next_row = self._build_next_row(finalized, dv)
            if next_row is None:
                break

            current = next_row
            idx += 1

        return rows
