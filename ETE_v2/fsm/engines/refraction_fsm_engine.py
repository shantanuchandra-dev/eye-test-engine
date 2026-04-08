from dataclasses import asdict
from typing import List, Optional

from fsm.charts.chart_scale import (
    chart_to_last_line_va,
    get_chart_index,
    get_next_chart,
    get_previous_chart,
    normalize_chart_param,
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
        "question": "Please read the line. If the letters are not clear, say blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "D": {
        "response_type": "clarity_3way",
        "question": "Please read the line. If the letters are not clear, say blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "E": {
        "response_type": "comparison_4way",
        "question": "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "F": {
        "response_type": "comparison_4way",
        "question": "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "G": {
        "response_type": "duochrome_4way",
        "question": "Please compare the green and red sides. Letters on which side look sharper and darker? Say green side, red side, both same, or repeat.",
        "options": ("GREEN", "RED", "SAME", "REPEAT"),
    },
    "H": {
        "response_type": "comparison_4way",
        "question": "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "I": {
        "response_type": "comparison_4way",
        "question": "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat.",
        "options": ("ONE", "TWO", "SAME", "REPEAT"),
    },
    "J": {
        "response_type": "duochrome_4way",
        "question": "Please compare the green and red sides. Letters on which side look sharper and darker? Say green side, red side, both same, or repeat.",
        "options": ("GREEN", "RED", "SAME", "REPEAT"),
    },
    "K": {
        "response_type": "distance_bino_4way",
        "question": "Please compare the letters on the bottom and the top line. Which line looks sharper? Say bottom line, top line, both same, or repeat.",
        "options": ("BOTTOM", "TOP", "SAME", "REPEAT"),
    },
    "P": {
        "response_type": "clarity_3way",
        "question": "Please read the last line. Is it clear, blurry, or should I repeat?",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "Q": {
        "response_type": "clarity_3way",
        "question": "Please read the last line. Is it clear, blurry, or should I repeat?",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "R": {
        "response_type": "clarity_3way",
        "question": "Please read the last line. Is it clear, blurry, or should I repeat?",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "S": {
        "response_type": "observe_only",
        "question": "This is the final confirmation before I finish your eye test. This is the first option. Please observe the line carefully.",
        "options": (),
    },
    "T": {
        "response_type": "observe_only",
        "question": "Now this is the second option. Please observe the line carefully.",
        "options": (),
    },
    "U": {
        "response_type": "comparison_4way",
        "question": "Which option was better, first option or second option?",
        "options": ("ONE", "TWO", "REPEAT"),
    },
    "C": {
        "response_type": "clarity_3way",
        "question": "Please read the line. If the letters are not clear, say blurry, or repeat.",
        "options": ("CLEAR", "BLURRY", "REPEAT"),
    },
    "L": {
        "response_type": "clarity_3way",
        "question": "Please read the line. If the letters are not clear, say blurry, or repeat.",
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

    def _parse_axis_step_sequence_text(self, sequence_text: str) -> list[float]:
        sequence = []
        for token in str(sequence_text or "").replace("|", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError:
                continue
            if value > 0:
                sequence.append(value)
        return sequence or [self._axis_fixed_step()]

    def _axis_lane_suffix(self, state: str) -> str:
        if state == "E":
            return "RE"
        if state == "H":
            return "LE"
        return ""

    def _axis_lane_metadata_for_state(self, state: str, dv) -> dict:
        suffix = self._axis_lane_suffix(state)
        if not suffix:
            return {
                "lane_id": "",
                "lane_name": "",
                "confidence_label": "",
                "source_used": "",
                "selection_reason": "",
                "is_near_cardinal": False,
                "cyl_magnitude_for_lane": 0.0,
                "step_sequence": str(int(self._axis_fixed_step())),
            }
        return {
            "lane_id": getattr(dv, f"dv_axis_lane_id_{suffix}", ""),
            "lane_name": getattr(dv, f"dv_axis_lane_name_{suffix}", ""),
            "confidence_label": getattr(dv, f"dv_axis_confidence_label_{suffix}", ""),
            "source_used": getattr(dv, f"dv_axis_source_used_{suffix}", ""),
            "selection_reason": getattr(dv, f"dv_axis_selection_reason_{suffix}", ""),
            "is_near_cardinal": bool(getattr(dv, f"dv_axis_is_near_cardinal_{suffix}", False)),
            "cyl_magnitude_for_lane": float(getattr(dv, f"dv_axis_cyl_magnitude_for_lane_{suffix}", 0.0) or 0.0),
            "step_sequence": getattr(dv, f"dv_axis_step_sequence_{suffix}", ""),
        }

    def _axis_nominal_step(self, sequence_text: str, step_index: int) -> float:
        sequence = self._parse_axis_step_sequence_text(sequence_text)
        bounded_index = min(max(int(step_index), 0), len(sequence) - 1)
        return float(sequence[bounded_index])

    def _axis_converges_on_terminal_reversal(
        self,
        current: FSMRuntimeRow,
        dv,
        normalized_response: str,
    ) -> bool:
        if current.state not in ("E", "H"):
            return False
        if normalized_response not in ("ONE", "TWO"):
            return False
        last_directional = current.axis_last_directional_response
        if last_directional not in ("ONE", "TWO"):
            return False
        if normalized_response == last_directional:
            return False
        current_step = self._axis_nominal_step(current.axis_step_sequence, current.axis_step_index)
        tolerance = float(getattr(dv, "dv_axis_tolerance_deg", 10.0) or 10.0)
        return current_step <= tolerance + 1e-9

    def _axis_lane_delta(
        self,
        current: FSMRuntimeRow,
        normalized_response: str,
    ) -> tuple[float, float, int, int, int, str]:
        sequence = self._parse_axis_step_sequence_text(current.axis_step_sequence)
        current_index = min(max(int(current.axis_step_index), 0), len(sequence) - 1)
        last_index = len(sequence) - 1
        last_directional = current.axis_last_directional_response

        if normalized_response not in ("ONE", "TWO"):
            step = float(sequence[current_index])
            return (
                0.0,
                step,
                current_index,
                int(current.axis_flip_count),
                int(current.axis_reversal_count),
                last_directional,
            )

        reversal = (
            last_directional in ("ONE", "TWO")
            and normalized_response != last_directional
        )
        next_index = current_index
        safety_flip_count = int(current.axis_flip_count)
        total_reversal_count = int(current.axis_reversal_count)

        if reversal:
            total_reversal_count += 1
            if current_index < last_index:
                next_index = current_index + 1
            else:
                safety_flip_count += 1

        step = float(sequence[next_index])
        delta = jcc_axis_delta(normalized_response, step, positive_for_better_1=True)
        return (
            delta,
            step,
            next_index,
            safety_flip_count,
            total_reversal_count,
            normalized_response,
        )

    def _jcc_power_terminal_negative_reversal(
        self,
        current: FSMRuntimeRow,
        normalized_response: str,
    ) -> bool:
        if current.state not in ("F", "I"):
            return False
        if normalized_response != "TWO":
            return False
        if current.response_value != "ONE":
            return False
        next_flip_count = int(current.duo_flip) + 1
        return next_flip_count >= int(self.cal.get("jcc_power_max_flips", 4))

    def _duochrome_terminal_negative_reversal(
        self,
        current: FSMRuntimeRow,
        dv,
        normalized_response: str,
    ) -> bool:
        if current.state not in ("G", "J"):
            return False
        if normalized_response != "RED":
            return False
        if current.response_value != "GREEN":
            return False
        next_flip_count = int(current.duo_flip) + 1
        return next_flip_count >= int(dv.dv_duochrome_max_flips)

    @staticmethod
    def _final_compare_outcome(choice_1: str, choice_2: str) -> str:
        if choice_1 == "ONE" and choice_2 == "ONE":
            return "Yes"
        if choice_1 or choice_2:
            return "No"
        return ""

    @staticmethod
    def _prompt_family_key(row: FSMRuntimeRow) -> str:
        if row.state in ("B", "D", "C", "L"):
            if row.state in ("B", "D") and row.coarse_compare_mode:
                return "session:line_compare"
            if row.state in ("B", "D") and row.coarse_recheck_mode:
                return "session:line_recheck"
            return "session:line_read"
        if row.state == "S":
            return "session:final_compare_option_1"
        if row.state == "T":
            return "session:final_compare_option_2"
        if row.state == "U":
            return "session:final_compare_choice"
        if row.state in ("E", "F", "H", "I"):
            return f"eye:{row.eye}:jcc_compare"
        if row.state in ("G", "J"):
            return "session:duochrome_compare"
        if row.state == "K":
            return "session:distance_bino_compare"
        if row.state in ("P", "Q", "R"):
            return "session:near_clarity"
        return f"state:{row.state}"

    def _is_early_prompt_for_row(self, row: FSMRuntimeRow) -> bool:
        key = self._prompt_family_key(row)
        seen = int((row.prompt_memory or {}).get(key, 0))
        return seen == 0

    @staticmethod
    def _coarse_chart_memory_key(row: FSMRuntimeRow) -> str:
        if row.state not in ("B", "D") or row.coarse_compare_mode or row.coarse_recheck_mode:
            return ""
        eye = row.eye or row.state
        return f"eye:{eye}:coarse_chart:{row.chart_param}"

    def _is_first_chart_exposure_for_row(self, row: FSMRuntimeRow) -> bool:
        key = self._coarse_chart_memory_key(row)
        if not key:
            return False
        seen = int((row.prompt_memory or {}).get(key, 0))
        return seen == 0

    def _coarse_compare_gate_chart(self) -> str:
        return normalize_chart_param(self.cal.get("distance_target_6_9_chart", "40_30_25"))

    def _should_enter_coarse_compare(self, chart_param: str) -> bool:
        gate_idx = get_chart_index(self._coarse_compare_gate_chart())
        chart_idx = get_chart_index(normalize_chart_param(chart_param))
        if gate_idx <= 0 or chart_idx <= 0:
            return False
        return chart_idx >= gate_idx

    def _remember_prompt(self, row: FSMRuntimeRow) -> None:
        key = self._prompt_family_key(row)
        memory = dict(row.prompt_memory or {})
        memory[key] = int(memory.get(key, 0)) + 1
        chart_key = self._coarse_chart_memory_key(row)
        if chart_key:
            memory[chart_key] = int(memory.get(chart_key, 0)) + 1
        row.prompt_memory = memory

    def _prompt_bundle_for_row(self, row: FSMRuntimeRow) -> tuple[str, tuple[str, ...]]:
        early = self._is_early_prompt_for_row(row)
        state = row.state

        if state in ("B", "D"):
            if row.coarse_compare_mode:
                question = (
                    "Did it get better than before? Say yes or no."
                    if early
                    else "Better than before now? Yes or no?"
                )
                return question, ("CLEAR", "BLURRY", "REPEAT")
            if row.coarse_recheck_mode:
                question = (
                    "Can you read the line now, or is it still blurry?"
                    if early
                    else "Read it now, or say still blurry."
                )
                return question, ("CLEAR", "BLURRY", "REPEAT")
            question = (
                "Please read the line. If the letters are not clear, say blurry, or repeat."
                if early
                else "Read the line, say blurry, or repeat."
            )
            return question, ("CLEAR", "BLURRY", "REPEAT")

        if state in ("E", "H"):
            question = (
                "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat."
                if early
                else "Which is better, or are both same?"
            )
            return question, ("ONE", "TWO", "SAME", "REPEAT")

        if state in ("F", "I"):
            question = (
                "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat."
                if early
                else "Which is better, or are both same?"
            )
            return question, ("ONE", "TWO", "SAME", "REPEAT")

        if state in ("G", "J"):
            question = (
                "Please compare the green and red sides. Letters on which side look sharper and darker? Say green side, red side, both same, or repeat."
                if early
                else "Green side, red side, both same, or repeat?"
            )
            return question, ("GREEN", "RED", "SAME", "REPEAT")

        if state == "K":
            question = (
                "Please compare the letters on the bottom and the top line. Which line looks sharper? Say bottom line, top line, both same, or repeat."
                if early
                else "Bottom line, top line, both same, or repeat?"
            )
            return question, ("BOTTOM", "TOP", "SAME", "REPEAT")

        if state in ("P", "Q", "R"):
            question = (
                "Please read the last line. Is it clear, blurry, or should I repeat?"
                if early
                else "Is the last line clear or blurry?"
            )
            return question, ("CLEAR", "BLURRY", "REPEAT")

        if state == "S":
            question = (
                "This is the final confirmation before I finish your eye test. This is the first option. Please observe the line carefully."
                if early
                else "First option. Please observe the line carefully."
            )
            return question, ()

        if state == "T":
            question = (
                "Now this is the second option. Please observe the line carefully."
                if early
                else "Second option. Please observe the line carefully."
            )
            return question, ()

        if state == "U":
            question = (
                "Which option was better, first option or second option?"
                if early
                else "Which was better, first option or second option?"
            )
            return question, ("ONE", "TWO", "REPEAT")

        if state in ("C", "L"):
            question = (
                "Please read the line. If the letters are not clear, say blurry, or repeat."
                if early
                else "Read the line, say blurry, or repeat."
            )
            return question, ("CLEAR", "BLURRY", "REPEAT")

        cfg = COMPACT_PROMPT_CONFIG.get(state, {})
        return str(cfg.get("question", "")), tuple(cfg.get("options", ()))

    def _normalize_response_value(self, state: str, response_value: str) -> str:
        value = str(response_value or "").strip().upper()
        if not value:
            return value

        if state in ("B", "C", "D", "L", "P", "Q"):
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

        if state in ("S", "T"):
            mapping = {
                "AUTO_ADVANCE": "AUTO_ADVANCE",
                "OBSERVED": "AUTO_ADVANCE",
            }
            return mapping.get(value, value)

        if state in ("E", "F", "H", "I", "U"):
            mapping = {
                "ONE": "ONE",
                "BETTER_1": "ONE",
                "TWO": "TWO",
                "BETTER_2": "TWO",
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
        coarse_compare_mode: bool = False,
        coarse_recheck_mode: bool = False,
        coarse_last_confirmed_chart_re: str = "",
        coarse_last_confirmed_chart_le: str = "",
        distance_va_re_chart: str = "",
        distance_va_le_chart: str = "",
        distance_va_re_line: str = "",
        distance_va_le_line: str = "",
        va_confirm_ceiling_chart: str = "",
        final_compare_enabled: bool = False,
        final_compare_round: int = 0,
        final_compare_option_source: str = "",
        final_compare_choice_round_1: str = "",
        final_compare_choice_round_2: str = "",
        patient_accepted_achieved_over_current_rx: str = "",
        final_compare_current_re_sph: Optional[float] = None,
        final_compare_current_re_cyl: Optional[float] = None,
        final_compare_current_re_axis: Optional[float] = None,
        final_compare_current_le_sph: Optional[float] = None,
        final_compare_current_le_cyl: Optional[float] = None,
        final_compare_current_le_axis: Optional[float] = None,
        final_compare_current_add_r: Optional[float] = None,
        final_compare_current_add_l: Optional[float] = None,
        final_compare_achieved_re_sph: Optional[float] = None,
        final_compare_achieved_re_cyl: Optional[float] = None,
        final_compare_achieved_re_axis: Optional[float] = None,
        final_compare_achieved_le_sph: Optional[float] = None,
        final_compare_achieved_le_cyl: Optional[float] = None,
        final_compare_achieved_le_axis: Optional[float] = None,
        final_compare_achieved_add_r: Optional[float] = None,
        final_compare_achieved_add_l: Optional[float] = None,
        axis_flip_count: int = 0,
        axis_quick_search_active: bool = False,
        axis_quick_phase: str = "",
        axis_last_directional_response: str = "",
        axis_reversal_count: int = 0,
        axis_step_index: int = 0,
        axis_step_sequence: str = "",
        axis_lane_id: str = "",
        axis_lane_name: str = "",
        axis_confidence_label: str = "",
        axis_source_used: str = "",
        axis_selection_reason: str = "",
        axis_is_near_cardinal: bool = False,
        axis_cyl_magnitude_for_lane: float = 0.0,
        prompt_memory: Optional[dict[str, int]] = None,
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
            prompt_memory=dict(prompt_memory or {}),
            duo_iter=duo_iter,
            duo_flip=duo_flip,
            coarse_compare_mode=coarse_compare_mode,
            coarse_recheck_mode=coarse_recheck_mode,
            coarse_last_confirmed_chart_re=coarse_last_confirmed_chart_re,
            coarse_last_confirmed_chart_le=coarse_last_confirmed_chart_le,
            distance_va_re_chart=distance_va_re_chart,
            distance_va_le_chart=distance_va_le_chart,
            distance_va_re_line=distance_va_re_line,
            distance_va_le_line=distance_va_le_line,
            va_confirm_ceiling_chart=va_confirm_ceiling_chart,
            final_compare_enabled=final_compare_enabled,
            final_compare_round=final_compare_round,
            final_compare_option_source=final_compare_option_source,
            final_compare_choice_round_1=final_compare_choice_round_1,
            final_compare_choice_round_2=final_compare_choice_round_2,
            patient_accepted_achieved_over_current_rx=patient_accepted_achieved_over_current_rx,
            final_compare_current_re_sph=final_compare_current_re_sph,
            final_compare_current_re_cyl=final_compare_current_re_cyl,
            final_compare_current_re_axis=final_compare_current_re_axis,
            final_compare_current_le_sph=final_compare_current_le_sph,
            final_compare_current_le_cyl=final_compare_current_le_cyl,
            final_compare_current_le_axis=final_compare_current_le_axis,
            final_compare_current_add_r=final_compare_current_add_r,
            final_compare_current_add_l=final_compare_current_add_l,
            final_compare_achieved_re_sph=final_compare_achieved_re_sph,
            final_compare_achieved_re_cyl=final_compare_achieved_re_cyl,
            final_compare_achieved_re_axis=final_compare_achieved_re_axis,
            final_compare_achieved_le_sph=final_compare_achieved_le_sph,
            final_compare_achieved_le_cyl=final_compare_achieved_le_cyl,
            final_compare_achieved_le_axis=final_compare_achieved_le_axis,
            final_compare_achieved_add_r=final_compare_achieved_add_r,
            final_compare_achieved_add_l=final_compare_achieved_add_l,
            next_state=state,
            row_active=True,
            axis_flip_count=axis_flip_count,
            axis_quick_search_active=axis_quick_search_active,
            axis_quick_phase=axis_quick_phase,
            axis_last_directional_response=axis_last_directional_response,
            axis_reversal_count=axis_reversal_count,
            axis_step_index=axis_step_index,
            axis_step_sequence=axis_step_sequence,
            axis_lane_id=axis_lane_id,
            axis_lane_name=axis_lane_name,
            axis_confidence_label=axis_confidence_label,
            axis_source_used=axis_source_used,
            axis_selection_reason=axis_selection_reason,
            axis_is_near_cardinal=axis_is_near_cardinal,
            axis_cyl_magnitude_for_lane=axis_cyl_magnitude_for_lane,
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
            row.eye = "RE"

        elif state == "D":
            row.phase_name = "Coarse Sphere LE"
            row.phase_type = "COARSE_SPHERE"
            row.stimulus_type = "COARSE_SPH"
            row.chart_type = "SNELLEN_FEET"
            row.eye = "LE"

        elif state == "E":
            row.phase_name = "JCC Axis RE"
            row.phase_type = "JCC_AXIS"
            row.stimulus_type = "JCC_AXIS"
            row.chart_type = "DOT_CHART_JCC"
            row.eye = "RE"

        elif state == "F":
            row.phase_name = "JCC Power RE"
            row.phase_type = "JCC_POWER"
            row.stimulus_type = "JCC_POWER"
            row.chart_type = "DOT_CHART_JCC"
            row.eye = "RE"

        elif state == "G":
            row.phase_name = "Duochrome RE"
            row.phase_type = "DUOCHROME"
            row.stimulus_type = "DUOCHROME"
            row.chart_type = "RED_GREEN_DUOCHROME"
            row.eye = "RE"

        elif state == "H":
            row.phase_name = "JCC Axis LE"
            row.phase_type = "JCC_AXIS"
            row.stimulus_type = "JCC_AXIS"
            row.chart_type = "DOT_CHART_JCC"
            row.eye = "LE"

        elif state == "I":
            row.phase_name = "JCC Power LE"
            row.phase_type = "JCC_POWER"
            row.stimulus_type = "JCC_POWER"
            row.chart_type = "DOT_CHART_JCC"
            row.eye = "LE"

        elif state == "J":
            row.phase_name = "Duochrome LE"
            row.phase_type = "DUOCHROME"
            row.stimulus_type = "DUOCHROME"
            row.chart_type = "RED_GREEN_DUOCHROME"
            row.eye = "LE"

        elif state == "C":
            row.phase_name = "Distance VA Confirm RE"
            row.phase_type = "DISTANCE_VA_CONFIRM"
            row.stimulus_type = "DISTANCE_VA_CONFIRM"
            row.chart_type = "SNELLEN_FEET"
            row.eye = "RE"

        elif state == "K":
            row.phase_name = "Binocular Balance"
            row.phase_type = "BINOC_BALANCE"
            row.stimulus_type = "BINOC_BALANCE"
            row.chart_type = "POLARIZED_BALANCE"
            row.eye = "BIN"

        elif state == "P":
            row.phase_name = "Near Add RE"
            row.phase_type = "NEAR_ADD_RE"
            row.stimulus_type = "NEAR_ADD"
            row.chart_type = "NEAR_CHART"
            row.eye = "RE"

        elif state == "Q":
            row.phase_name = "Near Add LE"
            row.phase_type = "NEAR_ADD_LE"
            row.stimulus_type = "NEAR_ADD"
            row.chart_type = "NEAR_CHART"
            row.eye = "LE"

        elif state == "R":
            row.phase_name = "Near Binocular"
            row.phase_type = "NEAR_BINOC"
            row.stimulus_type = "NEAR_BINOC"
            row.chart_type = "NEAR_CHART"
            row.eye = "BIN"

        elif state == "S":
            row.phase_name = "Final Compare First Option Achieved Rx"
            row.phase_type = "FINAL_RX_COMPARE"
            row.stimulus_type = "FINAL_RX_COMPARE"
            row.chart_type = "SNELLEN_FEET"
            row.eye = "BIN"

        elif state == "T":
            row.phase_name = "Final Compare Second Option PGP"
            row.phase_type = "FINAL_RX_COMPARE"
            row.stimulus_type = "FINAL_RX_COMPARE"
            row.chart_type = "SNELLEN_FEET"
            row.eye = "BIN"

        elif state == "U":
            row.phase_name = "Final Compare Decision"
            row.phase_type = "FINAL_RX_COMPARE"
            row.stimulus_type = "FINAL_RX_COMPARE"
            row.chart_type = "SNELLEN_FEET"
            row.eye = "BIN"

        elif state == "L":
            row.phase_name = "Distance VA Confirm LE"
            row.phase_type = "DISTANCE_VA_CONFIRM"
            row.stimulus_type = "DISTANCE_VA_CONFIRM"
            row.chart_type = "SNELLEN_FEET"
            row.eye = "LE"

        prompt_question, prompt_options = self._prompt_bundle_for_row(row)
        row.response_type = COMPACT_PROMPT_CONFIG.get(state, {}).get("response_type", "clarity_3way")
        row.question = prompt_question
        if len(prompt_options) > 0:
            row.opt_1 = prompt_options[0]
        if len(prompt_options) > 1:
            row.opt_2 = prompt_options[1]
        if len(prompt_options) > 2:
            row.opt_3 = prompt_options[2]
        if len(prompt_options) > 3:
            row.opt_4 = prompt_options[3]
        self._remember_prompt(row)

        return row

    def initialize_row(self, visit_id: str, dv, ar_re=None, ar_le=None) -> FSMRuntimeRow:
        coarse_start_chart = normalize_chart_param(self.cal.get("coarse_start_chart", "70_60_50"))
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
            coarse_compare_mode=False,
            coarse_recheck_mode=False,
            coarse_last_confirmed_chart_re="",
            coarse_last_confirmed_chart_le="",
            distance_va_re_chart="",
            distance_va_le_chart="",
            distance_va_re_line="",
            distance_va_le_line="",
            va_confirm_ceiling_chart="",
            final_compare_enabled=False,
            final_compare_round=0,
            final_compare_option_source="",
            final_compare_choice_round_1="",
            final_compare_choice_round_2="",
            patient_accepted_achieved_over_current_rx="",
            final_compare_current_re_sph=None,
            final_compare_current_re_cyl=None,
            final_compare_current_re_axis=None,
            final_compare_current_le_sph=None,
            final_compare_current_le_cyl=None,
            final_compare_current_le_axis=None,
            final_compare_current_add_r=None,
            final_compare_current_add_l=None,
            final_compare_achieved_re_sph=None,
            final_compare_achieved_re_cyl=None,
            final_compare_achieved_re_axis=None,
            final_compare_achieved_le_sph=None,
            final_compare_achieved_le_cyl=None,
            final_compare_achieved_le_axis=None,
            final_compare_achieved_add_r=None,
            final_compare_achieved_add_l=None,
            axis_step=self._axis_fixed_step(),
            axis_flip_count=0,
            axis_quick_search_active=False,
            axis_quick_phase="",
            axis_last_directional_response="",
            axis_reversal_count=0,
            axis_step_index=0,
            axis_step_sequence=str(int(self._axis_fixed_step())),
            axis_lane_id="",
            axis_lane_name="",
            axis_confidence_label="",
            axis_source_used="",
            axis_selection_reason="",
            axis_is_near_cardinal=False,
            axis_cyl_magnitude_for_lane=0.0,
            prompt_memory={},
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
            if row.coarse_compare_mode:
                return row.chart_param
            if row.response_value == "CLEAR" and row.next_state == "B":
                return row.next_chart_param
            return row.chart_param

        if row.state == "D":
            if row.coarse_compare_mode:
                return row.chart_param
            if row.response_value == "CLEAR" and row.next_state == "D":
                return row.next_chart_param
            return row.chart_param

        if row.state in ("C", "L"):
            if row.response_value == "BLURRY" and row.next_state == row.state:
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
        next_coarse_compare_mode = row.coarse_compare_mode if row.next_state == row.state else False
        next_coarse_recheck_mode = row.coarse_recheck_mode if row.next_state == row.state else False
        next_prompt_memory = dict(row.prompt_memory or {})
        next_coarse_last_confirmed_chart_re = row.coarse_last_confirmed_chart_re
        next_coarse_last_confirmed_chart_le = row.coarse_last_confirmed_chart_le
        next_distance_va_re_chart = row.distance_va_re_chart
        next_distance_va_le_chart = row.distance_va_le_chart
        next_distance_va_re_line = row.distance_va_re_line
        next_distance_va_le_line = row.distance_va_le_line
        next_va_confirm_ceiling_chart = row.va_confirm_ceiling_chart if row.next_state == row.state else ""
        next_final_compare_enabled = row.final_compare_enabled
        next_final_compare_round = row.final_compare_round if row.next_state == row.state else row.final_compare_round
        next_final_compare_option_source = row.final_compare_option_source
        next_final_compare_choice_round_1 = row.final_compare_choice_round_1
        next_final_compare_choice_round_2 = row.final_compare_choice_round_2
        next_patient_accepted_achieved = row.patient_accepted_achieved_over_current_rx
        next_final_compare_current_re_sph = row.final_compare_current_re_sph
        next_final_compare_current_re_cyl = row.final_compare_current_re_cyl
        next_final_compare_current_re_axis = row.final_compare_current_re_axis
        next_final_compare_current_le_sph = row.final_compare_current_le_sph
        next_final_compare_current_le_cyl = row.final_compare_current_le_cyl
        next_final_compare_current_le_axis = row.final_compare_current_le_axis
        next_final_compare_current_add_r = row.final_compare_current_add_r
        next_final_compare_current_add_l = row.final_compare_current_add_l
        next_final_compare_achieved_re_sph = row.final_compare_achieved_re_sph
        next_final_compare_achieved_re_cyl = row.final_compare_achieved_re_cyl
        next_final_compare_achieved_re_axis = row.final_compare_achieved_re_axis
        next_final_compare_achieved_le_sph = row.final_compare_achieved_le_sph
        next_final_compare_achieved_le_cyl = row.final_compare_achieved_le_cyl
        next_final_compare_achieved_le_axis = row.final_compare_achieved_le_axis
        next_final_compare_achieved_add_r = row.final_compare_achieved_add_r
        next_final_compare_achieved_add_l = row.final_compare_achieved_add_l
        next_axis_flip_count = row.axis_flip_count if row.next_state == row.state else 0
        next_axis_quick_search_active = row.axis_quick_search_active if row.next_state == row.state else False
        next_axis_quick_phase = row.axis_quick_phase if row.next_state == row.state else ""
        next_axis_last_directional_response = row.axis_last_directional_response if row.next_state == row.state else ""
        next_axis_reversal_count = row.axis_reversal_count if row.next_state == row.state else 0
        next_axis_step_index = row.axis_step_index if row.next_state == row.state else 0
        next_axis_step_sequence = row.axis_step_sequence if row.next_state == row.state else str(int(self._axis_fixed_step()))
        next_axis_lane_id = row.axis_lane_id if row.next_state == row.state else ""
        next_axis_lane_name = row.axis_lane_name if row.next_state == row.state else ""
        next_axis_confidence_label = row.axis_confidence_label if row.next_state == row.state else ""
        next_axis_source_used = row.axis_source_used if row.next_state == row.state else ""
        next_axis_selection_reason = row.axis_selection_reason if row.next_state == row.state else ""
        next_axis_is_near_cardinal = row.axis_is_near_cardinal if row.next_state == row.state else False
        next_axis_cyl_magnitude_for_lane = row.axis_cyl_magnitude_for_lane if row.next_state == row.state else 0.0
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
            next_chart_param = normalize_chart_param(self.cal.get("coarse_start_chart", "70_60_50"))
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
                entry_chart = normalize_chart_param(self.cal.get("coarse_start_chart", "70_60_50"))

            base_sph = dv.dv_start_rx_LE_sph or 0.0
            if getattr(dv, "dv_fogging_required", False):
                base_sph += float(dv.dv_fogging_amount_D or 0.0)
                next_fog_active = True
                next_fog_start_le_sph = base_sph

            next_le_sph = base_sph
            next_le_cyl = dv.dv_start_rx_LE_cyl
            next_le_axis = dv.dv_start_rx_LE_axis
            next_chart_param = normalize_chart_param(entry_chart)

        if row.next_state == "C" and row.state != "C":
            next_chart_param = "20_20_20"
            next_va_confirm_ceiling_chart = row.coarse_last_confirmed_chart_re or row.chart_param or "20_20_20"

        if row.next_state == "L" and row.state != "L":
            next_chart_param = "20_20_20"
            next_va_confirm_ceiling_chart = row.coarse_last_confirmed_chart_le or row.chart_param or "20_20_20"

        if row.next_state == "R" and row.state != "R":
            next_near_bino_start_add_r = next_add_r
            next_near_bino_start_add_l = next_add_l
            next_near_bino_direction = ""
            next_near_bino_reversed = False

        if row.next_state == "S":
            if row.state != "S":
                if row.state == "U" and row.response_value == "REPEAT":
                    next_final_compare_round = max(1, int(row.final_compare_round or 1))
                else:
                    next_final_compare_round = 1 if int(row.final_compare_round or 0) == 0 else max(1, int(row.final_compare_round or 0) + 1)
            next_final_compare_option_source = "Achieved"
            if int(row.final_compare_round or 0) == 0:
                next_final_compare_achieved_re_sph = next_re_sph
                next_final_compare_achieved_re_cyl = next_re_cyl
                next_final_compare_achieved_re_axis = next_re_axis
                next_final_compare_achieved_le_sph = next_le_sph
                next_final_compare_achieved_le_cyl = next_le_cyl
                next_final_compare_achieved_le_axis = next_le_axis
                next_final_compare_achieved_add_r = next_add_r
                next_final_compare_achieved_add_l = next_add_l
            next_re_sph = next_final_compare_achieved_re_sph
            next_re_cyl = next_final_compare_achieved_re_cyl
            next_re_axis = next_final_compare_achieved_re_axis
            next_le_sph = next_final_compare_achieved_le_sph
            next_le_cyl = next_final_compare_achieved_le_cyl
            next_le_axis = next_final_compare_achieved_le_axis
            next_add_r = next_final_compare_achieved_add_r
            next_add_l = next_final_compare_achieved_add_l
            next_chart_param = "20_20_20"

        if row.next_state == "T":
            next_final_compare_option_source = "PGP"
            next_re_sph = next_final_compare_current_re_sph
            next_re_cyl = next_final_compare_current_re_cyl
            next_re_axis = next_final_compare_current_re_axis
            next_le_sph = next_final_compare_current_le_sph
            next_le_cyl = next_final_compare_current_le_cyl
            next_le_axis = next_final_compare_current_le_axis
            next_add_r = next_final_compare_current_add_r
            next_add_l = next_final_compare_current_add_l
            next_chart_param = "20_20_20"

        if row.next_state == "U":
            next_final_compare_option_source = "PGP"
            next_re_sph = next_final_compare_current_re_sph
            next_re_cyl = next_final_compare_current_re_cyl
            next_re_axis = next_final_compare_current_re_axis
            next_le_sph = next_final_compare_current_le_sph
            next_le_cyl = next_final_compare_current_le_cyl
            next_le_axis = next_final_compare_current_le_axis
            next_add_r = next_final_compare_current_add_r
            next_add_l = next_final_compare_current_add_l
            next_chart_param = "20_20_20"

        if row.next_state == "F" and row.state != "F":
            next_jcc_power_start_re_cyl = next_re_cyl

        if row.next_state == "I" and row.state != "I":
            next_jcc_power_start_le_cyl = next_le_cyl

        if row.next_state in ("E", "H") and row.state != row.next_state:
            axis_lane = self._axis_lane_metadata_for_state(row.next_state, dv)
            next_axis_quick_search_active = bool(axis_lane["lane_id"] == "LANE_4")
            next_axis_quick_phase = ""
            next_axis_last_directional_response = ""
            next_axis_reversal_count = 0
            next_axis_step_index = 0
            next_axis_step_sequence = axis_lane["step_sequence"] or str(int(self._axis_fixed_step()))
            next_axis_lane_id = axis_lane["lane_id"]
            next_axis_lane_name = axis_lane["lane_name"]
            next_axis_confidence_label = axis_lane["confidence_label"]
            next_axis_source_used = axis_lane["source_used"]
            next_axis_selection_reason = axis_lane["selection_reason"]
            next_axis_is_near_cardinal = bool(axis_lane["is_near_cardinal"])
            next_axis_cyl_magnitude_for_lane = float(axis_lane["cyl_magnitude_for_lane"])

        if row.next_state in ("E", "H"):
            next_axis_step = self._axis_nominal_step(next_axis_step_sequence, next_axis_step_index)
        elif row.next_state == row.state:
            next_axis_step = row.axis_step
        else:
            next_axis_step = self._axis_fixed_step()

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
            coarse_compare_mode=next_coarse_compare_mode,
            coarse_recheck_mode=next_coarse_recheck_mode,
            coarse_last_confirmed_chart_re=next_coarse_last_confirmed_chart_re,
            coarse_last_confirmed_chart_le=next_coarse_last_confirmed_chart_le,
            distance_va_re_chart=next_distance_va_re_chart,
            distance_va_le_chart=next_distance_va_le_chart,
            distance_va_re_line=next_distance_va_re_line,
            distance_va_le_line=next_distance_va_le_line,
            va_confirm_ceiling_chart=next_va_confirm_ceiling_chart,
            final_compare_enabled=next_final_compare_enabled,
            final_compare_round=next_final_compare_round,
            final_compare_option_source=next_final_compare_option_source,
            final_compare_choice_round_1=next_final_compare_choice_round_1,
            final_compare_choice_round_2=next_final_compare_choice_round_2,
            patient_accepted_achieved_over_current_rx=next_patient_accepted_achieved,
            final_compare_current_re_sph=next_final_compare_current_re_sph,
            final_compare_current_re_cyl=next_final_compare_current_re_cyl,
            final_compare_current_re_axis=next_final_compare_current_re_axis,
            final_compare_current_le_sph=next_final_compare_current_le_sph,
            final_compare_current_le_cyl=next_final_compare_current_le_cyl,
            final_compare_current_le_axis=next_final_compare_current_le_axis,
            final_compare_current_add_r=next_final_compare_current_add_r,
            final_compare_current_add_l=next_final_compare_current_add_l,
            final_compare_achieved_re_sph=next_final_compare_achieved_re_sph,
            final_compare_achieved_re_cyl=next_final_compare_achieved_re_cyl,
            final_compare_achieved_re_axis=next_final_compare_achieved_re_axis,
            final_compare_achieved_le_sph=next_final_compare_achieved_le_sph,
            final_compare_achieved_le_cyl=next_final_compare_achieved_le_cyl,
            final_compare_achieved_le_axis=next_final_compare_achieved_le_axis,
            final_compare_achieved_add_r=next_final_compare_achieved_add_r,
            final_compare_achieved_add_l=next_final_compare_achieved_add_l,
            axis_step=next_axis_step,
            axis_flip_count=next_axis_flip_count,
            axis_quick_search_active=next_axis_quick_search_active,
            axis_quick_phase=next_axis_quick_phase,
            axis_last_directional_response=next_axis_last_directional_response,
            axis_reversal_count=next_axis_reversal_count,
            axis_step_index=next_axis_step_index,
            axis_step_sequence=next_axis_step_sequence,
            axis_lane_id=next_axis_lane_id,
            axis_lane_name=next_axis_lane_name,
            axis_confidence_label=next_axis_confidence_label,
            axis_source_used=next_axis_source_used,
            axis_selection_reason=next_axis_selection_reason,
            axis_is_near_cardinal=next_axis_is_near_cardinal,
            axis_cyl_magnitude_for_lane=next_axis_cyl_magnitude_for_lane,
            prompt_memory=next_prompt_memory,
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
        va_confirm_completed = False

        if current.state == "B":
            row.next_chart_param = str(current.chart_param)
            if current.coarse_compare_mode:
                if normalized_response == "CLEAR":
                    row.coarse_compare_mode = False
                    row.coarse_recheck_mode = True
                elif normalized_response == "BLURRY":
                    row.ds_re = abs(float(current.sph_step))
                    row.coarse_last_confirmed_chart_re = str(current.chart_param or row.coarse_last_confirmed_chart_re or "")
                    row.coarse_compare_mode = False
                    row.coarse_recheck_mode = False
                    coarse_endpoint_reached = True
                else:
                    row.coarse_compare_mode = True
                    row.coarse_recheck_mode = False
            else:
                row.coarse_recheck_mode = current.coarse_recheck_mode if normalized_response == "REPEAT" else False
                if normalized_response == "CLEAR":
                    row.coarse_last_confirmed_chart_re = str(current.chart_param)
                    row.next_chart_param = get_next_chart(str(current.chart_param))
                    if row.next_chart_param == str(current.chart_param):
                        coarse_endpoint_reached = True
                elif normalized_response == "BLURRY":
                    row.ds_re = coarse_sphere_delta(normalized_response, current.sph_step)
                    row.coarse_compare_mode = self._should_enter_coarse_compare(current.chart_param)

        elif current.state == "D":
            row.next_chart_param = str(current.chart_param)
            if current.coarse_compare_mode:
                if normalized_response == "CLEAR":
                    row.coarse_compare_mode = False
                    row.coarse_recheck_mode = True
                elif normalized_response == "BLURRY":
                    row.ds_le = abs(float(current.sph_step))
                    row.coarse_last_confirmed_chart_le = str(current.chart_param or row.coarse_last_confirmed_chart_le or "")
                    row.coarse_compare_mode = False
                    row.coarse_recheck_mode = False
                    coarse_endpoint_reached = True
                else:
                    row.coarse_compare_mode = True
                    row.coarse_recheck_mode = False
            else:
                row.coarse_recheck_mode = current.coarse_recheck_mode if normalized_response == "REPEAT" else False
                if normalized_response == "CLEAR":
                    row.coarse_last_confirmed_chart_le = str(current.chart_param)
                    row.next_chart_param = get_next_chart(str(current.chart_param))
                    if row.next_chart_param == str(current.chart_param):
                        coarse_endpoint_reached = True
                elif normalized_response == "BLURRY":
                    row.ds_le = coarse_sphere_delta(normalized_response, current.sph_step)
                    row.coarse_compare_mode = self._should_enter_coarse_compare(current.chart_param)

        elif current.state == "E":
            if self._axis_converges_on_terminal_reversal(current, dv, normalized_response):
                row.da_re = 0.0
                row.axis_step = self._axis_nominal_step(current.axis_step_sequence, current.axis_step_index)
                row.axis_step_index = current.axis_step_index
                row.axis_flip_count = current.axis_flip_count
                row.axis_reversal_count = current.axis_reversal_count + 1
                row.axis_last_directional_response = normalized_response
            else:
                (
                    row.da_re,
                    row.axis_step,
                    row.axis_step_index,
                    row.axis_flip_count,
                    row.axis_reversal_count,
                    row.axis_last_directional_response,
                ) = self._axis_lane_delta(current, normalized_response)

        elif current.state == "H":
            if self._axis_converges_on_terminal_reversal(current, dv, normalized_response):
                row.da_le = 0.0
                row.axis_step = self._axis_nominal_step(current.axis_step_sequence, current.axis_step_index)
                row.axis_step_index = current.axis_step_index
                row.axis_flip_count = current.axis_flip_count
                row.axis_reversal_count = current.axis_reversal_count + 1
                row.axis_last_directional_response = normalized_response
            else:
                (
                    row.da_le,
                    row.axis_step,
                    row.axis_step_index,
                    row.axis_flip_count,
                    row.axis_reversal_count,
                    row.axis_last_directional_response,
                ) = self._axis_lane_delta(current, normalized_response)

        elif current.state == "F":
            if self._jcc_power_terminal_negative_reversal(current, normalized_response):
                row.dc_re = 0.0
                row.ds_re = 0.0
            else:
                proposed_dc = jcc_power_cyl_delta(normalized_response, current.cyl_step)
                row.dc_re = clamp_cyl_delta_at_zero(current.re_cyl, proposed_dc)
                row.ds_re = jcc_power_sphere_compensation(
                    current_cyl=current.re_cyl,
                    proposed_cyl_delta=row.dc_re,
                    start_cyl=current.jcc_power_start_re_cyl,
                )

        elif current.state == "I":
            if self._jcc_power_terminal_negative_reversal(current, normalized_response):
                row.dc_le = 0.0
                row.ds_le = 0.0
            else:
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

            if self._duochrome_terminal_negative_reversal(current, dv, normalized_response):
                row.ds_re = 0.0
            else:
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

            if self._duochrome_terminal_negative_reversal(current, dv, normalized_response):
                row.ds_le = 0.0
            else:
                row.ds_le = duochrome_sphere_delta(
                    response_value=normalized_response,
                    endpoint_bias_policy=dv.dv_endpoint_bias_policy,
                    equal_confirmation_reached=equal_reached,
                    calibration=self.cal,
                )

        elif current.state == "C":
            row.next_chart_param = str(current.chart_param)
            if normalized_response == "CLEAR":
                row.distance_va_re_chart = str(current.chart_param)
                row.distance_va_re_line = chart_to_last_line_va(current.chart_param)
                va_confirm_completed = True
            elif normalized_response == "BLURRY":
                next_chart = get_previous_chart(str(current.chart_param))
                ceiling_chart = str(current.va_confirm_ceiling_chart or current.coarse_last_confirmed_chart_re or current.chart_param)
                if str(current.chart_param) == ceiling_chart or next_chart == str(current.chart_param):
                    final_chart = str(current.coarse_last_confirmed_chart_re or current.chart_param)
                    row.distance_va_re_chart = final_chart
                    row.distance_va_re_line = chart_to_last_line_va(final_chart)
                    va_confirm_completed = True
                else:
                    row.next_chart_param = next_chart

        elif current.state == "L":
            row.next_chart_param = str(current.chart_param)
            if normalized_response == "CLEAR":
                row.distance_va_le_chart = str(current.chart_param)
                row.distance_va_le_line = chart_to_last_line_va(current.chart_param)
                va_confirm_completed = True
            elif normalized_response == "BLURRY":
                next_chart = get_previous_chart(str(current.chart_param))
                ceiling_chart = str(current.va_confirm_ceiling_chart or current.coarse_last_confirmed_chart_le or current.chart_param)
                if str(current.chart_param) == ceiling_chart or next_chart == str(current.chart_param):
                    final_chart = str(current.coarse_last_confirmed_chart_le or current.chart_param)
                    row.distance_va_le_chart = final_chart
                    row.distance_va_le_line = chart_to_last_line_va(final_chart)
                    va_confirm_completed = True
                else:
                    row.next_chart_param = next_chart

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

        elif current.state in ("S", "T"):
            pass

        elif current.state == "U":
            if normalized_response in ("ONE", "TWO"):
                if int(current.final_compare_round or 0) <= 1:
                    row.final_compare_choice_round_1 = normalized_response
                else:
                    row.final_compare_choice_round_2 = normalized_response
                row.patient_accepted_achieved_over_current_rx = self._final_compare_outcome(
                    row.final_compare_choice_round_1,
                    row.final_compare_choice_round_2,
                )

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
            if normalized_response not in ("ONE", "TWO"):
                row.axis_flip_count = current.axis_flip_count
                row.axis_reversal_count = current.axis_reversal_count
                row.axis_step = self._axis_nominal_step(current.axis_step_sequence, current.axis_step_index)
                row.axis_step_index = current.axis_step_index
                row.axis_last_directional_response = current.axis_last_directional_response
        else:
            row.axis_flip_count = 0
            row.axis_reversal_count = 0

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

        if current.state == "C" and timeout and not row.distance_va_re_line:
            final_chart = str(current.coarse_last_confirmed_chart_re or current.chart_param)
            row.distance_va_re_chart = final_chart
            row.distance_va_re_line = chart_to_last_line_va(final_chart)
        if current.state == "L" and timeout and not row.distance_va_le_line:
            final_chart = str(current.coarse_last_confirmed_chart_le or current.chart_param)
            row.distance_va_le_chart = final_chart
            row.distance_va_le_line = chart_to_last_line_va(final_chart)

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
            "coarse_compare_mode": bool(current.coarse_compare_mode),
            "va_confirm_completed": bool(va_confirm_completed),
            "jcc_power_flip_limit_hit": row.duo_flip >= int(self.cal.get("jcc_power_max_flips", 4)),
            "jcc_cyl_at_zero": (
                (current.state == "F" and abs(((current.re_cyl or 0.0) + (row.dc_re or 0.0))) < 1e-9)
                or (current.state == "I" and abs(((current.le_cyl or 0.0) + (row.dc_le or 0.0))) < 1e-9)
            ),

            # FSM v2.3 additions
            "axis_same_required": int(dv.dv_jcc_axis_same_required),
            "axis_flip_count": int(row.axis_flip_count),
            "axis_flip_max": int(dv.dv_jcc_axis_max_flips),
            "axis_reversal_converged": bool(
                self._axis_converges_on_terminal_reversal(current, dv, normalized_response)
            ),

            "near_binoc_direction": row.near_bino_direction,
            "near_binoc_reversed": bool(row.near_bino_reversed),
            "near_binoc_max_plus_steps": int(dv.dv_near_binoc_max_plus_steps),
            "near_binoc_max_minus_steps": int(dv.dv_near_binoc_max_minus_steps),
            "final_compare_enabled": bool(current.final_compare_enabled),
            "final_compare_round": int(current.final_compare_round or 0),
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
