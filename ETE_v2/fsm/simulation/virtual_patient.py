from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fsm.charts.chart_scale import get_chart_index
from fsm.models.fsm_runtime import FSMRuntimeRow


@dataclass
class TruthRx:
    re_sph: float
    re_cyl: float
    re_axis: float
    le_sph: float
    le_cyl: float
    le_axis: float
    add_r: float = 0.0
    add_l: float = 0.0


def _axis_error(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    d = abs(float(a) - float(b)) % 180
    return min(d, 180 - d)


class VirtualPatient:
    """Rule-based virtual patient aligned with the current ETE_v2 FSM labels."""

    def __init__(self, truth: TruthRx):
        self.truth = truth

    def respond(self, row: FSMRuntimeRow) -> str:
        state = row.state

        if state == "B":
            return self._respond_coarse(row, eye="RE")
        if state == "D":
            return self._respond_coarse(row, eye="LE")
        if state == "C":
            return self._respond_distance_confirm(row, eye="RE")
        if state == "L":
            return self._respond_distance_confirm(row, eye="LE")
        if state == "E":
            return self._respond_jcc_axis(row.re_axis, self.truth.re_axis, row.axis_step)
        if state == "H":
            return self._respond_jcc_axis(row.le_axis, self.truth.le_axis, row.axis_step)
        if state == "F":
            return self._respond_jcc_power(row.re_cyl, self.truth.re_cyl, row.cyl_step)
        if state == "I":
            return self._respond_jcc_power(row.le_cyl, self.truth.le_cyl, row.cyl_step)
        if state == "G":
            return self._respond_duochrome(row.re_sph, self.truth.re_sph)
        if state == "J":
            return self._respond_duochrome(row.le_sph, self.truth.le_sph)
        if state == "K":
            return self._respond_binocular_balance(row)
        if state == "P":
            return self._respond_near(row.add_r, self.truth.add_r)
        if state == "Q":
            return self._respond_near(row.add_l, self.truth.add_l)
        if state == "R":
            return self._respond_near_binocular(row)
        if state in ("S", "T"):
            return "AUTO_ADVANCE"
        if state == "U":
            return self._respond_final_compare(row)

        return "REPEAT"

    def _distance_quality(self, current_sph: Optional[float], current_cyl: Optional[float], current_axis: Optional[float],
                          truth_sph: float, truth_cyl: float, truth_axis: float) -> float:
        sphere_err = abs(float(current_sph or 0.0) - float(truth_sph))
        cyl_err = abs(float(current_cyl or 0.0) - float(truth_cyl))
        axis_weight = min(max(abs(float(truth_cyl)), abs(float(current_cyl or 0.0))), 2.0)
        axis_equiv = (_axis_error(current_axis, truth_axis) / 45.0) * axis_weight * 0.5
        return sphere_err + 0.6 * cyl_err + axis_equiv

    def _highest_readable_chart_index(
        self,
        current_sph: Optional[float],
        current_cyl: Optional[float],
        current_axis: Optional[float],
        *,
        truth_sph: float,
        truth_cyl: float,
        truth_axis: float,
    ) -> int:
        quality = self._distance_quality(
            current_sph,
            current_cyl,
            current_axis,
            truth_sph,
            truth_cyl,
            truth_axis,
        )
        if quality <= 0.25:
            return 6
        if quality <= 0.50:
            return 5
        if quality <= 0.75:
            return 4
        if quality <= 1.25:
            return 3
        if quality <= 1.75:
            return 2
        return 1

    def _respond_coarse(self, row: FSMRuntimeRow, *, eye: str) -> str:
        current_sph = row.re_sph if eye == "RE" else row.le_sph
        current_cyl = row.re_cyl if eye == "RE" else row.le_cyl
        current_axis = row.re_axis if eye == "RE" else row.le_axis
        truth_sph = self.truth.re_sph if eye == "RE" else self.truth.le_sph
        truth_cyl = self.truth.re_cyl if eye == "RE" else self.truth.le_cyl
        truth_axis = self.truth.re_axis if eye == "RE" else self.truth.le_axis

        if row.coarse_compare_mode:
            before_sph = (float(current_sph or 0.0) + float(row.sph_step or 0.25))
            before_quality = self._distance_quality(
                before_sph,
                current_cyl,
                current_axis,
                truth_sph,
                truth_cyl,
                truth_axis,
            )
            after_quality = self._distance_quality(
                current_sph,
                current_cyl,
                current_axis,
                truth_sph,
                truth_cyl,
                truth_axis,
            )
            return "CLEAR" if after_quality < before_quality else "BLURRY"

        readable_index = self._highest_readable_chart_index(
            current_sph,
            current_cyl,
            current_axis,
            truth_sph=truth_sph,
            truth_cyl=truth_cyl,
            truth_axis=truth_axis,
        )
        chart_index = max(get_chart_index(str(row.chart_param)), 1)
        return "CLEAR" if chart_index <= readable_index else "BLURRY"

    def _respond_distance_confirm(self, row: FSMRuntimeRow, *, eye: str) -> str:
        current_sph = row.re_sph if eye == "RE" else row.le_sph
        current_cyl = row.re_cyl if eye == "RE" else row.le_cyl
        current_axis = row.re_axis if eye == "RE" else row.le_axis
        truth_sph = self.truth.re_sph if eye == "RE" else self.truth.le_sph
        truth_cyl = self.truth.re_cyl if eye == "RE" else self.truth.le_cyl
        truth_axis = self.truth.re_axis if eye == "RE" else self.truth.le_axis
        readable_index = self._highest_readable_chart_index(
            current_sph,
            current_cyl,
            current_axis,
            truth_sph=truth_sph,
            truth_cyl=truth_cyl,
            truth_axis=truth_axis,
        )
        chart_index = max(get_chart_index(str(row.chart_param)), 1)
        return "CLEAR" if chart_index <= readable_index else "BLURRY"

    def _respond_jcc_axis(
        self,
        current_axis: Optional[float],
        truth_axis: float,
        axis_step: Optional[float],
    ) -> str:
        if current_axis is None:
            return "REPEAT"

        err = _axis_error(current_axis, truth_axis)
        step = float(axis_step or 5.0)
        if err <= max(1.0, step / 2.0):
            return "SAME"

        pos_err = _axis_error((float(current_axis) + step) % 180, truth_axis)
        neg_err = _axis_error((float(current_axis) - step) % 180, truth_axis)
        if pos_err < neg_err:
            return "ONE"
        if neg_err < pos_err:
            return "TWO"
        return "SAME"

    def _respond_jcc_power(
        self,
        current_cyl: Optional[float],
        truth_cyl: float,
        cyl_step: Optional[float],
    ) -> str:
        if current_cyl is None:
            return "REPEAT"

        step = float(cyl_step or 0.25)
        err = float(current_cyl) - float(truth_cyl)
        if abs(err) <= step / 2.0:
            return "SAME"

        plus_err = abs((float(current_cyl) + step) - float(truth_cyl))
        minus_err = abs((float(current_cyl) - step) - float(truth_cyl))
        if plus_err < minus_err:
            return "ONE"
        if minus_err < plus_err:
            return "TWO"
        return "SAME"

    def _respond_duochrome(self, current_sph: Optional[float], truth_sph: float) -> str:
        if current_sph is None:
            return "REPEAT"
        err = float(current_sph) - float(truth_sph)
        if err > 0.125:
            return "RED"
        if err < -0.125:
            return "GREEN"
        return "SAME"

    def _respond_binocular_balance(self, row: FSMRuntimeRow) -> str:
        re_err = abs((row.re_sph or 0.0) - self.truth.re_sph)
        le_err = abs((row.le_sph or 0.0) - self.truth.le_sph)
        if abs(re_err - le_err) <= 0.125:
            return "SAME"
        if re_err > le_err:
            return "TOP"
        return "BOTTOM"

    def _respond_near(self, current_add: Optional[float], truth_add: float) -> str:
        err = abs((current_add or 0.0) - truth_add)
        return "CLEAR" if err <= 0.25 else "BLURRY"

    def _respond_near_binocular(self, row: FSMRuntimeRow) -> str:
        re_err = abs((row.add_r or 0.0) - self.truth.add_r)
        le_err = abs((row.add_l or 0.0) - self.truth.add_l)
        return "CLEAR" if max(re_err, le_err) <= 0.25 else "BLURRY"

    def _respond_final_compare(self, row: FSMRuntimeRow) -> str:
        achieved_score = self._payload_distance_score(
            row.final_compare_achieved_re_sph,
            row.final_compare_achieved_re_cyl,
            row.final_compare_achieved_re_axis,
            row.final_compare_achieved_add_r,
            row.final_compare_achieved_le_sph,
            row.final_compare_achieved_le_cyl,
            row.final_compare_achieved_le_axis,
            row.final_compare_achieved_add_l,
        )
        current_score = self._payload_distance_score(
            row.final_compare_current_re_sph,
            row.final_compare_current_re_cyl,
            row.final_compare_current_re_axis,
            row.final_compare_current_add_r,
            row.final_compare_current_le_sph,
            row.final_compare_current_le_cyl,
            row.final_compare_current_le_axis,
            row.final_compare_current_add_l,
        )
        return "ONE" if achieved_score <= current_score else "TWO"

    def _payload_distance_score(
        self,
        re_sph: Optional[float],
        re_cyl: Optional[float],
        re_axis: Optional[float],
        add_r: Optional[float],
        le_sph: Optional[float],
        le_cyl: Optional[float],
        le_axis: Optional[float],
        add_l: Optional[float],
    ) -> float:
        return (
            abs(float(re_sph or 0.0) - self.truth.re_sph)
            + abs(float(re_cyl or 0.0) - self.truth.re_cyl)
            + _axis_error(re_axis, self.truth.re_axis) / 20.0
            + abs(float(le_sph or 0.0) - self.truth.le_sph)
            + abs(float(le_cyl or 0.0) - self.truth.le_cyl)
            + _axis_error(le_axis, self.truth.le_axis) / 20.0
            + abs(float(add_r or 0.0) - self.truth.add_r) * 0.5
            + abs(float(add_l or 0.0) - self.truth.add_l) * 0.5
        )
