from dataclasses import dataclass
from typing import Optional

from refraction_engine.models.fsm_runtime import FSMRuntimeRow


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
    """
    Deterministic rule-based virtual patient.

    Returns phase-valid responses that should converge
    toward the supplied truth Rx.
    """

    def __init__(self, truth: TruthRx):
        self.truth = truth

    def respond(self, row: FSMRuntimeRow) -> str:
        state = row.state

        if state == "A":
            return self._respond_distance_baseline(row)

        if state == "B":
            return self._respond_coarse_re(row)

        if state == "D":
            return self._respond_coarse_le(row)

        if state == "E":
            return self._respond_jcc_axis_re(row)

        if state == "H":
            return self._respond_jcc_axis_le(row)

        if state == "F":
            return self._respond_jcc_power_re(row)

        if state == "I":
            return self._respond_jcc_power_le(row)

        if state == "G":
            return self._respond_duochrome_re(row)

        if state == "J":
            return self._respond_duochrome_le(row)

        if state == "K":
            return self._respond_binocular_balance(row)

        if state == "P":
            return self._respond_near_re(row)

        if state == "Q":
            return self._respond_near_le(row)

        if state == "R":
            return self._respond_near_binoc(row)

        return "CANT_TELL"

    # -----------------------------
    # Distance readability phases
    # -----------------------------

    def _respond_distance_baseline(self, row: FSMRuntimeRow) -> str:
        idx = int(row.chart_idx or 0)
        target_idx = int(row.target_chart_idx or 0)

        if idx < target_idx:
            return "READABLE"
        if idx == target_idx:
            return "BLURRY"
        return "NOT_READABLE"

    def _respond_coarse_re(self, row: FSMRuntimeRow) -> str:
        return self._coarse_response(
            current_sph=row.re_sph,
            truth_sph=self.truth.re_sph,
            chart_idx=row.chart_idx,
            target_chart_idx=row.target_chart_idx,
        )

    def _respond_coarse_le(self, row: FSMRuntimeRow) -> str:
        return self._coarse_response(
            current_sph=row.le_sph,
            truth_sph=self.truth.le_sph,
            chart_idx=row.chart_idx,
            target_chart_idx=row.target_chart_idx,
        )

    def _coarse_response(
        self,
        current_sph: Optional[float],
        truth_sph: float,
        chart_idx: Optional[int],
        target_chart_idx: Optional[int],
    ) -> str:
        if current_sph is None:
            return "BLURRY"

        err = float(current_sph) - float(truth_sph)

        if (chart_idx or 0) < (target_chart_idx or 0):
            return "READABLE"

        if abs(err) <= 0.25:
            return "READABLE"
        return "BLURRY"

    # -----------------------------
    # JCC Axis
    # -----------------------------

    def _respond_jcc_axis_re(self, row: FSMRuntimeRow) -> str:
        return self._jcc_axis_response(
            current_axis=row.re_axis,
            truth_axis=self.truth.re_axis,
            axis_step=row.axis_step,
        )

    def _respond_jcc_axis_le(self, row: FSMRuntimeRow) -> str:
        return self._jcc_axis_response(
            current_axis=row.le_axis,
            truth_axis=self.truth.le_axis,
            axis_step=row.axis_step,
        )

    def _jcc_axis_response(
        self,
        current_axis: Optional[float],
        truth_axis: float,
        axis_step: Optional[float],
    ) -> str:
        if current_axis is None:
            return "CANT_TELL"

        err = _axis_error(current_axis, truth_axis)
        step = float(axis_step or 0)

        if err <= max(1.0, step / 2.0):
            return "SAME"

        # In your FSM, BETTER_1 moves axis positive, BETTER_2 negative.
        pos_err = _axis_error((float(current_axis) + step) % 180, truth_axis)
        neg_err = _axis_error((float(current_axis) - step) % 180, truth_axis)

        if pos_err < neg_err:
            return "BETTER_1"
        if neg_err < pos_err:
            return "BETTER_2"
        return "CANT_TELL"

    # -----------------------------
    # JCC Power
    # -----------------------------

    def _respond_jcc_power_re(self, row: FSMRuntimeRow) -> str:
        return self._jcc_power_response(
            current_cyl=row.re_cyl,
            truth_cyl=self.truth.re_cyl,
            cyl_step=row.cyl_step,
        )

    def _respond_jcc_power_le(self, row: FSMRuntimeRow) -> str:
        return self._jcc_power_response(
            current_cyl=row.le_cyl,
            truth_cyl=self.truth.le_cyl,
            cyl_step=row.cyl_step,
        )

    def _jcc_power_response(
        self,
        current_cyl: Optional[float],
        truth_cyl: float,
        cyl_step: Optional[float],
    ) -> str:
        if current_cyl is None:
            return "CANT_TELL"

        step = float(cyl_step or 0.25)
        err = float(current_cyl) - float(truth_cyl)

        if abs(err) <= step / 2.0:
            return "SAME"

        # In your FSM:
        # BETTER_1 => + cyl step
        # BETTER_2 => - cyl step
        plus_err = abs((float(current_cyl) + step) - float(truth_cyl))
        minus_err = abs((float(current_cyl) - step) - float(truth_cyl))

        if plus_err < minus_err:
            return "BETTER_1"
        if minus_err < plus_err:
            return "BETTER_2"
        return "CANT_TELL"

    # -----------------------------
    # Duochrome
    # -----------------------------

    def _respond_duochrome_re(self, row: FSMRuntimeRow) -> str:
        return self._duochrome_response(row.re_sph, self.truth.re_sph)

    def _respond_duochrome_le(self, row: FSMRuntimeRow) -> str:
        return self._duochrome_response(row.le_sph, self.truth.le_sph)

    def _duochrome_response(self, current_sph: Optional[float], truth_sph: float) -> str:
        if current_sph is None:
            return "CANT_TELL"

        err = float(current_sph) - float(truth_sph)

        # In your FSM:
        # RED_CLEARER   -> dS = -0.25  (more minus)
        # GREEN_CLEARER -> dS = +0.25  (more plus)
        #
        # Therefore:
        # if current is too plus relative to truth, return RED_CLEARER
        # if current is too minus relative to truth, return GREEN_CLEARER

        if err > 0.125:
            return "RED_CLEARER"

        if err < -0.125:
            return "GREEN_CLEARER"

        return "EQUAL"

    # -----------------------------
    # Binocular balance
    # -----------------------------

    def _respond_binocular_balance(self, row: FSMRuntimeRow) -> str:
        re_err = abs((row.re_sph or 0.0) - self.truth.re_sph)
        le_err = abs((row.le_sph or 0.0) - self.truth.le_sph)

        if abs(re_err - le_err) <= 0.125:
            return "SAME"

        # TOP_CLEARER adds plus to RE in your FSM.
        # BOTTOM_CLEARER adds plus to LE.
        if re_err > le_err:
            return "TOP_CLEARER"
        return "BOTTOM_CLEARER"

    # -----------------------------
    # Near
    # -----------------------------

    def _respond_near_re(self, row: FSMRuntimeRow) -> str:
        err = abs((row.add_r or 0.0) - self.truth.add_r)
        return "READABLE" if err <= 0.25 else "BLURRY"

    def _respond_near_le(self, row: FSMRuntimeRow) -> str:
        err = abs((row.add_l or 0.0) - self.truth.add_l)
        return "READABLE" if err <= 0.25 else "BLURRY"

    def _respond_near_binoc(self, row: FSMRuntimeRow) -> str:
        re_err = abs((row.add_r or 0.0) - self.truth.add_r)
        le_err = abs((row.add_l or 0.0) - self.truth.add_l)
        return "TARGET_OK" if max(re_err, le_err) <= 0.25 else "NOT_CLEAR"