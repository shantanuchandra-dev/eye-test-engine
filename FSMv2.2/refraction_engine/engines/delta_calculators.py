from typing import Optional


def wrap_axis(axis: Optional[float]) -> Optional[float]:
    if axis is None:
        return None
    while axis < 0:
        axis += 180
    while axis > 180:
        axis -= 180
    return axis


def coarse_sphere_delta(response_value: str, sph_step: float) -> float:
    if response_value in ("NOT_READABLE", "BLURRY"):
        return -float(sph_step)
    return 0.0


def jcc_axis_delta(response_value: str, axis_step: float, positive_for_better_1: bool = True) -> float:
    if response_value == "BETTER_1":
        return float(axis_step) if positive_for_better_1 else -float(axis_step)
    if response_value == "BETTER_2":
        return -float(axis_step) if positive_for_better_1 else float(axis_step)
    return 0.0


def jcc_power_cyl_delta(response_value: str, cyl_step: float) -> float:
    if response_value == "BETTER_1":
        return float(cyl_step)
    if response_value == "BETTER_2":
        return -float(cyl_step)
    return 0.0


def jcc_power_sphere_compensation(
    current_cyl: Optional[float],
    proposed_cyl_delta: float,
    compensation_step: float = 0.25,
    threshold_step: float = 0.5,
) -> float:
    if current_cyl is None:
        return 0.0

    old_cyl = float(current_cyl)
    new_cyl = old_cyl + float(proposed_cyl_delta)

    old_half = int(abs(old_cyl) / threshold_step)
    new_half = int(abs(new_cyl) / threshold_step)

    if new_half > old_half and abs(new_cyl) > abs(old_cyl):
        return float(compensation_step)

    if new_half < old_half and abs(new_cyl) < abs(old_cyl):
        return -float(compensation_step)

    return 0.0


def duochrome_sphere_delta(
    response_value: str,
    endpoint_bias_policy: str,
    equal_confirmation_reached: bool,
    calibration,
) -> float:
    if response_value == "RED_CLEARER":
        return float(calibration.get("duochrome_red_step", -0.25))

    if response_value == "GREEN_CLEARER":
        return float(calibration.get("duochrome_green_step", 0.25))

    if response_value == "EQUAL" and equal_confirmation_reached:
        if endpoint_bias_policy == "Undercorrect":
            return float(calibration.get("undercorrect_step", 0.25))
        if endpoint_bias_policy == "Overcorrect":
            return -float(calibration.get("overcorrect_step", 0.25))

    return 0.0


def binocular_balance_delta(response_value: str, calibration, top_eye: bool = True) -> float:
    step = float(calibration.get("bino_balance_step", 0.25))
    if top_eye and response_value == "TOP_CLEARER":
        return step
    if (not top_eye) and response_value == "BOTTOM_CLEARER":
        return step
    return 0.0


def near_add_delta(response_value: str, calibration) -> float:
    step = float(calibration.get("near_add_step", 0.25))
    if response_value in ("NOT_READABLE", "BLURRY", "NOT_CLEAR"):
        return step
    return 0.0