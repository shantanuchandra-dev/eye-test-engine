from typing import Optional, Tuple


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
    """
    FSM v2.3 note:
    Axis stepping policy is now intended to remain fixed at 5 degrees.
    This function remains generic and simply applies the provided axis_step.
    The engine / calibration will ensure that axis_step stays fixed.
    """
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
    start_cyl: Optional[float],
    compensation_step: float = 0.25,
    threshold_step: float = 0.5,
) -> float:
    """
    Sphere compensation during JCC power refinement is based on RELATIVE
    cylinder displacement from the cylinder value at entry into the JCC
    power phase, not on absolute cylinder magnitude.

    Example:
        start at -1.75 cyl
        -1.75 -> -2.00  => 0.25 away from start => no sphere change
        -2.00 -> -2.25  => 0.50 away from start => sphere changes
        -2.25 -> -2.50  => 0.75 away from start => no sphere change
        -2.50 -> -2.75  => 1.00 away from start => sphere changes
    """
    if current_cyl is None or start_cyl is None:
        return 0.0

    old_cyl = float(current_cyl)
    new_cyl = old_cyl + float(proposed_cyl_delta)
    ref_cyl = float(start_cyl)

    old_rel = abs(old_cyl - ref_cyl)
    new_rel = abs(new_cyl - ref_cyl)

    old_bucket = int(old_rel / threshold_step)
    new_bucket = int(new_rel / threshold_step)

    if new_bucket != old_bucket:
        if proposed_cyl_delta > 0:
            return -float(compensation_step)
        if proposed_cyl_delta < 0:
            return float(compensation_step)

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


# ---------------------------------------------------------
# FSM v2.3 additions
# ---------------------------------------------------------

def near_binoc_ou_delta(direction: str, step: float) -> Tuple[float, float]:
    """
    Returns binocular add deltas for both eyes together.

    direction:
        "PLUS"  -> +step OU
        "MINUS" -> -step OU

    Used by FSM v2.3 near binocular logic in state R.
    """
    if direction == "PLUS":
        return float(step), float(step)
    if direction == "MINUS":
        return -float(step), -float(step)
    return 0.0, 0.0
