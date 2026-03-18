def compute_phase_max(state: str, dv_expected_convergence_time: str, calibration) -> int:
    speed = dv_expected_convergence_time

    if speed == "Fast":
        if state in ("B", "D"):
            return int(calibration.get("timeout_coarse_fast", 30))
        if state in ("E", "F", "H", "I"):
            return int(calibration.get("timeout_jcc_fast", 10))
        if state in ("G", "J"):
            return int(calibration.get("timeout_duochrome_fast", 12))
        if state == "K":
            return int(calibration.get("timeout_bino_fast", 10))
        if state in ("P", "Q", "R"):
            return int(calibration.get("timeout_near_fast", 16))
        return int(calibration.get("timeout_coarse_fast", 30))

    if speed == "Normal":
        if state in ("B", "D"):
            return int(calibration.get("timeout_coarse_normal", 48))
        if state in ("E", "F", "H", "I"):
            return int(calibration.get("timeout_jcc_normal", 14))
        if state in ("G", "J"):
            return int(calibration.get("timeout_duochrome_normal", 14))
        if state == "K":
            return int(calibration.get("timeout_bino_normal", 14))
        if state in ("P", "Q", "R"):
            return int(calibration.get("timeout_near_normal", 22))
        return int(calibration.get("timeout_coarse_normal", 48))

    if state in ("B", "D"):
        return int(calibration.get("timeout_coarse_slow", 60))
    if state in ("E", "F", "H", "I"):
        return int(calibration.get("timeout_jcc_slow", 18))
    if state in ("G", "J"):
        return int(calibration.get("timeout_duochrome_slow", 16))
    if state == "K":
        return int(calibration.get("timeout_bino_slow", 18))
    if state in ("P", "Q", "R"):
        return int(calibration.get("timeout_near_slow", 28))
    return int(calibration.get("timeout_coarse_slow", 60))


def compute_next_state(context: dict) -> str:
    state = context["state"]
    response = context["response"]
    timeout = context["timeout"]
    re_escalate = context["re_escalate"]
    le_escalate = context["le_escalate"]
    same_streak = int(context["same_streak"])
    power_same_n = int(context["power_same_n"])
    duo_equal_n = int(context["duo_equal_n"])
    duo_flip = int(context["duo_flip"])
    duo_max = int(context["duo_max"])
    bino_same_n = int(context["bino_same_n"])
    bino_flip = int(context.get("bino_flip", 0))
    bino_flip_limit = int(context.get("bino_flip_limit", 1))
    skip_bino_balance = bool(context.get("skip_bino_balance", False))
    near_target = context["near_target"]
    near_required = bool(context["near_required"])
    immediate_review = bool(context["immediate_review"])
    chart_idx = int(context["chart_idx"])
    target_chart_idx = int(context["target_chart_idx"])
    jcc_power_flip_limit_hit = bool(context["jcc_power_flip_limit_hit"])
    jcc_cyl_at_zero = bool(context.get("jcc_cyl_at_zero", False))

    # FSM v2.3 additions
    axis_same_required = int(context.get("axis_same_required", 2))
    axis_flip_count = int(context.get("axis_flip_count", 0))
    axis_flip_max = int(context.get("axis_flip_max", 3))

    if immediate_review:
        return "ESCALATE"

    # FSM v3.0: No State A. Test starts directly at B.

    if state == "B":
        if re_escalate or timeout:
            return "ESCALATE"
        if context.get("coarse_endpoint_reached") or context.get("coarse_oscillation_detected"):
            return "E"
        if response == "READABLE" and chart_idx >= target_chart_idx:
            return "E"
        return "B"

    if state == "E":
        if timeout:
            return "ESCALATE"
        if same_streak >= axis_same_required:
            return "F"
        if axis_flip_count >= axis_flip_max:
            return "F"
        return "E"

    if state == "F":
        if timeout:
            return "ESCALATE"
        # FSM v3.0: hard stop when cylinder reaches zero (no positive cyl allowed)
        if jcc_cyl_at_zero:
            return "G"
        if same_streak >= power_same_n or jcc_power_flip_limit_hit:
            return "G"
        return "F"

    if state == "G":
        if timeout:
            return "ESCALATE"
        if same_streak >= duo_equal_n or duo_flip >= duo_max:
            return "D"
        return "G"

    if state == "D":
        if le_escalate or timeout:
            return "ESCALATE"
        if context.get("coarse_endpoint_reached") or context.get("coarse_oscillation_detected"):
            return "H"
        if response == "READABLE" and chart_idx >= target_chart_idx:
            return "H"
        return "D"

    if state == "H":
        if timeout:
            return "ESCALATE"
        if same_streak >= axis_same_required:
            return "I"
        if axis_flip_count >= axis_flip_max:
            return "I"
        return "H"

    if state == "I":
        if timeout:
            return "ESCALATE"
        # FSM v3.0: hard stop when cylinder reaches zero
        if jcc_cyl_at_zero:
            return "J"
        if same_streak >= power_same_n or jcc_power_flip_limit_hit:
            return "J"
        return "I"

    if state == "J":
        if timeout:
            return "ESCALATE"
        if same_streak >= duo_equal_n or duo_flip >= duo_max:
            if skip_bino_balance:
                return "P" if near_required else "END"
            return "K"
        return "J"

    if state == "K":
        if timeout:
            return "P" if near_required else "END"
        if same_streak >= bino_same_n:
            return "P" if near_required else "END"
        if bino_flip >= bino_flip_limit:
            return "P" if near_required else "END"
        return "K"

    if state == "P":
        if timeout:
            return "ESCALATE"
        if response in ("READABLE", "SAME"):
            return "Q"
        return "P"

    if state == "Q":
        if timeout:
            return "ESCALATE"
        if response in ("READABLE", "SAME"):
            return "R"
        return "Q"

    if state == "R":
        if timeout:
            return "ESCALATE"
        if response in (near_target, "SAME", "TARGET_OK"):
            return "END"
        return "R"

    return "END"
