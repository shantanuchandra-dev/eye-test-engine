def compute_phase_max(state: str, dv_expected_convergence_time: str, calibration) -> int:
    speed = dv_expected_convergence_time

    if speed == "Fast":
        if state in ("B", "C", "D", "L"):
            return int(calibration.get("timeout_coarse_fast", 24))
        if state in ("E", "F", "H", "I"):
            return int(calibration.get("timeout_jcc_fast", 10))
        if state in ("G", "J"):
            return int(calibration.get("timeout_duochrome_fast", 8))
        if state == "K":
            return int(calibration.get("timeout_bino_fast", 10))
        if state in ("P", "Q", "R"):
            return int(calibration.get("timeout_near_fast", 16))
        if state in ("S", "T", "U"):
            return int(calibration.get("timeout_coarse_fast", 24))
        return int(calibration.get("timeout_coarse_fast", 24))

    if speed == "Normal":
        if state in ("B", "C", "D", "L"):
            return int(calibration.get("timeout_coarse_normal", 36))
        if state in ("E", "F", "H", "I"):
            return int(calibration.get("timeout_jcc_normal", 14))
        if state in ("G", "J"):
            return int(calibration.get("timeout_duochrome_normal", 10))
        if state == "K":
            return int(calibration.get("timeout_bino_normal", 14))
        if state in ("P", "Q", "R"):
            return int(calibration.get("timeout_near_normal", 22))
        if state in ("S", "T", "U"):
            return int(calibration.get("timeout_coarse_normal", 36))
        return int(calibration.get("timeout_coarse_normal", 36))

    if state in ("B", "C", "D", "L"):
        return int(calibration.get("timeout_coarse_slow", 48))
    if state in ("E", "F", "H", "I"):
        return int(calibration.get("timeout_jcc_slow", 18))
    if state in ("G", "J"):
        return int(calibration.get("timeout_duochrome_slow", 12))
    if state == "K":
        return int(calibration.get("timeout_bino_slow", 18))
    if state in ("P", "Q", "R"):
        return int(calibration.get("timeout_near_slow", 28))
    if state in ("S", "T", "U"):
        return int(calibration.get("timeout_coarse_slow", 48))
    return int(calibration.get("timeout_coarse_slow", 48))


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
    va_confirm_completed = bool(context.get("va_confirm_completed", False))
    final_compare_enabled = bool(context.get("final_compare_enabled", False))
    final_compare_round = int(context.get("final_compare_round", 0))

    # FSM v2.3 additions
    axis_same_required = int(context.get("axis_same_required", 2))
    axis_flip_count = int(context.get("axis_flip_count", 0))
    axis_flip_max = int(context.get("axis_flip_max", 3))

    if immediate_review:
        return "ESCALATE"


    if state == "B":
        if re_escalate or timeout:
            return "ESCALATE"
        if context.get("coarse_endpoint_reached") or context.get("coarse_oscillation_detected"):
            return "E"
        if response == "REPEAT":
            return "B"
        return "B"

    if state == "E":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "E"
        if same_streak >= axis_same_required:
            return "F"
        if axis_flip_count >= axis_flip_max:
            return "F"
        return "E"

    if state == "F":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "F"
        # FSM v3.1: hard stop when cylinder reaches zero (no positive cyl allowed)
        if jcc_cyl_at_zero:
            return "G"
        if same_streak >= power_same_n or jcc_power_flip_limit_hit:
            return "G"
        return "F"

    if state == "G":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "G"
        if same_streak >= duo_equal_n or duo_flip >= duo_max:
            return "C"
        return "G"

    if state == "C":
        if response == "REPEAT":
            return "C"
        if timeout or va_confirm_completed:
            return "D"
        return "C"

    if state == "D":
        if le_escalate or timeout:
            return "ESCALATE"
        if context.get("coarse_endpoint_reached") or context.get("coarse_oscillation_detected"):
            return "H"
        if response == "REPEAT":
            return "D"
        return "D"

    if state == "H":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "H"
        if same_streak >= axis_same_required:
            return "I"
        if axis_flip_count >= axis_flip_max:
            return "I"
        return "H"

    if state == "I":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "I"
        # FSM v3.1: hard stop when cylinder reaches zero
        if jcc_cyl_at_zero:
            return "J"
        if same_streak >= power_same_n or jcc_power_flip_limit_hit:
            return "J"
        return "I"

    if state == "J":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "J"
        if same_streak >= duo_equal_n or duo_flip >= duo_max:
            return "L"
        return "J"

    if state == "L":
        if response == "REPEAT":
            return "L"
        if timeout or va_confirm_completed:
            if skip_bino_balance:
                if near_required:
                    return "P"
                return "S" if final_compare_enabled else "END"
            return "K"
        return "L"

    if state == "K":
        if timeout:
            if near_required:
                return "P"
            return "S" if final_compare_enabled else "END"
        if response == "REPEAT":
            return "K"
        # FSM v3.1: exit as soon as stability OR flip limit reached
        if same_streak >= bino_same_n:
            if near_required:
                return "P"
            return "S" if final_compare_enabled else "END"
        if bino_flip >= bino_flip_limit:
            if near_required:
                return "P"
            return "S" if final_compare_enabled else "END"
        return "K"

    if state == "P":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "P"
        if response in ("CLEAR", "READABLE", "SAME"):
            return "Q"
        return "P"

    if state == "Q":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "Q"
        if response in ("CLEAR", "READABLE", "SAME"):
            return "R"
        return "Q"

    if state == "R":
        if timeout:
            return "ESCALATE"
        if response == "REPEAT":
            return "R"
        # FSM v3.1: terminate only when target condition met
        if response in (near_target, "CLEAR", "TARGET_OK", "SAME"):
            return "S" if final_compare_enabled else "END"
        return "R"

    if state == "S":
        if timeout:
            return "END"
        if response == "REPEAT":
            return "S"
        return "T"

    if state == "T":
        if timeout:
            return "END"
        if response == "REPEAT":
            return "T"
        return "U"

    if state == "U":
        if timeout:
            return "END"
        if response == "REPEAT":
            return "S"
        if final_compare_round >= 2:
            return "END"
        return "S"

    return "END"
