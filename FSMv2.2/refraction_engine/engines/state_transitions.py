def compute_phase_max(state: str, dv_expected_convergence_time: str, calibration) -> int:
    speed = dv_expected_convergence_time

    if speed == "Fast":
        if state in ("B", "D"):
            return int(calibration.get("timeout_coarse_fast", 24))
        if state in ("E", "F", "H", "I"):
            return int(calibration.get("timeout_jcc_fast", 10))
        if state in ("G", "J"):
            return int(calibration.get("timeout_duochrome_fast", 8))
        if state == "K":
            return int(calibration.get("timeout_bino_fast", 10))
        if state in ("P", "Q", "R"):
            return int(calibration.get("timeout_near_fast", 16))
        return int(calibration.get("timeout_coarse_fast", 24))

    if speed == "Normal":
        if state in ("B", "D"):
            return int(calibration.get("timeout_coarse_normal", 36))
        if state in ("E", "F", "H", "I"):
            return int(calibration.get("timeout_jcc_normal", 14))
        if state in ("G", "J"):
            return int(calibration.get("timeout_duochrome_normal", 10))
        if state == "K":
            return int(calibration.get("timeout_bino_normal", 14))
        if state in ("P", "Q", "R"):
            return int(calibration.get("timeout_near_normal", 22))
        return int(calibration.get("timeout_coarse_normal", 36))

    if state in ("B", "D"):
        return int(calibration.get("timeout_coarse_slow", 48))
    if state in ("E", "F", "H", "I"):
        return int(calibration.get("timeout_jcc_slow", 18))
    if state in ("G", "J"):
        return int(calibration.get("timeout_duochrome_slow", 12))
    if state == "K":
        return int(calibration.get("timeout_bino_slow", 18))
    if state in ("P", "Q", "R"):
        return int(calibration.get("timeout_near_slow", 28))
    return int(calibration.get("timeout_coarse_slow", 48))


def compute_next_state(context: dict) -> str:
    state = context["state"]
    response = context["response"]
    timeout = context["timeout"]
    re_escalate = context["re_escalate"]
    le_escalate = context["le_escalate"]
    same_streak = context["same_streak"]
    power_same_n = context["power_same_n"]
    duo_equal_n = context["duo_equal_n"]
    duo_flip = context["duo_flip"]
    duo_max = context["duo_max"]
    bino_same_n = context["bino_same_n"]
    bino_flip = int(context.get("bino_flip", 0))
    bino_flip_limit = int(context.get("bino_flip_limit", 1))
    skip_bino_balance = bool(context.get("skip_bino_balance", False))
    near_target = context["near_target"]
    near_required = context["near_required"]
    immediate_review = context["immediate_review"]
    chart_idx = int(context["chart_idx"])
    target_chart_idx = int(context["target_chart_idx"])
    jcc_power_flip_limit_hit = bool(context["jcc_power_flip_limit_hit"])

    if immediate_review:
        return "ESCALATE"

    if state == "A":
        if response in ("NOT_READABLE", "BLURRY"):
            return "B"
        if response == "READABLE" and chart_idx >= target_chart_idx:
            return "B"
        return "A"

    if state == "B":
        if re_escalate or timeout:
            return "ESCALATE"
        if response == "READABLE" and chart_idx >= target_chart_idx:
            return "E"
        return "B"

    if state == "E":
        if timeout:
            return "ESCALATE"
        if response in ("SAME", "CANT_TELL"):
            return "F"
        return "E"

    if state == "F":
        if timeout:
            return "ESCALATE"
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
        if response == "READABLE" and chart_idx >= target_chart_idx:
            return "H"
        return "D"

    if state == "H":
        if timeout:
            return "ESCALATE"
        if response in ("SAME", "CANT_TELL"):
            return "I"
        return "H"

    if state == "I":
        if timeout:
            return "ESCALATE"
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
        if response in (near_target, "SAME"):
            return "END"
        return "R"

    return "END"