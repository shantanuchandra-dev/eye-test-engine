def should_escalate_re(
    anomaly_watch: bool,
    current_sph: float,
    start_sph: float,
    ar_sph: float,
    max_delta_from_start: float,
    max_delta_from_ar: float,
    phase_step_count: int = 0,
    min_steps_before_escalation: int = 6,
) -> bool:
    # Do not escalate early in the phase
    if not anomaly_watch or phase_step_count < min_steps_before_escalation:
        return False

    delta_start = abs(current_sph - start_sph)
    delta_ar = abs(current_sph - ar_sph)

    # Allow exploration if still within AR-safe bounds
    if delta_ar <= max_delta_from_ar:
        return False

    # Escalate only if both bounds are violated
    return delta_start > max_delta_from_start and delta_ar > max_delta_from_ar


def should_escalate_le(
    anomaly_watch: bool,
    current_sph: float,
    start_sph: float,
    ar_sph: float,
    max_delta_from_start: float,
    max_delta_from_ar: float,
    phase_step_count: int = 0,
    min_steps_before_escalation: int = 6,
) -> bool:
    # Do not escalate early in the phase
    if not anomaly_watch or phase_step_count < min_steps_before_escalation:
        return False

    delta_start = abs(current_sph - start_sph)
    delta_ar = abs(current_sph - ar_sph)

    # Allow exploration if still within AR-safe bounds
    if delta_ar <= max_delta_from_ar:
        return False

    # Escalate only if both bounds are violated
    return delta_start > max_delta_from_start and delta_ar > max_delta_from_ar


def phase_timeout_reached(phase_step_count: int, phase_max: int) -> bool:
    return phase_step_count >= phase_max
