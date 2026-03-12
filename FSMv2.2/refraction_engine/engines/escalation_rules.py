def should_escalate_re(
    anomaly_watch: bool,
    current_sph: float,
    start_sph: float,
    ar_sph: float,
    max_delta_from_start: float,
    max_delta_from_ar: float,
) -> bool:
    if not anomaly_watch:
        return False

    return (
        abs(current_sph - start_sph) > max_delta_from_start
        or abs(current_sph - ar_sph) > max_delta_from_ar
    )


def should_escalate_le(
    anomaly_watch: bool,
    current_sph: float,
    start_sph: float,
    ar_sph: float,
    max_delta_from_start: float,
    max_delta_from_ar: float,
) -> bool:
    if not anomaly_watch:
        return False

    return (
        abs(current_sph - start_sph) > max_delta_from_start
        or abs(current_sph - ar_sph) > max_delta_from_ar
    )


def phase_timeout_reached(phase_step_count: int, phase_max: int) -> bool:
    return phase_step_count >= phase_max