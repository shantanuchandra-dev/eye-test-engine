DISTANCE_CHART_ORDER = [
    "400",
    "200_150",
    "100_80",
    "70_60_50",
    "40_30_25",
    "20_20_20",
]

def _normalize_chart(chart_param):
    if chart_param is None:
        return ""
    text = str(chart_param).strip()
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"') or
        (text[0] == "'" and text[-1] == "'")
    ):
        text = text[1:-1].strip()
    if text == "202020":
        return "20_20_20"
    if text == "403025":
        return "40_30_25"
    if text == "706050":
        return "70_60_50"
    if text == "10080":
        return "100_80"
    if text == "200150":
        return "200_150"
    return text

def get_chart_index(chart_param: str) -> int:
    text = _normalize_chart(chart_param)
    if text not in DISTANCE_CHART_ORDER:
        return -1
    return DISTANCE_CHART_ORDER.index(text) + 1

def get_next_chart(chart_param: str) -> str:
    text = _normalize_chart(chart_param)
    idx = get_chart_index(text)
    if idx <= 0:
        return text
    if idx >= len(DISTANCE_CHART_ORDER):
        return text
    return DISTANCE_CHART_ORDER[idx]

def target_line_to_chart(target_line: str, calibration) -> str:
    target_line = str(target_line).strip()
    if target_line == "6/6_target":
        raw = calibration.get("distance_target_6_6_chart", "20_20_20")
    else:
        raw = calibration.get("distance_target_6_9_chart", "40_30_25")
    text = _normalize_chart(raw)
    if text not in DISTANCE_CHART_ORDER:
        if target_line == "6/6_target":
            return "20_20_20"
        return "40_30_25"
    return text
