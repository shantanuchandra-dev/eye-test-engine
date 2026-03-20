"""Snellen chart letter matching for coarse sphere phases (FSM B / D).

Maps patient utterances that sound like reading the chart line to READABLE /
BLURRY / NOT_READABLE using longest common subsequence (LCS) against the
expected letter sequence for the current chart_param.
"""

from __future__ import annotations

import re
from typing import List, Union

# Listen duration for Coarse Sphere / Distance Vision (chart reading) phases
COARSE_SPHERE_LISTEN_SECONDS = 10
COARSE_SPHERE_SECONDS_PER_LETTER = 1.2
COARSE_SPHERE_MIN_SECONDS = 5
COARSE_SPHERE_MAX_SECONDS = 25

# Words/phrases that indicate a subjective answer, not spelling chart letters.
_SUBJECTIVE_TOKENS = frozenset({
    "clear", "readable", "blur", "blurry", "yes", "no", "okay", "ok", "fine",
    "good", "visible", "worse", "better", "same", "not", "cant", "cannot",
    "nothing", "hazy", "fuzzy", "sharp", "equal", "identical", "help", "stop",
    "read", "seeing", "see", "looks", "much", "about", "both", "tell",
    "haan", "theek", "nahi", "saaf", "dhundhla",
})


def listen_seconds(chart_letters: Union[str, List[str], None]) -> float:
    """Compute listen duration for Coarse Sphere based on number of letters in the chart line."""
    if not chart_letters:
        return COARSE_SPHERE_LISTEN_SECONDS
    n = len(chart_letters) if isinstance(chart_letters, str) else sum(len(s) for s in chart_letters)
    sec = COARSE_SPHERE_MIN_SECONDS + COARSE_SPHERE_SECONDS_PER_LETTER * n
    return min(COARSE_SPHERE_MAX_SECONDS, max(COARSE_SPHERE_MIN_SECONDS, sec))


CHART_LETTERS_MAP = {
    "400": ["E"],
    "200_150": "ENHSLC",
    "100_80": "HBVPHT",
    "70_60_50": "VLNEADAOFCEGNDH",
    "40_30_25": "FZBDEOFLCTAPEOF",
    "20_15_10": "TZVECOHPNTVLFTH",
    "20_20_20": "EVOTLTBGABHNFZC",
}


def resolve_chart_letters(chart_param: str) -> str:
    """Return the expected letter string for a chart_param key."""
    key = str(chart_param).strip()
    raw = CHART_LETTERS_MAP.get(key)
    if raw is None:
        raw = CHART_LETTERS_MAP["200_150"]
    if isinstance(raw, list):
        return "".join(raw)
    return str(raw)


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= c <= "\u097f" for c in text)


def _letter_tokens(transcript: str) -> List[str]:
    """Split transcript into cleaned tokens (STT often emits 'c,' or 'a, b' with commas)."""
    tl = transcript.lower().strip()
    if not tl:
        return []
    # Split on whitespace and common phrase punctuation between letters
    parts = re.split(r"[\s,;]+", tl)
    out: List[str] = []
    for p in parts:
        cleaned = re.sub(r"[^a-z0-9]", "", p)
        if cleaned:
            out.append(cleaned)
    return out


def looks_like_chart_letter_utterance(transcript: str, chart_param: str) -> bool:
    """True if the transcript is likely the patient naming letters, not a subjective phrase."""
    if _has_devanagari(transcript):
        return False
    tokens = _letter_tokens(transcript)
    if not tokens:
        return False

    if len(tokens) == 1:
        w = tokens[0]
        if w in _SUBJECTIVE_TOKENS:
            return False
        if chart_param == "400" and w == "e":
            return True
        if len(w) >= 4 and w.isalpha() and w not in _SUBJECTIVE_TOKENS:
            return True
        return False

    # Multiple short chunks — typical letter-by-letter reading (commas no longer break this)
    if all(len(t) <= 3 and t.isalpha() for t in tokens):
        if any(t in _SUBJECTIVE_TOKENS for t in tokens):
            return False
        return True

    return False


class ChartReadingDetector:
    """
    Detects whether text represents chart reading (Snellen-style letter sequences)
    and parses them into per-line segments. Also matches utterances to chart lines.
    """

    def __init__(self, min_letters: int = 3, max_letters_per_line: int = 6):
        self.min_letters = min_letters
        self.max_letters_per_line = max_letters_per_line

    def _normalize_chart_utterance(self, text: str) -> str:
        """Keep only letters and digits, uppercase, order preserved."""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _lcs_length(self, a: str, b: str) -> int:
        """Length of longest common subsequence of a and b (order preserved)."""
        na, nb = len(a), len(b)
        prev = [0] * (nb + 1)
        for i in range(1, na + 1):
            curr = [0] * (nb + 1)
            for j in range(1, nb + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[nb]

    def utterance_matches_chart(self, utterance: str, one_chart_letters: str) -> str:
        """
        Return match: READABLE / BLURRY / NOT_READABLE.
        Uses longest common subsequence (LCS) of chart and utterance: fraction of
        chart letters that appear in the utterance in order. Extra letters in the
        utterance (insertions) do not penalize.
        """
        u = self._normalize_chart_utterance(utterance)
        c = self._normalize_chart_utterance(one_chart_letters)
        n = len(c)
        if n == 0:
            return "READABLE"
        correct = self._lcs_length(c, u)
        percentage_correct = correct / n
        if percentage_correct >= 0.8:
            return "READABLE"
        if percentage_correct >= 0.5:
            return "BLURRY"
        return "NOT_READABLE"

    def preprocess_text(self, text: str) -> str:
        """Remove special characters and spaces for matching."""
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = text.replace(".", "")
        text = text.replace(" ", "")
        return text.upper()

    def get_chart_intent(self, _options: List[str], text: str, chart_letters: str) -> str:
        """Map utterance to READABLE / BLURRY / NOT_READABLE from chart LCS."""
        utterance = self.preprocess_text(text)
        return self.utterance_matches_chart(utterance, chart_letters)
