"""
All static strings that may be spoken via TTS in the exam (FSM prompts + UI chrome).

Dynamic retry lines like "I didn't catch that. <question>" are covered at playback time
by splitting known prefixes (see RETRY_PREFIXES_EN / RETRY_PREFIXES_HI in app.js) and
concatenating two cached clips.

Phrases that include a transition preface (minutes left) use the same templates as
session_orchestrator._build_transition_preface when no patient name is present.
"""

from __future__ import annotations

from typing import FrozenSet

# Prefixes concatenated with the current question in frontend/app.js (voice retry flows)
RETRY_PREFIXES_EN: tuple[str, ...] = (
    "Let me repeat. ",
    "I could not hear you. ",
    "I didn't catch that. ",
)
RETRY_PREFIXES_HI: tuple[str, ...] = (
    "फिर से सुनिए। ",
    "सुनाई नहीं दिया। ",
    "समझ नहीं आया। ",
)

# ── frontend/app.js: LANG_SELECT, terminal speech, JCC auto-flip (handleAutoFlip) ──
APP_UI_TTS_PHRASES: tuple[str, ...] = (
    "Please select your preferred language / कृपया अपनी भाषा चुनें",
    "Congratulations! Your eye test is complete. Thank you.",
    "Congratulations! Your eye test is complete. You preferred the achieved prescription over the PGP. Thank you.",
    "Congratulations! Your eye test is complete. You preferred the PGP. Thank you.",
    "बधाई हो! आपका आई टेस्ट पूरा हो गया है। धन्यवाद।",
    "बधाई हो। आपका आई टेस्ट पूरा हो गया है। आपने पी जी पी की तुलना में प्राप्त आर एक्स को पसंद किया। धन्यवाद।",
    "बधाई हो। आपका आई टेस्ट पूरा हो गया है। आपने पी जी पी को पसंद किया। धन्यवाद।",
    "This test requires optometrist review.",
    "इस टेस्ट को ऑप्टोमेट्रिस्ट की समीक्षा की आवश्यकता है।",
    "Here is the first option. Take your time and look carefully.",
    "यह पहला विकल्प है। आराम से ध्यान से देखिए।",
    # Flip 2 prefix-only (compound with normalized question body after JCC flip)
    "And now the second option.",
    "अब दूसरा विकल्प है।",
    "And now the second option. Say first option, second option, both same, or repeat.",
    "अब दूसरा विकल्प है। पहला, दूसरा, समान, या फिर से कहिए।",
)

# After frontend/app.js buildSpokenQuestionText() normalization (English TTS path)
CANONICAL_EN_SPOKEN_QUESTIONS: tuple[str, ...] = (
    "Please read the line.",
    "Can you read the line now, or is it still blurry?",
    "Read it now, or is it still blurry?",
    "Can you read all the letters in this line, or are they blurry?",
    "Read the line, or say blurry.",
    "Did that look better now? Say yes or no.",
    "Better now? Say yes or no.",
    "Which looks clearer: first option, second option, both same, or repeat?",
    "First option, second option, both same, or repeat?",
    "Which side looks clearer: green side, red side, or both same?",
    "Which line looks clearer: bottom line, top line, or both same?",
    "Green side, red side, or both same?",
    "Bottom line, top line, or both same?",
    "Do the letters look clear, or blurry?",
    "Please compare the two choices. Which one looks clearer or sharper? say first option, second option, both same, or repeat.",
    "Please compare the green and red sides. Which side looks sharper and darker? Say green side, red side, both same, or repeat.",
    "Please compare the letters on the bottom and the top line. Which line looks sharper? Say bottom line, top line, both same, or repeat.",
    "Please look at the near text. Are the letters clear, blurry, or do you want me to repeat?",
    "Option 1. Please observe the line carefully.",
    "Option 2. Please observe the line carefully.",
    "Which option was better, first option or second option? Say first option, second option, or repeat.",
    "Which was better, first option, second option, or repeat?",
)


def _ensure_speech_ending(text: str) -> str:
    t = " ".join(text.split()).strip()
    if not t:
        return ""
    return t if t[-1] in ".!?" else f"{t}."


def _join_speech_parts(*parts: str) -> str:
    """Match frontend joinSpeechParts (non-empty parts only)."""
    normalized: list[str] = []
    for p in parts:
        t = " ".join((p or "").split()).strip()
        if t:
            normalized.append(t)
    if not normalized:
        return ""
    for i in range(len(normalized) - 1):
        normalized[i] = _ensure_speech_ending(normalized[i])
    return " ".join(normalized)


def _english_transition_preface(minutes_left: int) -> str:
    label = "minute" if minutes_left == 1 else "minutes"
    return (
        f"You are doing great. Please blink a few times. About {minutes_left} {label} left."
    )


def _hindi_transition_preface(minutes_left: int) -> str:
    return (
        f"आप बहुत अच्छा कर रहे हैं। कृपया कुछ बार पलक झपकाइए। लगभग {minutes_left} मिनट बाकी हैं।"
    )


def _collect_transition_preface_only(*, max_minutes: int = 20) -> set[str]:
    """Standalone transition lines (no patient name) for compound TTS with flip1 / read-line."""
    out: set[str] = set()
    for n in range(1, max_minutes + 1):
        out.add(_english_transition_preface(n))
        out.add(_hindi_transition_preface(n))
    return out


def _collect_transition_preface_variants(*, en_tail: str, hi_tail: str, max_minutes: int = 20) -> set[str]:
    """Preface + tail for common post-transition TTS (no patient name)."""
    out: set[str] = set()
    for n in range(1, max_minutes + 1):
        out.add(_join_speech_parts(_english_transition_preface(n), en_tail))
        out.add(_join_speech_parts(_hindi_transition_preface(n), hi_tail))
    return out


def collect_all_tts_phrases() -> list[str]:
    """Return sorted unique phrases to pre-generate with ElevenLabs."""
    from fsm.audio.response_matching import (
        COMPACT_LOCALIZED_VOICE_PROMPTS,
        FSM_AUDIO_RESPONSE_LIBRARY_V1,
        LOCALIZED_LANGUAGE_SELECTION_PROMPTS,
        LOCALIZED_VOICE_PROMPTS,
        _reprompt_text,
    )
    from fsm.engines.refraction_fsm_engine import COMPACT_PROMPT_CONFIG

    phrases: set[str] = set()

    for _family, langs in COMPACT_LOCALIZED_VOICE_PROMPTS.items():
        for _lang, modes in langs.items():
            phrases.add(modes["initial"])
            phrases.add(modes["retry"])

    for _k, v in LOCALIZED_LANGUAGE_SELECTION_PROMPTS.items():
        phrases.add(v)

    for _lang, by_rt in LOCALIZED_VOICE_PROMPTS.items():
        for _rt, modes in by_rt.items():
            phrases.add(modes["initial"])
            phrases.add(modes["retry"])

    for rt in FSM_AUDIO_RESPONSE_LIBRARY_V1["fallback_policy"]["reprompt_texts"]:
        phrases.add(_reprompt_text(rt))

    phrases.add("Please repeat one of the shown response options.")

    for cfg in COMPACT_PROMPT_CONFIG.values():
        phrases.add(cfg["question"])

    phrases.update(APP_UI_TTS_PHRASES)
    phrases.update(RETRY_PREFIXES_EN)
    phrases.update(RETRY_PREFIXES_HI)
    phrases.update(CANONICAL_EN_SPOKEN_QUESTIONS)

    # JCC auto-flip (states E, F, H, I): flip2 uses full English question from DOM + Hindi compact prompt
    flip2_en = "And now the second option. "
    flip2_hi = "अब दूसरा विकल्प है। "
    jcc_compare_q = COMPACT_PROMPT_CONFIG["E"]["question"]
    phrases.add(flip2_en + jcc_compare_q)
    hi_compare = COMPACT_LOCALIZED_VOICE_PROMPTS["comparison_4way"]["hi"]["initial"]
    phrases.add(flip2_hi + hi_compare)

    # Transition preface + common following prompts (session_orchestrator._build_transition_preface)
    phrases.update(
        _collect_transition_preface_variants(
            en_tail="Please read the line.",
            hi_tail="अक्षर चार्ट। लाइन पढ़िए।",
        )
    )
    phrases.update(
        _collect_transition_preface_variants(
            en_tail="Here is the first option. Take your time and look carefully.",
            hi_tail="यह पहला विकल्प है। आराम से ध्यान से देखिए।",
        )
    )
    # Preface-only clips (frontend plays preface + flip1 body as two MP3s when combined hash misses)
    phrases.update(_collect_transition_preface_only(max_minutes=20))

    return sorted(phrases)


def phrase_set() -> FrozenSet[str]:
    return frozenset(collect_all_tts_phrases())
