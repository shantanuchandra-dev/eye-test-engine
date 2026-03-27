"""
All static strings that may be spoken via TTS in the exam (FSM prompts + UI chrome).

Dynamic retry lines like "I didn't catch that. <question>" are covered at playback time
by splitting known prefixes (see RETRY_PREFIXES_EN / RETRY_PREFIXES_HI in app.js) and
concatenating two cached clips.
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

# Hardcoded in frontend/app.js (not in FSM Python modules)
APP_UI_TTS_PHRASES: tuple[str, ...] = (
    # Must match frontend/app.js LANG_SELECT_SPOKEN_TEXT (display + TTS use the same string).
    "Please select your preferred language / कृपया अपनी भाषा चुनें",
    "Please select your preferred language. English or Hindi?",
    "Congratulations! Your eye test is complete. Thank you.",
    "बधाई हो! आपका आई टेस्ट पूरा हो गया है। धन्यवाद।",
    "This test requires optometrist review.",
    "इस टेस्ट को ऑप्टोमेट्रिस्ट की समीक्षा की आवश्यकता है।",
    "This is Dot Chart. First option. Look carefully.",
    "यह विकल्प 1 है। ध्यान से देखिए।",
    "This is second option. Which is better? Say first, second, same, or repeat.",
    "यह विकल्प 2 है। कौन सा बेहतर है? पहला, दूसरा, समान, या फिर से कहिए।",
)


def collect_all_tts_phrases() -> list[str]:
    """Return sorted unique phrases to pre-generate with ElevenLabs."""
    from fsm.audio.response_matching import (
        COMPACT_LOCALIZED_VOICE_PROMPTS,
        FSM_AUDIO_RESPONSE_LIBRARY_V1,
        LOCALIZED_LANGUAGE_SELECTION_PROMPTS,
        LOCALIZED_VOICE_PROMPTS,
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

    for _rt, txt in FSM_AUDIO_RESPONSE_LIBRARY_V1["fallback_policy"]["reprompt_texts"].items():
        phrases.add(txt)

    # Mirrors _reprompt_text() compact_map in response_matching.py
    phrases.update(
        {
            "Please say only: clear, blurry, or repeat.",
            "Please say only: one, two, same, or repeat.",
            "Please say only: red, green, same, or repeat.",
            "Please say only: top, bottom, same, or repeat.",
        }
    )
    phrases.add("Please repeat one of the shown response options.")

    for cfg in COMPACT_PROMPT_CONFIG.values():
        phrases.add(cfg["question"])

    phrases.update(APP_UI_TTS_PHRASES)
    phrases.update(RETRY_PREFIXES_EN)
    phrases.update(RETRY_PREFIXES_HI)

    return sorted(phrases)


def phrase_set() -> FrozenSet[str]:
    return frozenset(collect_all_tts_phrases())
