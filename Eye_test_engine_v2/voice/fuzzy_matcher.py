"""Fuzzy matcher: maps patient speech transcripts to FSM response options.

Each response_type has a fixed set of valid options with keyword aliases.
Two-pass matching: exact keyword hit first, then fuzzy fallback via rapidfuzz.
Supports English, romanized Hindi, and Devanagari Hindi keywords.
"""

from rapidfuzz import fuzz, process as rfprocess

# Keyword aliases for each response option, grouped by response_type.
# Keys = FSM option values, values = list of trigger phrases.
KEYWORD_MAP = {
    "READABILITY": {
        "READABLE": [
            "clear", "readable", "can read", "i can see", "yes", "fine",
            "sharp", "good", "visible", "okay", "ok",
            "better", "much better", "improved", "clearer",
            # Romanized Hindi
            "saaf", "dikh raha", "padh paa raha", "haan", "theek",
            # Devanagari Hindi
            "साफ़", "साफ", "दिख रहा", "पढ़ पा रहा", "हाँ", "हां", "ठीक",
            "पढ़ सकता हूँ", "दिखाई दे रहा", "अच्छा",
        ],
        "NOT_READABLE": [
            "not readable", "can't read", "cannot read", "nothing",
            "can't see", "cannot see", "no", "blank", "not visible",
            "worse", "much worse", "worse than before", "can't see anything",
            # Romanized Hindi
            "nahi dikh raha", "padh nahi paa raha", "nahi", "kuch nahi",
            # Devanagari Hindi
            "नहीं दिख रहा", "पढ़ नहीं पा रहा", "नहीं", "कुछ नहीं",
            "दिखाई नहीं दे रहा", "बिल्कुल नहीं",
        ],
        "BLURRY": [
            "blurry", "blurred", "blur", "slightly blurry", "a bit blurry",
            "hazy", "fuzzy", "not clear", "not sharp", "foggy",
            "a little worse", "slightly worse", "bit blurry",
            # Romanized Hindi
            "dhundhla", "thoda dhundhla", "halka",
            # Devanagari Hindi
            "धुंधला", "थोड़ा धुंधला", "हल्का", "धुंधला है",
        ],
    },
    "NEAR_READABILITY": {
        "READABLE": [
            "clear", "readable", "can read", "i can see", "yes", "fine",
            "sharp", "good", "visible", "okay", "ok",
            "better", "much better", "improved", "clearer",
            "saaf", "dikh raha", "padh paa raha", "haan", "theek",
            "साफ़", "साफ", "दिख रहा", "पढ़ पा रहा", "हाँ", "ठीक",
        ],
        "NOT_READABLE": [
            "not readable", "can't read", "cannot read", "nothing",
            "can't see", "cannot see", "no", "blank", "not visible",
            "worse", "much worse", "worse than before",
            "nahi dikh raha", "padh nahi paa raha", "nahi",
            "नहीं दिख रहा", "पढ़ नहीं पा रहा", "नहीं",
        ],
        "BLURRY": [
            "blurry", "blurred", "blur", "slightly blurry", "a bit blurry",
            "hazy", "fuzzy", "not clear", "not sharp", "foggy",
            "a little worse", "slightly worse", "bit blurry",
            "dhundhla", "thoda dhundhla",
            "धुंधला", "थोड़ा धुंधला",
        ],
    },
    "COMPARE_1_2": {
        "BETTER_1": [
            "one", "first", "1", "number one", "the first one",
            "better one", "one is better", "first is better",
            "one was clearer", "one was sharper",
            # Romanized Hindi
            "pehla", "ek", "pehla wala",
            # Devanagari Hindi
            "पहला", "एक", "पहला वाला", "पहला बेहतर",
        ],
        "BETTER_2": [
            "two", "second", "2", "number two", "the second one",
            "better two", "two is better", "second is better",
            "two was clearer", "two was sharper",
            "doosra", "do", "doosra wala",
            "दूसरा", "दो", "दूसरा वाला", "दूसरा बेहतर",
        ],
        "SAME": [
            "same", "equal", "no difference", "both same", "identical",
            "about the same", "both look same", "similar",
            "dono same", "barabar", "ek jaisa",
            "दोनों एक जैसे", "बराबर", "एक जैसा", "समान", "दोनों",
        ],
        "CANT_TELL": [
            "can't tell", "cant tell", "not sure", "don't know",
            "unsure", "hard to tell", "repeat", "show again",
            "pata nahi", "samajh nahi aaya",
            "पता नहीं", "समझ नहीं आया", "फिर से दिखाओ",
        ],
    },
    "COLOR_CHOICE": {
        "RED_CLEARER": [
            "red", "red side", "red is clearer", "red one",
            "left side", "red better",
            "lal", "lal wala",
            "लाल", "लाल वाला", "लाल तरफ़",
        ],
        "GREEN_CLEARER": [
            "green", "green side", "green is clearer", "green one",
            "right side", "green better",
            "hara", "hara wala",
            "हरा", "हरा वाला", "हरी तरफ़",
        ],
        "EQUAL": [
            "same", "equal", "about the same", "both same",
            "no difference", "barabar", "dono same",
            "बराबर", "दोनों एक जैसे", "एक जैसा",
        ],
        "CANT_TELL": [
            "can't tell", "cant tell", "not sure", "don't know",
            "unsure", "hard to tell", "repeat",
            "pata nahi",
            "पता नहीं", "समझ नहीं आया",
        ],
    },
    "TOP_BOTTOM": {
        "TOP_CLEARER": [
            "top", "upper", "top row", "top one", "first row",
            "upar", "upar wala", "top is clearer",
            "ऊपर", "ऊपर वाली", "ऊपर की",
        ],
        "BOTTOM_CLEARER": [
            "bottom", "lower", "bottom row", "bottom one", "second row",
            "neeche", "neeche wala", "bottom is clearer",
            "नीचे", "नीचे वाली", "नीचे की",
        ],
        "SAME": [
            "same", "equal", "about the same", "both same",
            "no difference", "barabar",
            "बराबर", "दोनों एक जैसी", "एक जैसा",
        ],
        "CANT_TELL": [
            "can't tell", "cant tell", "not sure", "don't know",
            "unsure", "hard to tell", "repeat", "pata nahi",
            "पता नहीं", "समझ नहीं आया",
        ],
    },
    "NEAR_BINOC": {
        "TARGET_OK": [
            "clear", "ok", "okay", "fine", "comfortable", "yes",
            "can read", "readable", "good", "theek hai",
            "साफ़", "ठीक है", "हाँ", "आरामदायक", "पढ़ पा रहा",
        ],
        "NOT_CLEAR": [
            "not clear", "blurry", "blur", "no", "can't read",
            "not readable", "not comfortable", "nahi dikh raha",
            "साफ़ नहीं", "धुंधला", "नहीं", "पढ़ नहीं पा रहा",
        ],
    },
}


def match_transcript(transcript: str, response_type: str, threshold: float = 60.0):
    """Match a speech transcript to the best FSM option.

    Args:
        transcript: Raw STT transcript text.
        response_type: FSM response_type (e.g. "READABILITY", "COMPARE_1_2").
        threshold: Minimum fuzzy score to accept a match (0-100).

    Returns:
        (matched_option, confidence) or (None, 0.0) if no match.
    """
    if not transcript or response_type not in KEYWORD_MAP:
        return None, 0.0

    text = transcript.lower().strip()
    keywords = KEYWORD_MAP[response_type]

    # Pass 1: exact substring match
    hits = []
    for option, phrases in keywords.items():
        for phrase in phrases:
            if phrase in text:
                # Prefer longer phrase matches (more specific)
                hits.append((option, len(phrase)))

    if hits:
        # Return the option with the longest matching phrase
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[0][0], 100.0

    # Pass 2: fuzzy match using rapidfuzz
    # Build flat list of (phrase, option) for extraction
    choices = {}
    for option, phrases in keywords.items():
        for phrase in phrases:
            choices[phrase] = option

    result = rfprocess.extractOne(
        text,
        choices.keys(),
        scorer=fuzz.partial_ratio,
    )
    if result is None:
        return None, 0.0

    matched_phrase, score, _ = result
    if score >= threshold:
        return choices[matched_phrase], score

    return None, 0.0
