"""Fuzzy matcher: maps patient speech transcripts to FSM response options.

Each response_type has a fixed set of valid options with extensive keyword aliases
covering misspellings, accent variations, common mishearings, and multilingual input.
Two-pass matching: exact keyword hit first, then fuzzy fallback via rapidfuzz.
"""

from rapidfuzz import fuzz, process as rfprocess

# ═══════════════════════════════════════════════════════════════════════════
# KEYWORD MAP — Comprehensive aliases per response option
# Covers: correct spellings, common STT mishearings, accent variations,
# romanized Hindi, Devanagari Hindi, and regional language keywords.
# ═══════════════════════════════════════════════════════════════════════════

KEYWORD_MAP = {
    # ── 1. COARSE SPHERE: Clarity (Phases B, D) ──────────────────────────
    "READABILITY": {
        "READABLE": [
            # Core
            "clear", "readable", "can read", "i can see", "yes", "fine",
            "sharp", "good", "visible", "okay", "ok",
            # Misspellings / STT variations
            "cleer", "clr", "clar", "cler", "clir", "klear", "kleer",
            "clearrr", "clearr", "clea", "claa", "cleyar", "cliyar",
            "cliyaar", "cleear", "clerr", "clean", "crisp",
            # Phrases
            "perfectly clear", "very clear", "fully clear", "completely clear",
            "totally clear", "quite clear", "looks clear", "seems clear",
            "now clear", "it's clear", "it is clear", "yes clear", "yeah clear",
            "all clear", "everything clear",
            "i can see clearly", "i see clearly", "i can read clearly",
            "i can read it", "i can see it",
            "fully readable", "easily readable", "fully visible",
            "clearly visible", "properly visible",
            "no blur", "no blurriness", "no issues", "no problem",
            "very good", "perfect", "looks good", "much better", "better now",
            "okay now", "nice", "excellent", "absolutely clear", "crystal clear",
            "super clear", "fully sharp", "fully crisp", "totally readable",
            "i can read everything", "all letters visible",
            "better", "improved", "clearer",
            # Romanized Hindi
            "saaf", "dikh raha", "padh paa raha", "haan", "theek",
            "bilkul saaf", "accha dikh raha", "sab dikh raha",
            # Devanagari Hindi
            "साफ़", "साफ", "दिख रहा", "पढ़ पा रहा", "हाँ", "हां", "ठीक",
            "पढ़ सकता हूँ", "दिखाई दे रहा", "अच्छा", "बिल्कुल साफ़",
        ],
        "NOT_READABLE": [
            # Core
            "cannot see", "can't see", "cant see", "can not see",
            "cannot see anything", "can't see anything",
            "not visible", "nothing visible", "completely invisible",
            "blank", "fully blurred out",
            "cannot read", "can't read", "cant read",
            "cannot recognize", "cannot identify",
            "nothing readable", "no letters visible",
            "cannot make out anything", "fully unclear", "total blur",
            "no visibility", "black", "blank screen",
            "no idea", "no idea what this is", "nothing at all",
            "cannot see letters", "i see nothing", "i can't make anything",
            "totally gone", "fully gone", "completely gone",
            "not readable", "nothing", "no",
            "worse", "much worse", "worse than before", "can't see anything",
            # Romanized Hindi
            "nahi dikh raha", "padh nahi paa raha", "nahi", "kuch nahi",
            "bilkul nahi", "kuch bhi nahi",
            # Devanagari Hindi
            "नहीं दिख रहा", "पढ़ नहीं पा रहा", "नहीं", "कुछ नहीं",
            "दिखाई नहीं दे रहा", "बिल्कुल नहीं",
        ],
        "BLURRY": [
            # Core
            "blurry", "blurred", "blur", "slightly blurry", "a bit blurry",
            "hazy", "fuzzy", "not clear", "not sharp", "foggy",
            # Misspellings / STT variations
            "blurr", "blury", "blr", "blu", "blar", "bler", "blor",
            "blair", "blare", "bloor", "bloar", "bloory", "bloi",
            "bloody", "bluddy", "bladi", "blody", "bladiy",
            "bluri", "bluryy", "blurrr", "blurrry", "blr hai",
            # Phrases
            "not clear", "unclear", "little blurry", "somewhat blurry",
            "mildly blurry", "very blurry", "extremely blurry",
            "looks blurry", "seems blurry", "still blurry",
            "not crisp", "not readable clearly",
            "hard to read", "difficult to read", "cannot read clearly",
            "partially visible", "out of focus", "low clarity",
            "not proper", "not good", "still not good",
            "smudged", "dull", "faded", "not sharp enough",
            "little unclear", "bit unclear",
            "a little worse", "slightly worse", "bit blurry",
            # Romanized Hindi
            "dhundhla", "thoda dhundhla", "halka", "thoda blur",
            # Devanagari Hindi
            "धुंधला", "थोड़ा धुंधला", "हल्का", "धुंधला है",
        ],
    },

    # ── 2. JCC / COMPARE 1 vs 2 (Phases E, F, H, I) ─────────────────────
    "COMPARE_1_2": {
        "BETTER_1": [
            # Core
            "one", "first", "1", "number one", "the first one",
            "option one", "first one",
            # Misspellings / STT
            "won", "wan", "wahn", "on",
            # Phrases
            "one is better", "first is better", "first looks better",
            "one looks better", "one is clearer", "first is clearer",
            "first is sharp", "first is more clear", "one is more clear",
            "first is readable", "one is readable",
            "i prefer one", "i choose one", "going with one",
            "definitely one", "clearly one", "one is good",
            "one better", "one clearer", "first better", "first clearer",
            "one is sharper", "first sharper", "looks better first",
            "one was clearer", "one was sharper",
            # Romanized Hindi
            "pehla", "ek", "pehla wala", "pehla accha",
            # Devanagari Hindi
            "पहला", "एक", "पहला वाला", "पहला बेहतर", "पहला अच्छा",
            # Telugu
            "మొదటిది", "ఒకటి",
            # Tamil
            "முதல்", "ஒன்று",
            # Kannada
            "ಮೊದಲನೆಯದು", "ಒಂದು",
            # Marathi
            "पहिला", "एक",
        ],
        "BETTER_2": [
            # Core
            "two", "second", "2", "number two", "the second one",
            "option two", "second one",
            # Misspellings / STT
            "to", "too", "tu", "do",
            # Phrases
            "two is better", "second is better", "second looks better",
            "two looks better", "two is clearer", "second is clearer",
            "second is sharp", "two is more clear", "second is more clear",
            "second is readable",
            "i prefer two", "i choose two", "going with two",
            "definitely two", "clearly two", "two is good",
            "two better", "second better", "second clearer",
            "looks better second",
            "two was clearer", "two was sharper",
            # Romanized Hindi
            "doosra", "do", "doosra wala", "doosra accha",
            # Devanagari Hindi
            "दूसरा", "दो", "दूसरा वाला", "दूसरा बेहतर",
            # Telugu
            "రెండవది", "రెండు",
            # Tamil
            "இரண்டாவது", "இரண்டு",
            # Kannada
            "ಎರಡನೆಯದು", "ಎರಡು",
            # Marathi
            "दुसरा", "दोन",
        ],
        "SAME": [
            "same", "same same", "both same", "both are same",
            "equal", "equally clear", "both clear", "both look same",
            "no difference", "cannot see difference", "no change",
            "identical", "similar", "equally good", "equally bad",
            "both fine", "both okay", "same only", "exactly same",
            "looks identical", "looks equal", "same clarity",
            "no change at all", "about the same",
            # Romanized Hindi
            "dono same", "barabar", "ek jaisa",
            # Devanagari Hindi
            "दोनों एक जैसे", "बराबर", "एक जैसा", "समान", "दोनों",
            # Telugu
            "ఒకేలా", "సమానం",
            # Tamil
            "ஒரே மாதிரி", "சமம்",
            # Kannada
            "ಒಂದೇ", "ಸಮಾನ",
            # Marathi
            "सारखं", "समान",
        ],
        "CANT_TELL": [
            "can't tell", "cannot tell", "cant tell",
            "not sure", "unsure", "don't know", "i don't know",
            "difficult to say", "hard to say", "cannot decide",
            "unable to decide", "confusing", "unclear difference",
            "i cannot choose", "no idea", "not able to compare",
            "too close to tell", "both confusing", "not able to judge",
            "very similar", "hard to differentiate",
            "hard to tell", "repeat", "show again",
            # Romanized Hindi
            "pata nahi", "samajh nahi aaya",
            # Devanagari Hindi
            "पता नहीं", "समझ नहीं आया", "फिर से दिखाओ",
            # Telugu
            "తెలియదు", "అర్థం కాలేదు",
            # Tamil
            "தெரியாது", "புரியவில்லை",
            # Kannada
            "ಗೊತ್ತಿಲ್ಲ", "ಅರ್ಥವಾಗಲಿಲ್ಲ",
            # Marathi
            "कळत नाही", "समजत नाही",
        ],
    },

    # ── 3. DUOCHROME: Red vs Green (Phases G, J) ─────────────────────────
    "COLOR_CHOICE": {
        "RED_CLEARER": [
            "red", "red clearer", "red is clearer", "red is better",
            "red side better", "red side clear", "red looks clearer",
            "red looks better", "red is sharp", "red more clear",
            "red more visible", "red easier to read", "red sharper",
            "red side sharper", "red better clarity", "red is more readable",
            "red one", "left side", "red better",
            # Romanized Hindi
            "lal", "lal wala", "lal taraf",
            # Devanagari Hindi
            "लाल", "लाल वाला", "लाल तरफ़",
            # Telugu
            "ఎరుపు",
            # Tamil
            "சிவப்பு",
            # Kannada
            "ಕೆಂಪು",
            # Marathi
            "लाल",
        ],
        "GREEN_CLEARER": [
            "green", "green clearer", "green is clearer", "green is better",
            "green side better", "green side clear", "green looks clearer",
            "green looks better", "green is sharp", "green more clear",
            "green more visible", "green easier to read", "green sharper",
            "green side sharper", "green better clarity", "green is more readable",
            "green one", "right side", "green better",
            # Romanized Hindi
            "hara", "hara wala", "hari taraf",
            # Devanagari Hindi
            "हरा", "हरा वाला", "हरी तरफ़",
            # Telugu
            "ఆకుపచ్చ",
            # Tamil
            "பச்சை",
            # Kannada
            "ಹಸಿರು",
            # Marathi
            "हिरवा",
        ],
        "EQUAL": [
            "same", "both same", "both equal", "equally clear",
            "both clear", "no difference", "both look same",
            "equal clarity", "identical", "no change",
            "same on both", "both sides same", "both look identical",
            "about the same",
            "barabar", "dono same",
            "बराबर", "दोनों एक जैसे", "एक जैसा",
            "ఒకేలా", "సమానం",
            "ஒரே மாதிரி", "சமம்",
            "ಒಂದೇ", "ಸಮಾನ",
            "सारखं", "समान",
        ],
        "CANT_TELL": [
            "can't tell", "cannot tell", "cant tell",
            "not sure", "unsure", "don't know",
            "difficult to say", "cannot decide", "unable to compare",
            "no idea", "both confusing", "both look similar",
            "cannot differentiate", "hard to tell", "repeat",
            "pata nahi",
            "पता नहीं", "समझ नहीं आया",
            "తెలియదు",
            "தெரியாது",
            "ಗೊತ್ತಿಲ್ಲ",
            "कळत नाही",
        ],
    },

    # ── 4. BINOCULAR BALANCE: Top vs Bottom (Phase K) ────────────────────
    "TOP_BOTTOM": {
        "TOP_CLEARER": [
            "top", "upper", "top row", "top one", "first row",
            "top is clearer", "top looks clearer", "upper row",
            "upar", "upar wala",
            "ऊपर", "ऊपर वाली", "ऊपर की",
            "పైది", "పైన",
            "மேல்",
            "ಮೇಲೆ", "ಮೇಲಿನ",
            "वरची",
        ],
        "BOTTOM_CLEARER": [
            "bottom", "lower", "bottom row", "bottom one", "second row",
            "bottom is clearer", "bottom looks clearer", "lower row",
            "neeche", "neeche wala",
            "नीचे", "नीचे वाली", "नीचे की",
            "కిందది", "కింద",
            "கீழ்",
            "ಕೆಳಗಡೆ", "ಕೆಳಗಿನ",
            "खालची",
        ],
        "SAME": [
            "same", "equal", "about the same", "both same",
            "no difference", "both look same", "identical",
            "barabar",
            "बराबर", "दोनों एक जैसी", "एक जैसा",
            "ఒకేలా",
            "ஒரே மாதிரி",
            "ಒಂದೇ",
            "सारखं",
        ],
        "CANT_TELL": [
            "can't tell", "cant tell", "not sure", "don't know",
            "unsure", "hard to tell", "repeat",
            "pata nahi",
            "पता नहीं", "समझ नहीं आया",
            "తెలియదు",
            "தெரியாது",
            "ಗೊತ್ತಿಲ್ಲ",
            "कळत नाही",
        ],
    },

    # ── 5. NEAR VISION: Readability (Phases P, Q) ────────────────────────
    "NEAR_READABILITY": {
        "READABLE": [
            # Core
            "clear", "readable", "can read", "i can see", "yes", "fine",
            "sharp", "good", "visible", "okay", "ok",
            # Misspellings
            "cleer", "clr", "clar",
            # Phrases
            "clearly readable", "easy to read", "fully readable",
            "comfortably readable", "can read easily",
            "i can read clearly", "text is clear", "looks clear",
            "perfectly readable", "sharp text", "very clear",
            "fully visible", "no issue reading", "can read all",
            "reading is easy", "everything readable",
            "letters clear", "text sharp",
            "better", "improved", "clearer",
            # Hindi
            "saaf", "padh paa raha", "haan", "theek",
            "साफ़", "साफ", "पढ़ पा रहा", "हाँ", "ठीक",
            # Telugu
            "స్పష్టం", "చదవగలను", "అవును",
            # Tamil
            "தெளிவு", "படிக்க முடிகிறது", "ஆம்",
            # Kannada
            "ಸ್ಪಷ್ಟ", "ಹೌದು",
            # Marathi
            "स्पष्ट", "होय",
        ],
        "NOT_READABLE": [
            "not readable", "can't read", "cannot read", "nothing",
            "can't see", "cannot see", "no", "blank", "not visible",
            "worse", "much worse",
            "cannot read", "cant read",
            # Hindi
            "nahi dikh raha", "padh nahi paa raha", "nahi",
            "नहीं दिख रहा", "पढ़ नहीं पा रहा", "नहीं",
            # Telugu
            "కనిపించడం లేదు", "చదవలేను", "లేదు",
            # Tamil
            "தெரியவில்லை", "படிக்க முடியவில்லை", "இல்லை",
            # Kannada
            "ಕಾಣುತ್ತಿಲ್ಲ", "ಇಲ್ಲ",
            # Marathi
            "दिसत नाही", "नाही",
        ],
        "BLURRY": [
            "blurry", "blurred", "blur", "slightly blurry", "a bit blurry",
            "hazy", "fuzzy", "not clear", "not sharp", "foggy",
            "bloody", "blury", "blurr",
            "hard to read", "difficult to read", "not readable properly",
            "unclear text", "fuzzy text", "out of focus", "low clarity",
            "struggling to read", "not easy to read",
            "little blur", "bit unclear", "text not sharp",
            "not comfortable", "uncomfortable", "strain", "eye strain",
            "too much effort", "difficult", "tiring",
            "a little worse", "slightly worse",
            # Hindi
            "dhundhla", "thoda dhundhla",
            "धुंधला", "थोड़ा धुंधला",
            # Telugu
            "మసకం",
            # Tamil
            "மங்கல்",
            # Kannada
            "ಮಸುಕು",
            # Marathi
            "धूसर",
        ],
    },

    # ── 6. NEAR BINOCULAR: Final confirmation (Phase R) ──────────────────
    "NEAR_BINOC": {
        "TARGET_OK": [
            # Core
            "yes", "yeah", "yep", "yup", "correct", "perfect",
            "good", "very good", "fine", "looks good", "all good",
            "everything good", "comfortable", "clear",
            "clear and comfortable", "looks perfect", "no issues",
            "satisfied", "happy with this", "this is good",
            "this works", "this is fine", "this is correct",
            "excellent", "great", "ideal", "perfect now",
            "no problem", "feels right",
            "ok", "okay", "can read", "readable",
            # Hindi
            "theek hai", "haan", "accha",
            "ठीक है", "हाँ", "आरामदायक", "साफ़", "पढ़ पा रहा",
            # Telugu
            "స్పష్టం", "బాగుంది", "అవును",
            # Tamil
            "தெளிவு", "ஆம்",
            # Kannada
            "ಸ್ಪಷ್ಟ", "ಹೌದು",
            # Marathi
            "स्पष्ट", "होय",
        ],
        "NOT_CLEAR": [
            "not good", "not perfect", "slightly off", "little blur",
            "still blurry", "not comfortable", "needs adjustment",
            "can be better", "improve this", "not satisfied",
            "not ideal", "not clear enough", "something feels off",
            "still not right", "not fully correct", "slight issue",
            "not comfortable yet", "little problem",
            "not clear", "blurry", "blur", "no", "can't read",
            "not readable", "not comfortable",
            # Hindi
            "nahi dikh raha", "theek nahi",
            "साफ़ नहीं", "धुंधला", "नहीं", "पढ़ नहीं पा रहा",
            # Telugu
            "స్పష్టం కాదు", "లేదు",
            # Tamil
            "தெளிவில்லை", "இல்லை",
            # Kannada
            "ಸ್ಪಷ್ಟವಲ್ಲ", "ಇಲ್ಲ",
            # Marathi
            "स्पष्ट नाही", "नाही",
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
                hits.append((option, len(phrase)))

    if hits:
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[0][0], 100.0

    # Pass 2: fuzzy match using rapidfuzz
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
