from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Optional


LEARNED_AUDIO_VOCAB_PATH = Path("results") / "interactive" / "learned_audio_vocab.json"
_LEARNED_AUDIO_VOCAB_CACHE: dict[str, Any] = {
    "path": None,
    "mtime": None,
    "entries_by_family": {},
}


def _normalize_variant_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _merge_variant_groups(*groups: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    merged: set[str] = set()
    for group in groups:
        for item in group:
            normalized = _normalize_variant_text(item)
            if normalized:
                merged.add(normalized)
    return sorted(merged)


def _expand_with_suffixes(roots: list[str], suffixes: list[str]) -> list[str]:
    variants: set[str] = set()
    for root in roots:
        normalized_root = _normalize_variant_text(root)
        if not normalized_root:
            continue
        variants.add(normalized_root)
        for suffix in suffixes:
            normalized_suffix = _normalize_variant_text(suffix)
            if normalized_suffix:
                variants.add(f"{normalized_root} {normalized_suffix}")
    return sorted(variants)


def _expand_with_templates(roots: list[str], templates: list[str]) -> list[str]:
    variants: set[str] = set()
    for root in roots:
        normalized_root = _normalize_variant_text(root)
        if not normalized_root:
            continue
        for template in templates:
            normalized_variant = _normalize_variant_text(template.format(root=normalized_root))
            if normalized_variant:
                variants.add(normalized_variant)
    return sorted(variants)


FSM_AUDIO_RESPONSE_LIBRARY_V1 = {
    "version": "fsm_audio_response_library_v1",
    "language": "en-IN",
    "matching_pipeline": {
        "steps": [
            "normalize_text",
            "exact_match",
            "phrase_match",
            "fuzzy_match",
            "semantic_match",
            "context_filter",
            "confidence_gate",
        ],
        "thresholds": {
            "exact_match_confidence": 1.0,
            "phrase_match_confidence": 0.95,
            "fuzzy_match_confidence": 0.85,
            "semantic_match_confidence": 0.78,
            "fallback_reprompt_below": 0.78,
        },
        "normalization": {
            "lowercase": True,
            "strip_punctuation": True,
            "collapse_spaces": True,
            "remove_fillers": ["uh", "um", "hmm", "huh", "ah", "er", "mm"],
            "common_asr_corrections": {
                "bloody": "blurry",
                "bluddy": "blurry",
                "bladi": "blurry",
                "blody": "blurry",
                "bloodyy": "blurry",
                "bloory": "blurry",
                "blooi": "blurry",
                "blurri": "blurry",
                "blurryy": "blurry",
                "cleer": "clear",
                "clar": "clear",
                "cler": "clear",
                "clir": "clear",
                "klear": "clear",
                "kleer": "clear",
                "clearr": "clear",
                "clearrr": "clear",
                "redable": "readable",
                "reedable": "readable",
                "readible": "readable",
                "reedble": "readable",
                "vizible": "visible",
                "visibl": "visible",
                "vizble": "visible",
                "won": "one",
                "wan": "one",
                "wahn": "one",
                "1": "one",
                "2": "two",
                "cleared": "clear",
                "cant": "can't",
                "dont": "don't",
                "safi": "saaf",
                "safi hai": "saaf hai",
                "saif": "saaf",
                "saif hai": "saaf hai",
                "saif nahi hai": "dhundla hai",
                "safi nahi hai": "dhundla hai",
                "saaf nahi hai": "dhundla hai",
                "saf nahi hai": "dhundla hai",
                "saath hai": "saaf hai",
                "saath nahi hai": "dhundla hai",
                "saad hai": "saaf hai",
                "safi hain": "saaf hai",
                "do no saman": "dono saman",
                "do no same": "dono same",
                "do no barabar": "dono barabar",
                "donosaman": "dono saman",
                "दोनोंसमान": "दोनों समान",
                "दोनोंसमान है": "दोनों समान",
                "दोनों समान है": "दोनों समान",
                "दोनोंसैम": "दोनों समान",
                "jule hai": "dhundla hai",
                "jule hain": "dhundla hai",
                "dundle hai": "dhundla hai",
                "dundle hai": "dhundla hai",
                "dunai hai": "dhundla hai",
                "dundla hai": "dhundla hai",
                "dundhla hai": "dhundla hai",
            },
        },
    },
    "states": {
        "B_COARSE_SPHERE": {
            "response_type": "clarity_3way",
            "allowed_labels": ["CLEAR", "BLURRY", "NOT_VISIBLE"],
            "labels": {
                "CLEAR": {
                    "exact": ["clear", "readable", "visible", "sharp", "crisp", "good", "perfect", "fine", "okay"],
                    "phrases": [
                        "it is clear",
                        "it's clear",
                        "looks clear",
                        "seems clear",
                        "very clear",
                        "fully clear",
                        "completely clear",
                        "totally clear",
                        "perfectly clear",
                        "all clear",
                        "everything clear",
                        "i can see clearly",
                        "i see clearly",
                        "i can read clearly",
                        "i can read it",
                        "i can see it",
                        "fully readable",
                        "easily readable",
                        "clearly visible",
                        "properly visible",
                        "looks good",
                        "very good",
                        "perfect",
                        "fine",
                        "much better",
                        "better now",
                        "okay now",
                        "crystal clear",
                        "super clear",
                        "all letters visible",
                        "i can read everything",
                        "no blur",
                        "no blurriness",
                        "i can read",
                        "i can read this",
                        "everything is clear",
                        "everything is readable",
                        "now it is clear",
                        "yes clear",
                        "yeah clear",
                        "all good",
                        "totally readable",
                        "fully sharp",
                        "fully crisp",
                    ],
                    "fuzzy": [
                        "clr",
                        "clearrr",
                        "clearr",
                        "clea",
                        "claa",
                        "cleyar",
                        "cliyar",
                        "cleear",
                        "clerr",
                    ],
                    "context_only": ["yes", "yeah", "yep", "okay", "ok", "good", "fine", "perfect"],
                },
                "BLURRY": {
                    "exact": ["blur", "blurry", "unclear", "hazy", "fuzzy", "not clear"],
                    "phrases": [
                        "it is blurry",
                        "it's blurry",
                        "looks blurry",
                        "seems blurry",
                        "still blurry",
                        "slightly blurry",
                        "little blurry",
                        "bit blurry",
                        "somewhat blurry",
                        "mildly blurry",
                        "very blurry",
                        "extremely blurry",
                        "not clear",
                        "not sharp",
                        "not crisp",
                        "hard to read",
                        "difficult to read",
                        "cannot read clearly",
                        "partially visible",
                        "out of focus",
                        "low clarity",
                        "not proper",
                        "not good",
                        "worse",
                        "still not good",
                        "smudged",
                        "dull",
                        "faded",
                        "not sharp enough",
                        "little unclear",
                        "bit unclear",
                        "letters are blurry",
                        "text is blurry",
                        "not readable clearly",
                        "slightly blur",
                        "still not clear",
                        "bit blur",
                        "little blur",
                        "a bit blurry",
                        "some blur",
                        "worse now",
                    ],
                    "fuzzy": [
                        "blurr",
                        "blury",
                        "blr",
                        "blu",
                        "blar",
                        "bler",
                        "blor",
                        "bloar",
                        "bloory",
                        "blooy",
                        "bluri",
                        "bluryy",
                        "blurrr",
                        "blurrry",
                        "bloody",
                        "bluddy",
                        "bladi",
                        "blody",
                    ],
                },
                "NOT_VISIBLE": {
                    "exact": ["cannot see", "can't see", "not visible", "cannot read", "can't read", "not readable"],
                    "phrases": [
                        "cannot see anything",
                        "can't see anything",
                        "nothing visible",
                        "completely invisible",
                        "blank",
                        "fully blurred out",
                        "cannot recognize",
                        "cannot identify",
                        "nothing readable",
                        "no letters visible",
                        "cannot make out anything",
                        "fully unclear",
                        "total blur",
                        "no visibility",
                        "black",
                        "blank screen",
                        "i see nothing",
                        "i cannot make anything out",
                        "totally gone",
                        "completely gone",
                        "cannot see letters",
                        "cannot read this at all",
                        "i cannot see",
                        "i can't see it",
                        "i can't read it",
                        "cannot make out letters",
                        "nothing is visible",
                    ],
                    "fuzzy": ["cant see", "cant read", "can not see", "can not read"],
                },
            },
        },
        "E_F_JCC_COMPARISON": {
            "response_type": "comparison_4way",
            "allowed_labels": ["BETTER_1", "BETTER_2", "SAME", "CANT_TELL"],
            "labels": {
                "BETTER_1": {
                    "exact": ["1", "one", "first"],
                    "phrases": [
                        "option one",
                        "first one",
                        "number one",
                        "one is better",
                        "first is better",
                        "first looks better",
                        "one looks better",
                        "one is clearer",
                        "first is clearer",
                        "first is sharp",
                        "one is more clear",
                        "first is more clear",
                        "first is readable",
                        "one is readable",
                        "i prefer one",
                        "i choose one",
                        "going with one",
                        "definitely one",
                        "clearly one",
                        "one better",
                        "one clearer",
                        "first better",
                        "first clearer",
                        "one sharper",
                        "first sharper",
                        "looks better first",
                        "one is good",
                        "one is more readable",
                        "first one better",
                    ],
                    "fuzzy": ["won", "wan", "wahn", "on"],
                },
                "BETTER_2": {
                    "exact": ["2", "two", "second"],
                    "phrases": [
                        "option two",
                        "second one",
                        "number two",
                        "two is better",
                        "second is better",
                        "second looks better",
                        "two looks better",
                        "two is clearer",
                        "second is clearer",
                        "second is sharp",
                        "two is more clear",
                        "second is more clear",
                        "second is readable",
                        "i prefer two",
                        "i choose two",
                        "going with two",
                        "definitely two",
                        "clearly two",
                        "two better",
                        "second better",
                        "second clearer",
                        "looks better second",
                        "two is good",
                        "two is more readable",
                        "second one better",
                    ],
                    "fuzzy": ["to", "too", "tu", "do"],
                },
                "SAME": {
                    "exact": ["same", "equal", "similar"],
                    "phrases": [
                        "both same",
                        "both are same",
                        "equally clear",
                        "both clear",
                        "both look same",
                        "no difference",
                        "cannot see difference",
                        "no change",
                        "identical",
                        "similar",
                        "equally good",
                        "equally bad",
                        "both fine",
                        "both okay",
                        "same only",
                        "exactly same",
                        "looks identical",
                        "looks equal",
                        "same clarity",
                        "no difference at all",
                        "about the same",
                        "they look same",
                        "they are the same",
                        "still the same",
                    ],
                },
                "CANT_TELL": {
                    "exact": ["unsure", "uncertain", "confusing"],
                    "phrases": [
                        "can't tell",
                        "cannot tell",
                        "cant tell",
                        "not sure",
                        "dont know",
                        "don't know",
                        "difficult to say",
                        "hard to say",
                        "cannot decide",
                        "unable to decide",
                        "confusing",
                        "unclear difference",
                        "i cannot choose",
                        "no idea",
                        "not able to compare",
                        "too close to tell",
                        "both confusing",
                        "not able to judge",
                        "very similar",
                        "hard to differentiate",
                        "hard to tell",
                        "too close",
                    ],
                    "fuzzy": ["too close", "cannot chose"],
                },
            },
        },
        "G_J_DUOCHROME": {
            "response_type": "red_green_4way",
            "allowed_labels": ["RED_CLEARER", "GREEN_CLEARER", "SAME", "CANT_TELL"],
            "labels": {
                "RED_CLEARER": {
                    "exact": ["red"],
                    "phrases": [
                        "red clearer",
                        "red is clearer",
                        "red is better",
                        "red side better",
                        "red side clear",
                        "red looks clearer",
                        "red looks better",
                        "red is sharp",
                        "red more clear",
                        "red more visible",
                        "red easier to read",
                        "red sharper",
                        "red side sharper",
                        "red better clarity",
                        "red is more readable",
                        "red side",
                    ],
                },
                "GREEN_CLEARER": {
                    "exact": ["green"],
                    "phrases": [
                        "green clearer",
                        "green is clearer",
                        "green is better",
                        "green side better",
                        "green side clear",
                        "green looks clearer",
                        "green looks better",
                        "green is sharp",
                        "green more clear",
                        "green more visible",
                        "green easier to read",
                        "green sharper",
                        "green side sharper",
                        "green better clarity",
                        "green is more readable",
                        "green side",
                    ],
                },
                "SAME": {
                    "exact": ["same", "equal", "similar"],
                    "phrases": [
                        "both same",
                        "both equal",
                        "equally clear",
                        "both clear",
                        "no difference",
                        "both look same",
                        "equal clarity",
                        "identical",
                        "no change",
                        "same on both",
                        "both sides same",
                        "both look identical",
                    ],
                },
                "CANT_TELL": {
                    "exact": ["unsure", "uncertain"],
                    "phrases": [
                        "can't tell",
                        "cannot tell",
                        "cant tell",
                        "not sure",
                        "unsure",
                        "don't know",
                        "difficult to say",
                        "cannot decide",
                        "unable to compare",
                        "no idea",
                        "both confusing",
                        "both look similar",
                        "cannot differentiate",
                        "hard to tell",
                    ],
                },
            },
        },
        "K_BINOCULAR_BALANCE": {
            "response_type": "binocular_balance_4way",
            "allowed_labels": ["TOP_CLEARER", "BOTTOM_CLEARER", "SAME", "CANT_TELL"],
            "labels": {
                "TOP_CLEARER": {
                    "exact": ["top", "upper"],
                    "phrases": [
                        "top is clearer",
                        "top clearer",
                        "top is better",
                        "upper is better",
                        "top looks better",
                        "top one is better",
                        "top side",
                    ],
                },
                "BOTTOM_CLEARER": {
                    "exact": ["bottom", "lower"],
                    "phrases": [
                        "bottom is clearer",
                        "bottom clearer",
                        "bottom is better",
                        "lower is better",
                        "bottom looks better",
                        "bottom one is better",
                        "bottom side",
                    ],
                },
                "SAME": {
                    "exact": ["same", "equal", "similar"],
                    "phrases": [
                        "both same",
                        "both equal",
                        "no difference",
                        "looks same",
                        "equally clear",
                    ],
                },
                "CANT_TELL": {
                    "exact": ["unsure", "uncertain"],
                    "phrases": [
                        "can't tell",
                        "cannot tell",
                        "not sure",
                        "unsure",
                        "hard to say",
                        "no idea",
                    ],
                },
            },
        },
        "P_Q_NEAR_CLARITY": {
            "response_type": "near_clarity_3way",
            "allowed_labels": ["CLEAR", "BLURRY", "NOT_VISIBLE"],
            "labels": {
                "CLEAR": {
                    "exact": ["clear", "readable", "visible", "sharp"],
                    "phrases": [
                        "clearly readable",
                        "easy to read",
                        "fully readable",
                        "comfortably readable",
                        "can read easily",
                        "i can read clearly",
                        "text is clear",
                        "looks clear",
                        "perfectly readable",
                        "sharp text",
                        "very clear",
                        "fully visible",
                        "no issue reading",
                        "can read all",
                        "reading is easy",
                        "everything readable",
                        "letters clear",
                        "text sharp",
                        "i can read it",
                        "i can read this",
                        "looks readable",
                        "comfortably readable",
                    ],
                    "fuzzy": ["cleer", "clr", "clar", "cler", "clir", "klear", "kleer"],
                },
                "BLURRY": {
                    "exact": ["blur", "blurry", "unclear", "fuzzy", "hazy", "not clear"],
                    "phrases": [
                        "not clear",
                        "not that clear",
                        "not very clear",
                        "slightly blurry",
                        "hard to read",
                        "difficult to read",
                        "not readable properly",
                        "unclear text",
                        "fuzzy text",
                        "out of focus",
                        "low clarity",
                        "not sharp",
                        "struggling to read",
                        "not easy to read",
                        "little blur",
                        "bit unclear",
                        "text not sharp",
                        "text is blurry",
                        "reading is hard",
                        "struggling to read it",
                    ],
                    "fuzzy": ["bloody", "bluddy", "bladi", "blody", "blurr", "blury", "blr"],
                },
                "NOT_VISIBLE": {
                    "exact": ["cannot read", "can't read", "not readable"],
                    "phrases": [
                        "cannot read this",
                        "can't read this",
                        "nothing readable",
                        "cannot make out text",
                        "not visible",
                        "cannot see letters",
                        "too blurred to read",
                        "cannot identify text",
                        "i cannot read at all",
                        "can't read at all",
                        "not able to read",
                    ],
                },
            },
        },
        "R_NEAR_BINOCULAR": {
            "response_type": "near_final_3way",
            "allowed_labels": ["TARGET_OK", "NEED_ADJUSTMENT", "NOT_COMFORTABLE"],
            "labels": {
                "TARGET_OK": {
                    "exact": ["yes", "good", "fine", "perfect", "comfortable", "clear", "correct", "satisfied"],
                    "phrases": [
                        "clear and comfortable",
                        "it is comfortable",
                        "it's comfortable",
                        "looks perfect",
                        "no issues",
                        "satisfied",
                        "happy with this",
                        "this is good",
                        "this works",
                        "this is fine",
                        "this is correct",
                        "all good",
                        "everything good",
                        "excellent",
                        "great",
                        "ideal",
                        "perfect now",
                        "no problem",
                        "feels right",
                        "all okay",
                        "looks fine",
                    ],
                    "context_only": ["yes", "yeah", "yep", "okay", "ok", "good", "fine", "perfect"],
                },
                "NEED_ADJUSTMENT": {
                    "exact": ["adjust", "better"],
                    "phrases": [
                        "not good",
                        "not perfect",
                        "slightly off",
                        "little blur",
                        "still blurry",
                        "needs adjustment",
                        "can be better",
                        "improve this",
                        "not satisfied",
                        "not ideal",
                        "not clear enough",
                        "something feels off",
                        "still not right",
                        "not fully correct",
                        "slight issue",
                        "little problem",
                        "needs a little adjustment",
                        "can improve",
                    ],
                },
                "NOT_COMFORTABLE": {
                    "exact": ["uncomfortable", "strain"],
                    "phrases": [
                        "not comfortable",
                        "eye strain",
                        "too much effort",
                        "difficult",
                        "tiring",
                        "stressful",
                        "hard to focus",
                        "not easy",
                        "takes effort",
                        "not relaxing",
                        "feels heavy",
                        "not smooth",
                        "not natural",
                        "stressful to read",
                        "eyes hurting",
                        "pressure on eyes",
                        "uncomfortable to read",
                        "not easy reading",
                    ],
                },
            },
        },
    },
    "context_rules": {
        "yes_like_terms": {
            "allowed_in": {
                "clarity_3way": "CLEAR",
                "near_final_3way": "TARGET_OK",
            },
            "disallowed_in": ["comparison_4way", "red_green_4way", "binocular_balance_4way"],
        },
        "good_like_terms": {
            "allowed_in": {
                "clarity_3way": "CLEAR",
                "near_final_3way": "TARGET_OK",
            },
            "disallowed_in": ["comparison_4way", "red_green_4way", "binocular_balance_4way"],
        },
        "better_like_terms": {
            "allowed_in": {
                "comparison_4way": "needs_disambiguation",
                "near_final_3way": "NEED_ADJUSTMENT",
            }
        },
    },
    "fallback_policy": {
        "on_low_confidence": "REPROMPT",
        "reprompt_texts": {
            "clarity_3way": "Please say only: clear, blurry, or not readable.",
            "comparison_4way": "Please say only: one, two, same, or can't tell.",
            "red_green_4way": "Please say only: red, green, same, or can't tell.",
            "binocular_balance_4way": "Please say only: top, bottom, same, or can't tell.",
            "near_clarity_3way": "Please say only: clear, blurry, or not readable.",
            "near_final_3way": "Please say only: okay, needs adjustment, or not comfortable.",
        },
    },
}


MULTILINGUAL_RESPONSE_ALIASES = {
    "clarity_3way": {
        "CLEAR": {
            "exact": [
                "हाँ", "haan", "haanji", "ठीक", "theek", "thik", "sahi", "सही",
            ],
            "phrases": [
                "साफ है", "साफ दिख रहा है", "दिख रहा है", "पढ़ पा रहा हूं", "पढ पा रहा हूं",
                "saaf dikh raha", "dikh raha", "padh pa raha", "i can read", "i can see",
            ],
            "fuzzy": [
                "haan", "han", "haanjii", "thek", "thik hai", "saf", "saaf",
                "dikhrha", "dikhra", "padh pa raha", "pad pa raha",
            ],
        },
        "BLURRY": {
            "exact": ["धुंधला", "dhundla", "ब्लर", "blur"],
            "phrases": [
                "ब्लर है", "धुंधला है", "साफ नहीं", "थोड़ा blur है", "थोड़ा धुंधला है",
            ],
            "fuzzy": [
                "dhundhla", "dhudla", "blur hai", "thoda blur",
            ],
        },
        "NOT_VISIBLE": {
            "exact": ["नहीं", "nahi"],
            "phrases": [
                "नहीं दिख रहा", "कुछ नहीं दिख रहा", "पढ़ नहीं पा रहा", "पढ नहीं पा रहा",
                "nahi dikh raha", "nahi padh pa raha", "nahi padh raha",
                "not visible", "not readable",
            ],
            "fuzzy": [
                "nahi dikh raha", "nhi dikh raha", "kuch nahi dikh",
            ],
        },
    },
    "comparison_4way": {
        "BETTER_1": {
            "exact": ["एक", "ek", "पहला", "pehla"],
            "phrases": ["पहला बेहतर", "एक बेहतर", "पहला साफ"],
            "fuzzy": ["ik", "yek", "pahla"],
        },
        "BETTER_2": {
            "exact": ["दो", "do", "दूसरा", "dusra"],
            "phrases": ["दूसरा बेहतर", "दो बेहतर", "दूसरा साफ"],
            "fuzzy": ["doo", "dusara"],
        },
        "SAME": {
            "exact": ["बराबर", "barabar", "बरावर", "समान", "same"],
            "phrases": ["दोनों same", "कोई फर्क नहीं", "एक जैसा"],
            "fuzzy": ["barabr", "barber"],
        },
        "CANT_TELL": {
            "exact": ["पता नहीं", "unsure"],
            "phrases": ["समझ नहीं आ रहा", "तय नहीं", "not able to say"],
            "fuzzy": ["pata nai", "samaj nahi"],
        },
    },
    "red_green_4way": {
        "RED_CLEARER": {
            "exact": ["लाल", "lal"],
            "phrases": ["लाल साफ है", "लाल side"],
            "fuzzy": ["laal"],
        },
        "GREEN_CLEARER": {
            "exact": ["हरा", "hara"],
            "phrases": ["हरा साफ है", "हरा side"],
            "fuzzy": ["haraa"],
        },
        "SAME": {
            "exact": ["बराबर", "barabar", "same"],
            "phrases": ["कोई फर्क नहीं", "दोनों same"],
            "fuzzy": ["barabr"],
        },
        "CANT_TELL": {
            "exact": ["पता नहीं", "unsure"],
            "phrases": ["समझ नहीं आ रहा", "तय नहीं"],
            "fuzzy": ["pata nai"],
        },
    },
    "binocular_balance_4way": {
        "TOP_CLEARER": {
            "exact": ["ऊपर", "upar"],
            "phrases": ["ऊपर साफ है", "ऊपर better"],
            "fuzzy": ["uparr"],
        },
        "BOTTOM_CLEARER": {
            "exact": ["नीचे", "neeche"],
            "phrases": ["नीचे साफ है", "नीचे better"],
            "fuzzy": ["niche", "nichay"],
        },
        "SAME": {
            "exact": ["बराबर", "barabar", "same"],
            "phrases": ["कोई फर्क नहीं", "दोनों same"],
            "fuzzy": ["barabr"],
        },
        "CANT_TELL": {
            "exact": ["पता नहीं", "unsure"],
            "phrases": ["समझ नहीं आ रहा"],
            "fuzzy": ["pata nai"],
        },
    },
    "near_clarity_3way": {
        "CLEAR": {
            "exact": ["हाँ", "haan", "ठीक", "theek"],
            "phrases": ["पढ़ पा रहा हूं", "साफ है", "clear hai"],
            "fuzzy": ["padh pa raha", "saf"],
        },
        "BLURRY": {
            "exact": ["धुंधला", "dhundla"],
            "phrases": ["पढ़ने में blur है", "थोड़ा blur है"],
            "fuzzy": ["dhundhla", "blur hai"],
        },
        "NOT_VISIBLE": {
            "exact": ["नहीं", "nahi"],
            "phrases": ["पढ़ नहीं पा रहा", "दिख नहीं रहा"],
            "fuzzy": ["nahi dikh"],
        },
    },
    "near_final_3way": {
        "TARGET_OK": {
            "exact": ["ठीक", "theek", "सही", "sahi"],
            "phrases": ["ठीक है", "सही है", "आराम है", "comfortable है"],
            "fuzzy": ["thek", "sahi hai"],
        },
        "NEED_ADJUSTMENT": {
            "exact": ["adjust", "बेहतर", "better"],
            "phrases": ["adjust करना है", "और better", "थोड़ा adjust"],
            "fuzzy": ["adjest", "better chahiye"],
        },
        "NOT_COMFORTABLE": {
            "exact": ["असुविधा", "strain"],
            "phrases": ["आराम नहीं है", "strain है", "comfortable नहीं"],
            "fuzzy": ["aaraam nahi", "comfort nahi"],
        },
    },
}


LOCALIZED_VOICE_PROMPTS = {
    "hi": {
        "clarity_3way": {
            "initial": "अक्षरों को देखकर बताइए। साफ दिख रहा है, थोड़ा धुंधला है, या पढ़ा नहीं जा रहा?",
            "retry": "कृपया केवल यह कहें: साफ, धुंधला, या पढ़ा नहीं जा रहा।",
        },
        "comparison_4way": {
            "initial": "एक और दो में कौन बेहतर है? एक, दो, दोनों समान, या बताना मुश्किल है?",
            "retry": "कृपया केवल यह कहें: एक, दो, समान, या बता नहीं सकता।",
        },
        "red_green_4way": {
            "initial": "लाल और हरे में कौन सा साफ दिख रहा है? लाल, हरा, समान, या बताना मुश्किल है?",
            "retry": "कृपया केवल यह कहें: लाल, हरा, समान, या बता नहीं सकता।",
        },
        "binocular_balance_4way": {
            "initial": "ऊपर और नीचे में कौन साफ दिख रहा है? ऊपर, नीचे, समान, या बताना मुश्किल है?",
            "retry": "कृपया केवल यह कहें: ऊपर, नीचे, समान, या बता नहीं सकता।",
        },
        "near_clarity_3way": {
            "initial": "पास के अक्षरों को देखकर बताइए। साफ है, थोड़ा धुंधला है, या पढ़ा नहीं जा रहा?",
            "retry": "कृपया केवल यह कहें: साफ, धुंधला, या पढ़ा नहीं जा रहा।",
        },
        "near_final_3way": {
            "initial": "दोनों आँखों से पास का टेक्स्ट साफ और आरामदायक है, या अभी भी ठीक नहीं है?",
            "retry": "कृपया केवल यह कहें: ठीक है, एडजस्ट चाहिए, या आरामदायक नहीं है।",
        },
    },
}


LOCALIZED_LANGUAGE_SELECTION_PROMPTS = {
    "initial": (
        "English or Hindi? "
        "English या Hindi?"
    ),
    "retry": (
        "Please say only: English or Hindi. "
        "कृपया केवल English या Hindi कहें."
    ),
}


LANGUAGE_SELECTION_ALIASES = {
    "en": {
        "exact": ["english", "angrezi", "अंग्रेजी", "इंग्लिश"],
        "phrases": [
            "in english",
            "english language",
            "english please",
            "i prefer english",
            "i am comfortable in english",
            "angrezi mein",
            "mujhe english",
            "english mein",
        ],
        "fuzzy": ["inglish", "englis", "englis", "angrejee", "angrezii"],
    },
    "hi": {
        "exact": ["hindi", "हिंदी", "हिन्दी", "hi"],
        "phrases": [
            "in hindi",
            "hindi language",
            "hindi please",
            "i prefer hindi",
            "i am comfortable in hindi",
            "hindi mein",
            "mujhe hindi",
        ],
        "fuzzy": ["hindhi", "hindee", "hndi", "hindii", "hindy", "indi", "indy", "hii", "hei", "hey", "hai"],
    },
}


LOCALIZED_OPTION_LABELS = {
    "hi": {
        "CLEAR": "साफ",
        "BLURRY": "धुंधला",
        "REPEAT": "फिर से",
        "ONE": "एक",
        "TWO": "दो",
        "SAME": "समान",
        "RED": "लाल",
        "GREEN": "हरा",
        "TOP": "ऊपर",
        "BOTTOM": "नीचे",
    },
}


HINDI_QUESTION_TRANSLATIONS = {
    "Can you read all the letters in this line, or are they blurry?": "क्या आप इस लाइन के सभी अक्षर पढ़ पा रहे हैं, या वे धुंधले हैं?",
    "Please read the line. If the letters are not clear, say blurry, or repeat.": "क्या अक्षर साफ हैं, धुंधले हैं, या फिर से?",
    "Read the line, say blurry, or repeat.": "साफ, धुंधला, या फिर से?",
    "Read the line, or say blurry.": "साफ या धुंधला?",
    "Read the line.": "क्या अक्षर साफ हैं या धुंधले?",
    "Did it get better now? Say yes or no.": "क्या अब यह बेहतर हुआ? हाँ या नहीं कहिए।",
    "Better now, yes or no?": "अब बेहतर है, हाँ या नहीं?",
    "Did it get better than before? Say yes or no.": "क्या यह पहले से बेहतर हुआ? हाँ या नहीं कहिए।",
    "Better than before now? Yes or no?": "अब यह पहले से बेहतर है? हाँ या नहीं?",
    "Can you read the line now, or is it still blurry?": "क्या अब यह साफ है, अभी भी धुंधला है, या फिर से?",
    "Read it now, or say still blurry.": "साफ, अभी भी धुंधला, या फिर से?",
    "Please compare the two choices. Which one is clearer or sharper? say first option, second option, both same, or repeat.": "कृपया दोनों विकल्पों की तुलना कीजिए। कौन सा ज़्यादा साफ या शार्प दिख रहा है? पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से कहिए।",
    "Please compare the two dot patterns. Which one is clearer or sharper? say first option, second option, both same, or repeat.": "कृपया दोनों डॉट पैटर्न की तुलना कीजिए। कौन सा ज़्यादा साफ या शार्प दिख रहा है? पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से कहिए।",
    "First, second, or both same?": "पहला, दूसरा, या दोनों समान?",
    "First option, second option, both same, or repeat?": "पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से?",
    "First option, second option, or both same?": "पहला विकल्प, दूसरा विकल्प, या दोनों समान?",
    "Which is better, or are both same?": "कौन बेहतर है, या दोनों समान हैं?",
    "Please compare the green and red sides. Letters on which side look sharper and darker? Say green side, red side, both same, or repeat.": "कृपया हरे और लाल साइड की तुलना कीजिए। अक्षर किस साइड पर ज़्यादा शार्प और गहरे दिख रहे हैं? हरा साइड, लाल साइड, दोनों समान, या फिर से कहिए।",
    "Green side, red side, both same, or repeat?": "हरा साइड, लाल साइड, दोनों समान, या फिर से?",
    "Please compare the letters on the bottom and the top line. Which line looks sharper? Say bottom line, top line, both same, or repeat.": "कृपया नीचे और ऊपर की लाइन के अक्षरों की तुलना कीजिए। कौन सी लाइन ज़्यादा शार्प दिख रही है? नीचे की लाइन, ऊपर की लाइन, दोनों समान, या फिर से कहिए।",
    "Bottom line, top line, both same, or repeat?": "नीचे की लाइन, ऊपर की लाइन, दोनों समान, या फिर से?",
    "Please look at the near text. Is it clear, blurry, or repeat?": "कृपया पास के टेक्स्ट को देखिए। क्या यह साफ है, धुंधला है, या \"फिर से\"?",
    "Clear, blurry, or repeat?": "साफ, धुंधला, या फिर से?",
    "Please read the last line. Is it clear, blurry, or should I repeat?": "कृपया आखिरी लाइन पढ़िए। क्या यह साफ है, धुंधला है, या क्या मैं फिर से कहूँ?",
    "Is the last line clear or blurry?": "क्या आखिरी लाइन साफ है या धुंधली?",
    "Please read the line.": "कृपया लाइन पढ़िए।",
    "Please read the line, or say blurry.": "कृपया लाइन पढ़िए, या धुंधला कहिए।",
    "This is the final confirmation before I finish your eye test. This is the first option. Please observe the line carefully.": "आई टेस्ट समाप्त करने से पहले यह अंतिम पुष्टि है। यह पहला विकल्प है। कृपया लाइन को ध्यान से देखिए।",
    "First option. Please observe the line carefully.": "पहला विकल्प। कृपया लाइन को ध्यान से देखिए।",
    "Now this is the second option. Please observe the line carefully.": "अब यह दूसरा विकल्प है। कृपया लाइन को ध्यान से देखिए।",
    "Second option. Please observe the line carefully.": "दूसरा विकल्प। कृपया लाइन को ध्यान से देखिए।",
    "Which option was better, first option or second option? Say first option, second option, or repeat.": "कौन सा विकल्प बेहतर था, पहला विकल्प या दूसरा विकल्प? पहला विकल्प, दूसरा विकल्प, या फिर से कहिए।",
    "Which was better, first option, second option, or repeat?": "कौन बेहतर था, पहला विकल्प, दूसरा विकल्प, या फिर से?",
    "Which option was better, first option or second option?": "कौन सा विकल्प बेहतर था, पहला विकल्प या दूसरा विकल्प?",
    "Which was better, first option or second option?": "कौन बेहतर था, पहला विकल्प या दूसरा विकल्प?",
}


LOCALIZED_REPROMPT_TEXTS = {
    "hi": {
        "clarity_3way": "कृपया केवल यह कहें: साफ, धुंधला, या फिर से।",
        "comparison_4way": "कृपया केवल यह कहें: पहला विकल्प, दूसरा विकल्प, दोनों समान, या फिर से।",
        "duochrome_4way": "कृपया केवल यह कहें: हरा साइड, लाल साइड, दोनों समान, या फिर से।",
        "distance_bino_4way": "कृपया केवल यह कहें: नीचे की लाइन, ऊपर की लाइन, दोनों समान, या फिर से।",
    },
}


def _localized_reprompt_text(response_type: str, language: Optional[str]) -> str:
    if language and language not in {"auto", "en"}:
        localized = LOCALIZED_REPROMPT_TEXTS.get(language, {}).get(response_type)
        if localized:
            return localized
    return _reprompt_text(response_type)


def _localized_hindi_question(*, state: str, fallback_question: str, retry: bool) -> str:
    compact_cfg = COMPACT_INTERACTIVE_STATE_MAP.get(state)
    if retry and compact_cfg is not None:
        return _localized_reprompt_text(compact_cfg["response_type"], "hi")

    prompt = str(fallback_question or "").strip()
    if not prompt:
        return ""

    translated = HINDI_QUESTION_TRANSLATIONS.get(prompt)
    if translated:
        return translated

    prompt_lc = prompt.lower()
    if state in {"B", "D"} and ("better now" in prompt_lc or "better than before" in prompt_lc or "yes or no" in prompt_lc):
        if "better than before" in prompt_lc:
            return "अब यह पहले से बेहतर है? हाँ या नहीं?"
        return "अब बेहतर है, हाँ या नहीं?"
    if state in {"B", "D"} and "still blurry" in prompt_lc:
        return "क्या अब यह साफ है, अभी भी धुंधला है, या फिर से?"
    if state in {"B", "D", "C", "L"} and "read the line" in prompt_lc:
        if "say blurry, or repeat" in prompt_lc:
            if "if the letters are not clear" in prompt_lc or "if it is not clear" in prompt_lc:
                return "क्या अक्षर साफ हैं, धुंधले हैं, या फिर से?"
            return "साफ, धुंधला, या फिर से?"
        if "say blurry" in prompt_lc:
            return "साफ, धुंधला, या फिर से?"
    if state in {"S", "T"} and ("look carefully at the line" in prompt_lc or "observe the line carefully" in prompt_lc):
        return "कृपया लाइन को ध्यान से देखिए।"
    if state in {"B", "C", "D", "L", "S", "T"} and "read the line" in prompt_lc:
        return "लाइन पढ़िए।"
    if state in {"E", "F", "H", "I", "U"} and "both same" in prompt_lc:
        return "पहला विकल्प, दूसरा विकल्प, या दोनों समान?"
    if state == "U" and "first option" in prompt_lc and "second option" in prompt_lc:
        return "कौन बेहतर था, पहला विकल्प, दूसरा विकल्प, या फिर से?"
    if state in {"G", "J"} and "both same" in prompt_lc:
        return "हरा साइड, लाल साइड, दोनों समान, या फिर से?"
    if state == "K" and "both same" in prompt_lc:
        return "नीचे की लाइन, ऊपर की लाइन, दोनों समान, या फिर से?"
    if state in {"P", "Q", "R"} and ("last line" in prompt_lc or "clear or blurry" in prompt_lc or "letters clear" in prompt_lc or "is it clear" in prompt_lc):
        if "please read the last line" in prompt_lc:
            return "कृपया आखिरी लाइन पढ़िए। क्या यह साफ है, धुंधला है, या क्या मैं फिर से कहूँ?"
        if "is the last line clear or blurry" in prompt_lc:
            return "क्या आखिरी लाइन साफ है या धुंधली?"
        if "please look at the near text" in prompt_lc:
            return "कृपया पास के टेक्स्ट को देखिए। क्या यह साफ है, धुंधला है, या \"फिर से\"?"
        return "साफ, धुंधला, या फिर से?"
    return ""


def _hindi_option_label(option: str, *, state: str, question: str) -> str:
    prompt = str(question or "").lower()
    if state in {"B", "D"} and ("better now" in prompt or "better than before" in prompt or "yes or no" in prompt):
        return {
            "CLEAR": "हाँ",
            "BLURRY": "नहीं",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state in {"B", "D"} and "still blurry" in prompt:
        return {
            "CLEAR": "साफ",
            "BLURRY": "अभी भी धुंधला",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state in {"B", "C", "D", "L", "S", "T"}:
        return {
            "CLEAR": "साफ",
            "BLURRY": "धुंधला",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state in {"E", "F", "H", "I", "U"}:
        return {
            "ONE": "पहला विकल्प",
            "TWO": "दूसरा विकल्प",
            "SAME": "दोनों समान",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state in {"G", "J"}:
        return {
            "RED": "लाल साइड",
            "GREEN": "हरा साइड",
            "SAME": "दोनों समान",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state == "K":
        return {
            "TOP": "ऊपर की लाइन",
            "BOTTOM": "नीचे की लाइन",
            "SAME": "दोनों समान",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state in {"P", "Q", "R"}:
        return {
            "CLEAR": "साफ",
            "BLURRY": "धुंधला",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state in {"S", "T"}:
        return {
            "CLEAR": "लाइन पढ़ी",
            "BLURRY": "धुंधला",
            "REPEAT": "फिर से",
        }.get(option, option)

    if state == "U":
        return {
            "ONE": "पहला विकल्प",
            "TWO": "दूसरा विकल्प",
            "SAME": "दोनों समान",
            "REPEAT": "फिर से",
        }.get(option, option)

    return LOCALIZED_OPTION_LABELS.get("hi", {}).get(option, option)


COMPACT_INTERACTIVE_STATE_MAP = {
    "B": {
        "family": "coarse_clarity",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "REPEAT": "__REPEAT__",
        },
    },
    "D": {
        "family": "coarse_clarity",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "REPEAT": "__REPEAT__",
        },
    },
    "E": {
        "family": "comparison_4way",
        "response_type": "comparison_4way",
        "allowed_labels": ["ONE", "TWO", "SAME", "REPEAT"],
        "label_to_response": {
            "ONE": "BETTER_1",
            "TWO": "BETTER_2",
            "SAME": "SAME",
            "REPEAT": "__REPEAT__",
        },
    },
    "F": {
        "family": "comparison_4way",
        "response_type": "comparison_4way",
        "allowed_labels": ["ONE", "TWO", "SAME", "REPEAT"],
        "label_to_response": {
            "ONE": "BETTER_1",
            "TWO": "BETTER_2",
            "SAME": "SAME",
            "REPEAT": "__REPEAT__",
        },
    },
    "G": {
        "family": "duochrome_4way",
        "response_type": "duochrome_4way",
        "allowed_labels": ["GREEN", "RED", "SAME", "REPEAT"],
        "label_to_response": {
            "RED": "RED_CLEARER",
            "GREEN": "GREEN_CLEARER",
            "SAME": "EQUAL",
            "REPEAT": "__REPEAT__",
        },
    },
    "C": {
        "family": "coarse_clarity",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "REPEAT": "__REPEAT__",
        },
    },
    "H": {
        "family": "comparison_4way",
        "response_type": "comparison_4way",
        "allowed_labels": ["ONE", "TWO", "SAME", "REPEAT"],
        "label_to_response": {
            "ONE": "BETTER_1",
            "TWO": "BETTER_2",
            "SAME": "SAME",
            "REPEAT": "__REPEAT__",
        },
    },
    "I": {
        "family": "comparison_4way",
        "response_type": "comparison_4way",
        "allowed_labels": ["ONE", "TWO", "SAME", "REPEAT"],
        "label_to_response": {
            "ONE": "BETTER_1",
            "TWO": "BETTER_2",
            "SAME": "SAME",
            "REPEAT": "__REPEAT__",
        },
    },
    "J": {
        "family": "duochrome_4way",
        "response_type": "duochrome_4way",
        "allowed_labels": ["GREEN", "RED", "SAME", "REPEAT"],
        "label_to_response": {
            "RED": "RED_CLEARER",
            "GREEN": "GREEN_CLEARER",
            "SAME": "EQUAL",
            "REPEAT": "__REPEAT__",
        },
    },
    "L": {
        "family": "coarse_clarity",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "REPEAT": "__REPEAT__",
        },
    },
    "K": {
        "family": "distance_bino_4way",
        "response_type": "distance_bino_4way",
        "allowed_labels": ["BOTTOM", "TOP", "SAME", "REPEAT"],
        "label_to_response": {
            "TOP": "TOP_CLEARER",
            "BOTTOM": "BOTTOM_CLEARER",
            "SAME": "SAME",
            "REPEAT": "__REPEAT__",
        },
    },
    "P": {
        "family": "near_clarity",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "REPEAT": "__REPEAT__",
        },
    },
    "Q": {
        "family": "near_clarity",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "REPEAT": "__REPEAT__",
        },
    },
    "R": {
        "family": "near_bino_3way",
        "response_type": "clarity_3way",
        "allowed_labels": ["CLEAR", "BLURRY", "REPEAT"],
        "label_to_response": {
            "CLEAR": "TARGET_OK",
            "BLURRY": "NOT_CLEAR",
            "REPEAT": "__REPEAT__",
        },
    },
    "S": {
        "family": "observe_only",
        "response_type": "observe_only",
        "allowed_labels": [],
        "label_to_response": {},
    },
    "T": {
        "family": "observe_only",
        "response_type": "observe_only",
        "allowed_labels": [],
        "label_to_response": {},
    },
    "U": {
        "family": "comparison_4way",
        "response_type": "comparison_4way",
        "allowed_labels": ["ONE", "TWO", "REPEAT"],
        "label_to_response": {
            "ONE": "BETTER_1",
            "TWO": "BETTER_2",
            "REPEAT": "__REPEAT__",
        },
    },
}


COMPACT_LOCALIZED_VOICE_PROMPTS = {
    "coarse_clarity": {
        "en": {
            "initial": "Distance letter chart. Read the line.",
            "retry": "Distance letter chart. Read the line.",
        },
        "hi": {
            "initial": "अक्षर चार्ट। लाइन पढ़िए।",
            "retry": "अक्षर चार्ट। लाइन पढ़िए।",
        },
    },
    "near_clarity": {
        "en": {
            "initial": "Near text chart. Say clear, blurry, or repeat.",
            "retry": "Near text chart. Say clear, blurry, or repeat.",
        },
        "hi": {
            "initial": "पास का text chart। साफ, धुंधला, या फिर से कहिए।",
            "retry": "पास का text chart। केवल साफ, धुंधला, या फिर से कहिए।",
        },
    },
    "near_bino_3way": {
        "en": {
            "initial": "Near text with both eyes. Say clear, blurry, or repeat.",
            "retry": "Near text with both eyes. Say clear, blurry, or repeat.",
        },
        "hi": {
            "initial": "दोनों आँखों से near text। साफ, धुंधला, या फिर से कहिए।",
            "retry": "दोनों आँखों से near text। केवल साफ, धुंधला, या फिर से कहिए।",
        },
    },
    "comparison_4way": {
        "en": {
            "initial": "Dot chart. Say first, second, same, or repeat.",
            "retry": "Dot chart. Say first, second, same, or repeat.",
        },
        "hi": {
            "initial": "Dot chart। पहला, दूसरा, समान, या फिर से कहिए।",
            "retry": "Dot chart। केवल पहला, दूसरा, समान, या फिर से कहिए।",
        },
    },
    "duochrome_4way": {
        "en": {
            "initial": "On which background do the letters appear darker or sharper? Red, Green or are both the same?",
            "retry": "On which background do the letters appear darker or sharper? Red, Green or are both the same?",
        },
        "hi": {
            "initial": "लाल-हरा chart। लाल, हरा, समान, या फिर से कहिए।",
            "retry": "लाल-हरा chart। केवल लाल, हरा, समान, या फिर से कहिए।",
        },
    },
    "distance_bino_4way": {
        "en": {
            "initial": "Which line looks clearer? Top, Bottom, or are they Similar?",
            "retry": "Which line looks clearer? Top, Bottom, or are they Similar?",
        },
        "hi": {
            "initial": "ऊपर-नीचे balance chart। ऊपर, नीचे, समान, या फिर से कहिए।",
            "retry": "ऊपर-नीचे balance chart। केवल ऊपर, नीचे, समान, या फिर से कहिए।",
        },
    },
}


COMPACT_MATCH_ALIASES = {
    "clarity_3way": {
        "CLEAR": {
            "exact": ["clear", "readable", "visible", "sharp", "crisp", "yes", "good", "fine", "okay", "ok", "clear hai", "साफ", "ठीक", "सही", "haan", "हाँ"],
            "phrases": [
                "it is clear", "looks clear", "very clear", "fully clear", "i can read", "i can read it", "i can see it",
                "readable hai", "साफ है", "साफ दिख रहा है", "दिख रहा है", "पढ़ पा रहा हूं", "padh pa raha", "saaf dikh raha", "saf hai", "saf dikh raha",
                "comfortable", "clear and comfortable", "आराम है",
                "letters clear", "last line clear", "read last line", "got clearer", "became clearer",
                "got better", "became better", "it got better", "yes it got better", "better now", "better than before", "yes better than before",
                "क्लियर", "क्लियर है", "हाँ बेहतर है", "हां बेहतर है", "हाँ बेहतर", "हां बेहतर",
            ],
            "fuzzy": ["bloody clear", "cleer", "clar", "cler", "clir", "klear", "kleer", "saaf", "saf", "theek", "thik", "haanji", "पड़बारा हूं", "पढ़वाराएं", "here", "hear", "cheers", "cheer"],
        },
        "BLURRY": {
            "exact": ["blur", "blurry", "unclear", "hazy", "fuzzy", "bloody", "धुंधला", "ब्लर", "not clear", "no", "nahin", "nahi", "नहीं"],
            "phrases": [
                "it is blurry", "looks blurry", "slightly blurry", "not clear", "not that clear", "not very clear",
                "its not clear", "it is not clear", "not so clear", "still not clear", "not sharp", "not readable",
                "hard to read", "difficult to read", "nahi clear", "clear nahi",
                "धुंधला है", "ब्लर है", "थोड़ा blur है", "साफ नहीं", "थुला है", "हुलाहुला", "थुला थुला", "dhundla hai", "dhundhla hai",
                "not comfortable", "eye strain", "still blurry", "not good", "not proper",
                "letters blurry", "more blurry", "got more blurry", "became more blurry",
                "not better", "did not get better", "didn't get better", "no better", "not improved",
            ],
            "fuzzy": ["blurr", "blury", "blr", "bluddy", "bladi", "dhundla", "dhundhla", "dhundlaa", "dhundhlaa", "bilari", "blari", "blairy", "larry", "thula", "thula hai", "hulahula"],
        },
        "REPEAT": {
            "exact": ["repeat", "again", "dubara", "दोबारा", "फिर", "again please", "repeat please"],
            "phrases": [
                "say again", "please repeat", "repeat question", "repeat please", "again please",
                "फिर से", "दोबारा बोलिए", "फिर से बोलिए", "question repeat", "repeat karo", "फिरसे", "फिर से कहिए", "fir se", "firse", "phir se", "phirse",
                "again say",
                "not sure", "don't know", "cannot tell", "hard to tell", "समझ नहीं आ रहा",
                "not readable", "cannot read", "can't read", "cannot see", "can't see", "कुछ नहीं दिख रहा", "nahi dikh raha",
            ],
            "fuzzy": ["repit", "repeet", "dobara", "phis se", "fir se", "firse", "phirse", "cant tell", "notshur"],
        },
    },
    "comparison_4way": {
        "ONE": {
            "exact": ["1", "one", "first", "ek", "एक"],
            "phrases": ["option one", "first one", "number one", "one is better", "first is better", "first option", "एक बेहतर", "पहला", "pehla vikalp", "पहला विकल्प"],
            "fuzzy": ["won", "wan", "wahn", "on", "ik", "yek", "pahla", "pehla"],
        },
        "TWO": {
            "exact": ["2", "two", "second", "do", "दो"],
            "phrases": ["option two", "second one", "number two", "two is better", "second is better", "second option", "दो बेहतर", "दूसरा", "dusra vikalp", "doosra vikalp", "दूसरा विकल्प"],
            "fuzzy": ["to", "too", "tu", "doo", "dusra", "doosra"],
        },
        "SAME": {
            "exact": ["same", "equal", "both", "बराबर", "barabar", "दोनों", "dono"],
            "phrases": ["both same", "both are same", "no difference", "about the same", "दोनों same", "कोई फर्क नहीं", "dono saman", "दोनों समान"],
            "fuzzy": ["barabr", "saman"],
        },
        "REPEAT": {
            "exact": ["repeat", "again", "dubara", "दोबारा"],
            "phrases": ["say again", "please repeat", "repeat please", "can't tell", "cannot tell", "hard to tell", "not sure", "समझ नहीं आ रहा", "फिर से", "फिरसे"],
            "fuzzy": ["repit", "repeet", "cant tell", "pata nai", "fir se", "firse", "phirse"],
        },
    },
    "duochrome_4way": {
        "RED": {
            "exact": ["red", "लाल", "lal"],
            "phrases": ["red side", "red is clearer", "red side clearer", "लाल साफ है", "lal side", "laal side"],
            "fuzzy": ["redd", "laal"],
        },
        "GREEN": {
            "exact": ["green", "हरा", "hara"],
            "phrases": ["green side", "green is clearer", "green side clearer", "हरा साफ है", "hara side"],
            "fuzzy": ["grean", "haraa"],
        },
        "SAME": {
            "exact": ["same", "equal", "both", "बराबर", "barabar", "दोनों", "dono"],
            "phrases": ["both same", "both are same", "no difference", "दोनों same"],
            "fuzzy": ["barabr"],
        },
        "REPEAT": {
            "exact": ["repeat", "again", "dubara", "दोबारा"],
            "phrases": ["say again", "please repeat", "can't tell", "cannot tell", "hard to tell", "समझ नहीं आ रहा", "फिर से", "फिरसे"],
            "fuzzy": ["repit", "cant tell", "fir se", "firse", "phirse"],
        },
    },
    "distance_bino_4way": {
        "TOP": {
            "exact": ["top", "upper", "ऊपर", "upar"],
            "phrases": ["top is clearer", "top better", "top line", "top line clearer", "ऊपर साफ है", "upar line", "upar ki line"],
            "fuzzy": ["topp", "uparr"],
        },
        "BOTTOM": {
            "exact": ["bottom", "lower", "नीचे", "neeche"],
            "phrases": ["bottom is clearer", "bottom better", "bottom line", "bottom line clearer", "नीचे साफ है", "neeche line", "neeche ki line"],
            "fuzzy": ["botom", "niche", "neecha"],
        },
        "SAME": {
            "exact": ["same", "equal", "both", "बराबर", "barabar", "दोनों", "dono"],
            "phrases": ["both same", "both are same", "no difference", "दोनों same"],
            "fuzzy": ["barabr"],
        },
        "REPEAT": {
            "exact": ["repeat", "again", "dubara", "दोबारा"],
            "phrases": ["say again", "please repeat", "can't tell", "hard to tell", "समझ नहीं आ रहा", "फिर से", "फिरसे"],
            "fuzzy": ["repit", "cant tell", "fir se", "firse", "phirse"],
        },
    },
}


HINDI_CLEAR_VARIANTS = _merge_variant_groups(
    ["साफ", "साफ है", "ठीक", "ठीक है", "सही", "सही है", "हाँ", "हाँ जी"],
    _expand_with_suffixes(
        ["saaf", "saf", "safi", "saif", "safe", "saff", "saafh", "safh", "saafi", "safee", "saph", "sahf"],
        ["hai", "he", "hey", "h", "hai ji", "he ji", "ji"],
    ),
    _expand_with_suffixes(
        ["haan", "han", "haan ji", "han ji", "haanji", "hanji", "haan जी", "han जी"],
        ["", "please"],
    ),
    _expand_with_suffixes(
        ["theek", "thik", "thek", "tik", "teek", "sahi", "sai"],
        ["hai", "he", "hey", "h", "ji"],
    ),
    [
        "saaf dikh raha",
        "saf dikh raha",
        "saaf dikhraha",
        "saf dikhraha",
        "saaf nazar aa raha",
        "saf nazar aa raha",
        "clear hai",
        "readable hai",
        "ab saaf hai",
        "ab saf hai",
        "bilkul saaf",
        "ekdum saaf",
    ],
)

HINDI_BLURRY_VARIANTS = _merge_variant_groups(
    ["धुंधला", "धुंधला है", "धुंदला", "धुन्दला", "ब्लर", "ब्लर है", "नहीं", "साफ नहीं"],
    _expand_with_suffixes(
        [
            "dhundla", "dhundhla", "dundla", "dundhla", "dhoondla", "dhoondhla",
            "dhudla", "dhudhla", "dudla", "dudhla", "dundle", "jule", "jula",
            "jhula", "thula", "hulahula", "hula hula", "blurry", "blur", "blurr", "blari",
        ],
        ["hai", "he", "hey", "h", "hain", "sa"],
    ),
    [
        "abhi bhi dhundla",
        "abhi bhi dhundhla",
        "abhi bhi blurry",
        "saaf nahi",
        "saf nahi",
        "nahi saaf",
        "not clear",
        "blur hai",
        "blurry hai",
        "nahi dikh raha",
        "abhi dhundla hai",
        "bilkul dhundla",
        "ekdum dhundla",
        "still dhundla",
        "still blurry",
        "no clear",
        "nahi hua better",
    ],
    _expand_with_suffixes(["nahi", "nahin", "nhi", "nai", "na"], ["", "ji", "hai"]),
)

HINDI_REPEAT_VARIANTS = _merge_variant_groups(
    ["फिर से", "फिरसे", "दोबारा", "repeat", "again"],
    _expand_with_suffixes(
        [
            "fir se", "phir se", "firse", "phirse", "firsey", "phirsey",
            "fir say", "phir say", "fir see", "phir see", "repeat", "again",
            "dobara", "dobaara", "dobaraa", "doobara", "dubara", "duwara",
        ],
        ["boliye", "bolo", "kahiye", "please", "kar do", "again"],
    ),
    [
        "samajh nahi aaya",
        "samajh nahi aya",
        "samaj nahi aaya",
        "phir se kahiye",
        "phirse boliye",
        "fir se boliye",
        "dobara boliye",
        "dubara boliye",
        "ek baar aur",
        "once more",
        "repeat please",
        "again please",
    ],
)

HINDI_ONE_VARIANTS = _merge_variant_groups(
    ["पहला", "पहला विकल्प"],
    _expand_with_suffixes(
        ["pehla", "pahla", "pahila", "pehlaa", "phela", "pela", "pehला"],
        ["option", "vikalp", "wala", "better", "clear", "saaf", "number one", "option hai"],
    ),
    [
        "option one",
        "first option",
        "pehla better hai",
        "pehla zyada saaf",
        "pehla clear hai",
        "pehla wala",
        "pahla vikalp",
        "pahla option",
    ],
)

HINDI_TWO_VARIANTS = _merge_variant_groups(
    ["दूसरा", "दूसरा विकल्प", "दो"],
    _expand_with_suffixes(
        ["dusra", "doosra", "dosra", "doosra", "duusra", "dugra", "do"],
        ["option", "vikalp", "wala", "better", "clear", "saaf", "number two", "option hai"],
    ),
    [
        "option two",
        "second option",
        "dusra better hai",
        "dusra zyada saaf",
        "dusra clear hai",
        "dusra wala",
        "doosra vikalp",
        "doosra option",
    ],
)

HINDI_SAME_VARIANTS = _merge_variant_groups(
    ["दोनों", "दोनों समान", "बराबर"],
    [
        "dono",
        "dono same",
        "dono saman",
        "dono barabar",
        "barabar",
        "same hai",
        "same same",
        "koi fark nahi",
        "no difference",
        "ek jaisa",
        "bilkul same",
        "lagbhag same",
        "barabar hai",
        "dono ek jaisa",
        "dono bilkul same",
        "same same",
        "same hi hai",
        "same lag raha",
        "saman hai",
        "bilkul barabar",
        "almost same",
        "kafi same",
        "same jaisa",
        "ekdam same",
        "dono hi same",
        "dono hi barabar",
        "dono equal",
        "equal hai",
        "ek jaisi clarity",
        "same clarity",
    ],
    _expand_with_templates(
        ["dono", "both", "ye dono", "donon"],
        [
            "{root} same",
            "{root} saman",
            "{root} barabar",
            "{root} same hai",
            "{root} ek jaisa",
            "{root} bilkul same",
            "{root} equal",
            "{root} equal hai",
            "{root} same lag raha",
        ],
    ),
)

HINDI_RED_VARIANTS = _merge_variant_groups(
    ["लाल", "लाल साइड"],
    _expand_with_suffixes(
        ["lal", "laal", "laalh", "laal side", "red", "rad"],
        ["side", "wala", "better", "saaf", "clear", "clearer", "zyada saaf", "side hai", "zyaada clear"],
    ),
    [
        "lal side better",
        "laal side clear",
        "red side better",
        "red side clear",
        "lal better hai",
        "laal better hai",
        "lal zyada saaf",
        "laal zyada saaf",
        "lal clear hai",
        "laal clear hai",
        "red clearer hai",
        "red hi better",
        "lal hi better",
    ],
)

HINDI_GREEN_VARIANTS = _merge_variant_groups(
    ["हरा", "हरा साइड"],
    _expand_with_suffixes(
        ["hara", "haraa", "hara side", "har", "harr", "hari", "हरी", "हरा", "green", "grean"],
        ["side", "wala", "better", "saaf", "clear", "clearer", "zyada saaf", "side hai", "zyaada clear"],
    ),
    [
        "hara side better",
        "green side clear",
        "green side better",
        "hara side zyada saaf",
        "hara better hai",
        "green better hai",
        "hara clear hai",
        "green clear hai",
        "hara zyada saaf",
        "green zyada saaf",
        "green clearer hai",
        "green hi better",
        "hara hi better",
        "har side",
        "har clear",
        "hari side",
    ],
)

HINDI_TOP_VARIANTS = _merge_variant_groups(
    ["ऊपर", "ऊपर की लाइन"],
    _expand_with_suffixes(
        ["upar", "uparr", "upper", "top", "top line", "upar line"],
        ["line", "ki line", "wala", "better", "saaf", "clear", "clearer", "zyada saaf", "line hai", "zyaada clear"],
    ),
    [
        "upar line better",
        "top line clear",
        "upar ki line better",
        "top zyada saaf",
        "upar better hai",
        "top better hai",
        "upar clear hai",
        "top clear hai",
        "upar wali line",
        "top wali line",
        "upar hi better",
        "top hi better",
        "upper line better",
    ],
)

HINDI_BOTTOM_VARIANTS = _merge_variant_groups(
    ["नीचे", "नीचे की लाइन"],
    _expand_with_suffixes(
        ["neeche", "niche", "neche", "lower", "bottom"],
        ["line", "ki line", "wala", "better", "saaf", "clear", "clearer", "zyada saaf", "line hai"],
    ),
    ["neeche line better", "bottom line clear", "neeche ki line better", "bottom zyada saaf"],
)


ENGLISH_CLEAR_VARIANTS = _merge_variant_groups(
    [
        "clear", "clear now", "clearly", "readable", "readable now", "visible", "visible now",
        "sharp", "sharp now", "crisp", "good", "good now", "fine", "fine now", "okay", "ok",
        "yes", "yeah", "yep", "yup", "all clear", "crystal clear",
    ],
    [
        "it's clear", "it is clear", "looks clear", "looks good", "looks sharp", "looks readable",
        "i can read", "i can read it", "i can read now", "i can see it", "i can see now",
        "got better", "better now", "better than before", "much better", "more clear", "fully clear", "very clear",
        "all letters clear", "everything clear", "everything is clear", "easy to read",
        "comfortable now", "perfectly clear", "super clear", "can read this", "can read line",
        "yes better", "yes it is better", "yes clearer", "yes it is clear", "looks fine",
    ],
    ["clearenough", "allclear", "readablenow", "visibleenough"],
)

ENGLISH_BLURRY_VARIANTS = _merge_variant_groups(
    [
        "blur", "blurry", "blurred", "unclear", "hazy", "fuzzy", "not clear", "no", "nope",
        "still blurry", "blurrier", "more blurry",
    ],
    [
        "it is blurry", "it's blurry", "looks blurry", "still not clear", "hard to read",
        "difficult to read", "not readable", "not visible", "not sharp", "not crisp",
        "got worse", "more blurry now", "became more blurry", "not better", "no better",
        "did not get better", "didn't get better", "not improved", "out of focus",
        "still not readable", "still not visible", "cannot read clearly", "too blurry",
        "blurry now", "worse now", "still fuzzy", "too hazy", "cannot read this",
    ],
    ["notclear", "stillblurry", "moreblurry", "blurrynow"],
)

ENGLISH_REPEAT_VARIANTS = _merge_variant_groups(
    [
        "repeat", "again", "once more", "one more time", "say again", "repeat please",
        "please repeat", "come again", "pardon", "repeat that", "again please",
    ],
    [
        "can you repeat", "please say again", "say that again", "repeat the question",
        "i am not sure", "not sure", "can't tell", "cannot tell", "hard to tell",
        "too close to tell", "no idea", "say again please", "repeat it", "tell me again",
    ],
    ["repeatplease", "sayagain", "bothsameagain"],
)

ENGLISH_ONE_VARIANTS = _merge_variant_groups(
    [
        "one", "1", "first", "option one", "option 1", "first option", "first one",
        "number one", "one is better", "first is better", "one better", "first better",
    ],
    [
        "one is clearer", "first is clearer", "one looks better", "first looks better",
        "i choose one", "i prefer one", "going with one", "one is sharper",
        "first is sharper", "one is more clear", "first option better", "one clearer",
        "first clearer", "option one better", "first one better",
    ],
    ["firstoption", "optionone", "onebetter", "firstbetter"],
)

ENGLISH_TWO_VARIANTS = _merge_variant_groups(
    [
        "two", "2", "second", "option two", "option 2", "second option", "second one",
        "number two", "two is better", "second is better", "two better", "second better",
    ],
    [
        "two is clearer", "second is clearer", "two looks better", "second looks better",
        "i choose two", "i prefer two", "going with two", "two is sharper",
        "second is sharper", "two is more clear", "second option better", "two clearer",
        "second clearer", "option two better", "second one better",
    ],
    ["secondoption", "optiontwo", "twobetter", "secondbetter"],
)

ENGLISH_SAME_VARIANTS = _merge_variant_groups(
    [
        "same", "both", "equal", "similar", "both same", "both are same",
        "no difference", "about the same", "identical", "same same",
    ],
    [
        "they are the same", "they look the same", "both look same", "both look the same",
        "both are similar", "equally clear", "equally blurry", "pretty much same",
        "looks identical", "almost the same", "still the same", "both seem same",
        "both equal", "same on both", "same for both",
    ],
    ["bothsame", "botharesame", "lookssame", "samesame", "bothlooksame", "bothlooksthesame", "bootsame", "butsame"],
)

ENGLISH_RED_VARIANTS = _merge_variant_groups(
    ["red", "red side", "red one", "reddish side"],
    [
        "red is better", "red is clearer", "red looks better", "red looks clearer",
        "red side better", "red side clearer", "red is darker", "red looks darker",
        "red is sharper", "red sharper", "red side sharper", "more red", "red clearer",
    ],
    ["redside", "readside", "redclearer"],
)

ENGLISH_GREEN_VARIANTS = _merge_variant_groups(
    ["green", "green side", "green one", "greenish side"],
    [
        "green is better", "green is clearer", "green looks better", "green looks clearer",
        "green side better", "green side clearer", "green is darker", "green looks darker",
        "green is sharper", "green sharper", "green side sharper", "more green", "green clearer",
    ],
    ["greenside", "greenclearer", "greenone"],
)

ENGLISH_TOP_VARIANTS = _merge_variant_groups(
    ["top", "top line", "upper", "upper line"],
    [
        "top is better", "top is clearer", "top looks better", "top looks clearer",
        "top line better", "top line clearer", "upper is better", "upper line better",
        "top sharper", "top line sharper", "upper looks clearer", "top clearer",
    ],
    ["topline", "upperline", "topclearer"],
)

ENGLISH_BOTTOM_VARIANTS = _merge_variant_groups(
    ["bottom", "bottom line", "lower", "lower line"],
    [
        "bottom is better", "bottom is clearer", "bottom looks better", "bottom looks clearer",
        "bottom line better", "bottom line clearer", "lower is better", "lower line better",
        "bottom sharper", "bottom line sharper", "lower looks clearer", "bottom clearer",
    ],
    ["bottomline", "lowerline", "bottomclearer"],
)


HINDI_VARIANT_LIBRARY = {
    "CLEAR": HINDI_CLEAR_VARIANTS,
    "BLURRY": HINDI_BLURRY_VARIANTS,
    "REPEAT": HINDI_REPEAT_VARIANTS,
    "ONE": HINDI_ONE_VARIANTS,
    "TWO": HINDI_TWO_VARIANTS,
    "SAME": HINDI_SAME_VARIANTS,
    "RED": HINDI_RED_VARIANTS,
    "GREEN": HINDI_GREEN_VARIANTS,
    "TOP": HINDI_TOP_VARIANTS,
    "BOTTOM": HINDI_BOTTOM_VARIANTS,
}

ENGLISH_VARIANT_LIBRARY = {
    "CLEAR": ENGLISH_CLEAR_VARIANTS,
    "BLURRY": ENGLISH_BLURRY_VARIANTS,
    "REPEAT": ENGLISH_REPEAT_VARIANTS,
    "ONE": ENGLISH_ONE_VARIANTS,
    "TWO": ENGLISH_TWO_VARIANTS,
    "SAME": ENGLISH_SAME_VARIANTS,
    "RED": ENGLISH_RED_VARIANTS,
    "GREEN": ENGLISH_GREEN_VARIANTS,
    "TOP": ENGLISH_TOP_VARIANTS,
    "BOTTOM": ENGLISH_BOTTOM_VARIANTS,
}


def _extend_compact_hindi_matcher_variants() -> None:
    corrections = FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["normalization"]["common_asr_corrections"]
    canonical_targets = {
        "CLEAR": "saaf hai",
        "BLURRY": "dhundla hai",
        "REPEAT": "fir se",
        "ONE": "pehla",
        "TWO": "dusra",
        "SAME": "dono saman",
        "RED": "lal",
        "GREEN": "hara",
        "TOP": "upar",
        "BOTTOM": "neeche",
    }

    alias_targets = {
        ("clarity_3way", "CLEAR"): HINDI_CLEAR_VARIANTS,
        ("clarity_3way", "BLURRY"): HINDI_BLURRY_VARIANTS,
        ("clarity_3way", "REPEAT"): HINDI_REPEAT_VARIANTS,
        ("comparison_4way", "ONE"): HINDI_ONE_VARIANTS,
        ("comparison_4way", "TWO"): HINDI_TWO_VARIANTS,
        ("comparison_4way", "SAME"): HINDI_SAME_VARIANTS,
        ("comparison_4way", "REPEAT"): HINDI_REPEAT_VARIANTS,
        ("duochrome_4way", "RED"): HINDI_RED_VARIANTS,
        ("duochrome_4way", "GREEN"): HINDI_GREEN_VARIANTS,
        ("duochrome_4way", "SAME"): HINDI_SAME_VARIANTS,
        ("duochrome_4way", "REPEAT"): HINDI_REPEAT_VARIANTS,
        ("distance_bino_4way", "TOP"): HINDI_TOP_VARIANTS,
        ("distance_bino_4way", "BOTTOM"): HINDI_BOTTOM_VARIANTS,
        ("distance_bino_4way", "SAME"): HINDI_SAME_VARIANTS,
        ("distance_bino_4way", "REPEAT"): HINDI_REPEAT_VARIANTS,
    }

    for label, variants in HINDI_VARIANT_LIBRARY.items():
        canonical = canonical_targets[label]
        for variant in variants:
            if variant and variant != canonical:
                corrections.setdefault(variant, canonical)

    for (family, label), variants in alias_targets.items():
        alias_cfg = COMPACT_MATCH_ALIASES[family][label]
        alias_cfg["exact"] = _merge_variant_groups(alias_cfg.get("exact", []), variants)

    english_alias_targets = {
        ("clarity_3way", "CLEAR"): ENGLISH_CLEAR_VARIANTS,
        ("clarity_3way", "BLURRY"): ENGLISH_BLURRY_VARIANTS,
        ("clarity_3way", "REPEAT"): ENGLISH_REPEAT_VARIANTS,
        ("comparison_4way", "ONE"): ENGLISH_ONE_VARIANTS,
        ("comparison_4way", "TWO"): ENGLISH_TWO_VARIANTS,
        ("comparison_4way", "SAME"): ENGLISH_SAME_VARIANTS,
        ("comparison_4way", "REPEAT"): ENGLISH_REPEAT_VARIANTS,
        ("duochrome_4way", "RED"): ENGLISH_RED_VARIANTS,
        ("duochrome_4way", "GREEN"): ENGLISH_GREEN_VARIANTS,
        ("duochrome_4way", "SAME"): ENGLISH_SAME_VARIANTS,
        ("duochrome_4way", "REPEAT"): ENGLISH_REPEAT_VARIANTS,
        ("distance_bino_4way", "TOP"): ENGLISH_TOP_VARIANTS,
        ("distance_bino_4way", "BOTTOM"): ENGLISH_BOTTOM_VARIANTS,
        ("distance_bino_4way", "SAME"): ENGLISH_SAME_VARIANTS,
        ("distance_bino_4way", "REPEAT"): ENGLISH_REPEAT_VARIANTS,
    }

    english_canonical_targets = {
        "CLEAR": "clear",
        "BLURRY": "blurry",
        "REPEAT": "repeat",
        "ONE": "first option",
        "TWO": "second option",
        "SAME": "both same",
        "RED": "red side",
        "GREEN": "green side",
        "TOP": "top line",
        "BOTTOM": "bottom line",
    }
    for label, variants in ENGLISH_VARIANT_LIBRARY.items():
        canonical = english_canonical_targets[label]
        for variant in variants:
            if variant and variant != canonical:
                corrections.setdefault(variant, canonical)

    for (family, label), variants in english_alias_targets.items():
        alias_cfg = COMPACT_MATCH_ALIASES[family][label]
        alias_cfg["exact"] = _merge_variant_groups(alias_cfg.get("exact", []), variants)


_extend_compact_hindi_matcher_variants()


COARSE_RESPONSE_STATES = {"B", "C", "D", "L"}


def _question_expects_line_reading(*, state: str, question: Optional[str], language: Optional[str] = None) -> bool:
    if language == "hi" and state in {"B", "C", "D", "L"}:
        return False
    prompt = str(question or "").strip().lower()
    if state in {"C", "L"}:
        return True
    if state not in {"B", "D"}:
        return False
    if not prompt:
        return True
    if (
        "better now" in prompt
        or "better than before" in prompt
        or "yes or no" in prompt
        or "हाँ या नहीं" in prompt
        or "ha ya nahi" in prompt
        or "बेहतर" in prompt
    ):
        return False
    return True


def _is_line_clarity_phase(*, state: str, question: Optional[str]) -> bool:
    if state in {"C", "L"}:
        return True
    if state not in {"B", "D"}:
        return False
    prompt = str(question or "").strip().lower()
    if (
        "better now" in prompt
        or "better than before" in prompt
        or "yes or no" in prompt
        or "हाँ या नहीं" in prompt
        or "ha ya nahi" in prompt
        or "बेहतर" in prompt
    ):
        return False
    return True


CHART_LETTER_ALIASES = {
    "A": {"a", "ay", "eh", "ए"},
    "B": {"b", "bee", "be", "बी"},
    "C": {"c", "see", "cee", "sea", "सी"},
    "D": {"d", "dee", "di", "डी"},
    "E": {"e", "ee", "ई"},
    "F": {"f", "eff", "एफ"},
    "G": {"g", "gee", "ji", "जी"},
    "H": {"h", "aitch", "etch", "एच"},
    "K": {"k", "kay", "के"},
    "L": {"l", "ell", "el", "एल"},
    "N": {"n", "en", "एन"},
    "O": {"o", "oh", "ओ"},
    "P": {"p", "pee", "पी"},
    "R": {"r", "are", "ar", "आर"},
    "S": {"s", "ess", "एस"},
    "T": {"t", "tee", "ti", "टी"},
    "V": {"v", "vee", "we", "वी"},
    "Z": {"z", "zee", "zed", "जेड", "ज़ेड"},
}


LANGUAGE_SCRIPT_PATTERNS = {
    "hi": re.compile(r"[\u0900-\u097F]"),
}


ROMANIZED_LANGUAGE_HINTS = {
    "hi": {
        "haan",
        "haanji",
        "theek",
        "sahi",
        "dhundla",
        "barabar",
        "lal",
        "hara",
        "upar",
        "neeche",
        "pehla",
        "dusra",
        "padh",
        "padh",
        "dikh",
    },
}


STATE_CONFIG_MAP = {
    "B": {
        "library_key": "B_COARSE_SPHERE",
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "NOT_VISIBLE": "NOT_READABLE",
        },
    },
    "D": {
        "library_key": "B_COARSE_SPHERE",
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "NOT_VISIBLE": "NOT_READABLE",
        },
    },
    "E": {
        "library_key": "E_F_JCC_COMPARISON",
        "label_to_response": {
            "BETTER_1": "BETTER_1",
            "BETTER_2": "BETTER_2",
            "SAME": "SAME",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "F": {
        "library_key": "E_F_JCC_COMPARISON",
        "label_to_response": {
            "BETTER_1": "BETTER_1",
            "BETTER_2": "BETTER_2",
            "SAME": "SAME",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "H": {
        "library_key": "E_F_JCC_COMPARISON",
        "label_to_response": {
            "BETTER_1": "BETTER_1",
            "BETTER_2": "BETTER_2",
            "SAME": "SAME",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "I": {
        "library_key": "E_F_JCC_COMPARISON",
        "label_to_response": {
            "BETTER_1": "BETTER_1",
            "BETTER_2": "BETTER_2",
            "SAME": "SAME",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "G": {
        "library_key": "G_J_DUOCHROME",
        "label_to_response": {
            "RED_CLEARER": "RED_CLEARER",
            "GREEN_CLEARER": "GREEN_CLEARER",
            "SAME": "EQUAL",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "J": {
        "library_key": "G_J_DUOCHROME",
        "label_to_response": {
            "RED_CLEARER": "RED_CLEARER",
            "GREEN_CLEARER": "GREEN_CLEARER",
            "SAME": "EQUAL",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "K": {
        "library_key": "K_BINOCULAR_BALANCE",
        "label_to_response": {
            "TOP_CLEARER": "TOP_CLEARER",
            "BOTTOM_CLEARER": "BOTTOM_CLEARER",
            "SAME": "SAME",
            "CANT_TELL": "CANT_TELL",
        },
    },
    "P": {
        "library_key": "P_Q_NEAR_CLARITY",
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "NOT_VISIBLE": "NOT_READABLE",
        },
    },
    "Q": {
        "library_key": "P_Q_NEAR_CLARITY",
        "label_to_response": {
            "CLEAR": "READABLE",
            "BLURRY": "BLURRY",
            "NOT_VISIBLE": "NOT_READABLE",
        },
    },
    "R": {
        "library_key": "R_NEAR_BINOCULAR",
        "label_to_response": {
            "TARGET_OK": "TARGET_OK",
            "NEED_ADJUSTMENT": "NOT_CLEAR",
            "NOT_COMFORTABLE": "NOT_CLEAR",
        },
    },
}


@dataclass
class MatchResult:
    transcript: str
    normalized_text: str
    response_type: str
    canonical_label: Optional[str]
    response_value: Optional[str]
    confidence: float
    method: str
    reprompt_text: str
    accepted: bool
    reason: str = ""


@dataclass
class LanguageChoiceResult:
    transcript: str
    normalized_text: str
    language_code: Optional[str]
    confidence: float
    method: str
    accepted: bool
    reprompt_text: str
    reason: str = ""


def normalize_text(text: str, corrections: dict[str, str]) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    fillers = set(FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["normalization"]["remove_fillers"])
    tokens = [token for token in text.split() if token not in fillers]
    text = " ".join(tokens)

    if text in corrections:
        text = corrections[text]

    corrected = [corrections.get(token, token) for token in text.split()]
    return " ".join(corrected)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _phrase_match_score(normalized_text: str, normalized_item: str) -> float:
    if not normalized_text or not normalized_item:
        return 0.0
    if normalized_text == normalized_item:
        return 1.0
    if normalized_item in normalized_text:
        return 0.95 + min(0.02, 0.005 * len(normalized_item.split()))
    if normalized_text in normalized_item:
        text_tokens = normalized_text.split()
        item_tokens = normalized_item.split()
        if len(text_tokens) >= 2 and all(token in item_tokens for token in text_tokens):
            return 0.88 + min(0.02, 0.005 * len(text_tokens))
    return 0.0


def _semantic_similarity(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    return max(jaccard, similarity(a, b))


def _reprompt_text(response_type: str) -> str:
    compact_map = {
        "clarity_3way": "Please say only: clear, blurry, or repeat.",
        "comparison_4way": "Please say only: first option, second option, both same, or repeat.",
        "duochrome_4way": "Please say only: green side, red side, both same, or repeat.",
        "distance_bino_4way": "Please say only: bottom line, top line, both same, or repeat.",
    }
    if response_type in compact_map:
        return compact_map[response_type]
    return FSM_AUDIO_RESPONSE_LIBRARY_V1["fallback_policy"]["reprompt_texts"].get(
        response_type,
        "Please repeat one of the shown response options.",
    )


def localized_voice_prompt(
    *,
    state: str,
    language: Optional[str],
    retry: bool,
    fallback_question: str,
) -> str:
    if language == "hi":
        translated = _localized_hindi_question(
            state=state,
            fallback_question=fallback_question,
            retry=retry,
        )
        if translated:
            return translated

    compact_cfg = COMPACT_INTERACTIVE_STATE_MAP.get(state)
    if compact_cfg is not None:
        if not language or language in {"auto", "en"}:
            if retry:
                return _reprompt_text(compact_cfg["response_type"])
            return fallback_question
        family = compact_cfg["family"]
        localized_prompt = COMPACT_LOCALIZED_VOICE_PROMPTS.get(family, {}).get(language or "en")
        if localized_prompt is None:
            localized_prompt = COMPACT_LOCALIZED_VOICE_PROMPTS.get(family, {}).get("en")
        if localized_prompt is not None:
            return localized_prompt["retry" if retry else "initial"]

    if not language or language in {"auto", "en"}:
        if retry:
            _, state_cfg = _state_match_config(state)
            return _reprompt_text(state_cfg["response_type"])
        return fallback_question

    _, state_cfg = _state_match_config(state)
    response_type = state_cfg["response_type"]
    localized_prompt = LOCALIZED_VOICE_PROMPTS.get(language, {}).get(response_type)
    if localized_prompt is None:
        if retry:
            return _reprompt_text(response_type)
        return fallback_question

    return localized_prompt["retry" if retry else "initial"]


def localized_language_selection_prompt(*, retry: bool = False) -> str:
    return LOCALIZED_LANGUAGE_SELECTION_PROMPTS["retry" if retry else "initial"]


def _english_option_label(option: str, *, state: str, question: str) -> str:
    prompt = str(question or "").lower()
    if state in {"B", "D"} and ("better now" in prompt or "better than before" in prompt or "yes or no" in prompt):
        return {
            "CLEAR": "Yes",
            "BLURRY": "No",
            "REPEAT": "Repeat",
        }.get(option, option)

    if state in {"B", "D"} and "still blurry" in prompt:
        return {
            "CLEAR": "Can read now",
            "BLURRY": "Still blurry",
            "REPEAT": "Repeat",
        }.get(option, option)

    if state in {"B", "C", "D", "L", "S", "T"}:
        return {
            "CLEAR": "Read line",
            "BLURRY": "Blurry",
            "REPEAT": "Repeat",
        }.get(option, option)

    if state in {"E", "F", "H", "I", "U"}:
        return {
            "ONE": "First option",
            "TWO": "Second option",
            "SAME": "Both same",
            "REPEAT": "Repeat",
        }.get(option, option)

    if state in {"G", "J"}:
        return {
            "RED": "Red side",
            "GREEN": "Green side",
            "SAME": "Both same",
            "REPEAT": "Repeat",
        }.get(option, option)

    if state == "K":
        return {
            "TOP": "Top line",
            "BOTTOM": "Bottom line",
            "SAME": "Both same",
            "REPEAT": "Repeat",
        }.get(option, option)

    if state in {"P", "Q", "R"}:
        return {
            "CLEAR": "Clear",
            "BLURRY": "Blurry",
            "REPEAT": "Repeat",
        }.get(option, option)

    return option


def localized_option_label(option: str, language: Optional[str], *, state: str = "", question: str = "") -> str:
    if not language or language in {"auto", "en"}:
        return _english_option_label(option, state=state, question=question)
    if language == "hi":
        return _hindi_option_label(option, state=state, question=question)
    return LOCALIZED_OPTION_LABELS.get(language, {}).get(option, option)


def interactive_option_labels(state: str) -> list[str]:
    compact_cfg = COMPACT_INTERACTIVE_STATE_MAP.get(state)
    if compact_cfg is not None:
        return list(compact_cfg["allowed_labels"])

    state_map, state_cfg = _state_match_config(state)
    ordered_labels: list[str] = []
    for label in state_cfg["labels"].keys():
        response_value = state_map["label_to_response"].get(label)
        if response_value is not None:
            ordered_labels.append(label)
    return ordered_labels


def interactive_response_family(state: str) -> Optional[str]:
    compact_cfg = COMPACT_INTERACTIVE_STATE_MAP.get(state)
    if compact_cfg is not None:
        return compact_cfg["family"]
    return None


def learned_audio_vocab_path(store_path: Optional[str | Path] = None) -> Path:
    if store_path is None:
        return LEARNED_AUDIO_VOCAB_PATH
    return Path(store_path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_active_learned_vocab_entries(store_path: Optional[str | Path] = None) -> dict[str, list[dict[str, Any]]]:
    path = learned_audio_vocab_path(store_path)
    path_key = str(path.resolve()) if path.is_absolute() else str(path)

    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _LEARNED_AUDIO_VOCAB_CACHE.update({"path": path_key, "mtime": None, "entries_by_family": {}})
        return {}

    if (
        _LEARNED_AUDIO_VOCAB_CACHE.get("path") == path_key
        and _LEARNED_AUDIO_VOCAB_CACHE.get("mtime") == mtime
    ):
        return _LEARNED_AUDIO_VOCAB_CACHE.get("entries_by_family", {})

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        _LEARNED_AUDIO_VOCAB_CACHE.update({"path": path_key, "mtime": mtime, "entries_by_family": {}})
        return {}

    entries_by_family: dict[str, list[dict[str, Any]]] = {}
    for entry in payload.get("entries", []):
        if entry.get("status") != "active":
            continue
        family = entry.get("family")
        normalized_text = str(entry.get("normalized_text") or "").strip()
        active_label = entry.get("active_label")
        if not family or not normalized_text or not active_label:
            continue
        entries_by_family.setdefault(family, []).append(entry)

    _LEARNED_AUDIO_VOCAB_CACHE.update(
        {
            "path": path_key,
            "mtime": mtime,
            "entries_by_family": entries_by_family,
        }
    )
    return entries_by_family


def update_learned_vocab_store(
    observations: list[dict[str, Any]],
    *,
    store_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    path = learned_audio_vocab_path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {"version": 1, "updated_at": None, "entries": []}

    entries: list[dict[str, Any]] = payload.get("entries", [])
    entries_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        family = str(entry.get("family") or "").strip()
        normalized_text = str(entry.get("normalized_text") or "").strip()
        if family and normalized_text:
            entries_by_key[(family, normalized_text)] = entry

    processed = 0
    activated_now = 0
    skipped = 0

    for observation in observations:
        family = str(observation.get("family") or "").strip()
        normalized_text = str(observation.get("normalized_input_text") or "").strip()
        canonical_label = str(observation.get("canonical_label") or "").strip().upper()
        if not family or not normalized_text or not canonical_label or canonical_label == "REPEAT":
            skipped += 1
            continue

        processed += 1
        entry = entries_by_key.get((family, normalized_text))
        if entry is None:
            entry = {
                "family": family,
                "normalized_text": normalized_text,
                "label_counts": {},
                "total_observations": 0,
                "keyboard_supervised_observations": 0,
                "prompt_languages": [],
                "states": [],
                "example_raw_inputs": [],
                "source_test_ids": [],
                "status": "pending",
                "active_label": None,
                "activation_confidence": 0.0,
                "activation_reason": "insufficient_support",
                "first_seen": _utc_now_iso(),
                "last_seen": _utc_now_iso(),
            }
            entries.append(entry)
            entries_by_key[(family, normalized_text)] = entry

        was_active = entry.get("status") == "active"
        label_counts = entry.setdefault("label_counts", {})
        label_counts[canonical_label] = int(label_counts.get(canonical_label, 0)) + 1
        entry["total_observations"] = int(entry.get("total_observations", 0)) + 1

        if observation.get("source_kind") == "keyboard_supervised":
            entry["keyboard_supervised_observations"] = int(entry.get("keyboard_supervised_observations", 0)) + 1

        prompt_language = str(observation.get("prompt_language_used") or "").strip()
        if prompt_language and prompt_language not in entry["prompt_languages"]:
            entry["prompt_languages"].append(prompt_language)

        state = str(observation.get("state") or "").strip()
        if state and state not in entry["states"]:
            entry["states"].append(state)

        raw_input = str(observation.get("raw_input_text") or "").strip()
        if raw_input and raw_input not in entry["example_raw_inputs"] and len(entry["example_raw_inputs"]) < 5:
            entry["example_raw_inputs"].append(raw_input)

        source_test_id = str(observation.get("source_test_id") or "").strip()
        if source_test_id and source_test_id not in entry["source_test_ids"]:
            entry["source_test_ids"].append(source_test_id)

        entry["last_seen"] = _utc_now_iso()

        dominant_label, dominant_count = max(label_counts.items(), key=lambda item: int(item[1]))
        total_count = sum(int(value) for value in label_counts.values())
        dominant_share = float(dominant_count) / float(total_count) if total_count else 0.0
        keyboard_supervised_count = int(entry.get("keyboard_supervised_observations", 0))

        if dominant_label == "REPEAT":
            entry["status"] = "ignored"
            entry["active_label"] = None
            entry["activation_confidence"] = 0.0
            entry["activation_reason"] = "repeat_not_learned"
        elif keyboard_supervised_count >= 1 and len(label_counts) == 1:
            entry["status"] = "active"
            entry["active_label"] = dominant_label
            entry["activation_confidence"] = max(0.93, dominant_share)
            entry["activation_reason"] = "keyboard_supervised"
        elif dominant_count >= 2 and dominant_share >= 0.80:
            entry["status"] = "active"
            entry["active_label"] = dominant_label
            entry["activation_confidence"] = max(0.85, dominant_share)
            entry["activation_reason"] = "repeated_consensus"
        elif len(label_counts) > 1 and dominant_share < 0.80:
            entry["status"] = "pending_review"
            entry["active_label"] = None
            entry["activation_confidence"] = dominant_share
            entry["activation_reason"] = "conflicting_observations"
        else:
            entry["status"] = "pending"
            entry["active_label"] = None
            entry["activation_confidence"] = dominant_share
            entry["activation_reason"] = "insufficient_support"

        if not was_active and entry.get("status") == "active":
            activated_now += 1

    payload["updated_at"] = _utc_now_iso()
    payload["entries"] = sorted(
        entries,
        key=lambda entry: (
            str(entry.get("family") or ""),
            str(entry.get("normalized_text") or ""),
        ),
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    _LEARNED_AUDIO_VOCAB_CACHE.update({"path": None, "mtime": None, "entries_by_family": {}})

    active_entries = [entry for entry in payload["entries"] if entry.get("status") == "active"]
    pending_entries = [entry for entry in payload["entries"] if entry.get("status") == "pending"]
    pending_review_entries = [entry for entry in payload["entries"] if entry.get("status") == "pending_review"]

    return {
        "store_path": str(path),
        "processed_observations": processed,
        "skipped_observations": skipped,
        "entries_total": len(payload["entries"]),
        "entries_active": len(active_entries),
        "entries_pending": len(pending_entries),
        "entries_pending_review": len(pending_review_entries),
        "activated_now": activated_now,
        "updated_at": payload["updated_at"],
    }


def _learned_vocab_match(normalized_text: str, family: str) -> tuple[Optional[str], float, str]:
    if not normalized_text:
        return None, 0.0, ""

    active_entries = _load_active_learned_vocab_entries().get(family, [])
    if not active_entries:
        return None, 0.0, ""

    for entry in active_entries:
        candidate = str(entry.get("normalized_text") or "").strip()
        if normalized_text == candidate:
            confidence = float(entry.get("activation_confidence") or 0.90)
            return str(entry.get("active_label")), max(confidence, 0.90), "learned_exact"

    best_label = None
    best_score = 0.0
    best_method = ""
    for entry in active_entries:
        candidate = str(entry.get("normalized_text") or "").strip()
        if not candidate:
            continue
        if len(candidate.split()) >= 2 or len(normalized_text.split()) >= 2:
            phrase_score = _phrase_match_score(normalized_text, candidate)
            if phrase_score > best_score:
                best_score = phrase_score
                best_label = str(entry.get("active_label"))
                best_method = "learned_phrase"
        if len(candidate) >= 4:
            fuzzy_score = similarity(normalized_text, candidate)
            if fuzzy_score >= 0.92 and fuzzy_score > best_score:
                best_score = fuzzy_score
                best_label = str(entry.get("active_label"))
                best_method = "learned_fuzzy"

    if best_label is not None:
        return best_label, best_score, best_method
    return None, 0.0, ""


def infer_response_language(
    *,
    transcript: str,
    detected_language: Optional[str] = None,
    detected_language_probability: Optional[float] = None,
) -> Optional[str]:
    corrections = FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["normalization"]["common_asr_corrections"]
    normalized_text = normalize_text(transcript, corrections)

    for language_code, pattern in LANGUAGE_SCRIPT_PATTERNS.items():
        if pattern.search(transcript):
            return language_code

    if (
        detected_language in {"en", "hi"}
        and detected_language_probability is not None
        and float(detected_language_probability) >= 0.45
    ):
        return detected_language

    token_set = set(normalized_text.split())
    if token_set:
        scored_languages = {
            language_code: len(token_set & hints)
            for language_code, hints in ROMANIZED_LANGUAGE_HINTS.items()
        }
        best_language = max(scored_languages, key=scored_languages.get)
        if scored_languages[best_language] > 0:
            return best_language

    if detected_language in {"en", "hi"}:
        return detected_language

    return None


def match_language_choice(
    *,
    transcript: str,
    detected_language: Optional[str] = None,
    detected_language_probability: Optional[float] = None,
) -> LanguageChoiceResult:
    corrections = FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["normalization"]["common_asr_corrections"]
    normalized_text = normalize_text(transcript, corrections)
    reprompt_text = localized_language_selection_prompt(retry=True)

    if not normalized_text:
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code=None,
            confidence=0.0,
            method="fallback",
            accepted=False,
            reprompt_text=reprompt_text,
            reason="empty_transcript",
        )

    tokens = normalized_text.split()
    english_direct_tokens = {"english", "inglish", "englis", "angrezi", "angrejee", "अंग्रेजी", "इंग्लिश"}
    hindi_direct_tokens = {
        "hindi", "hindhi", "hindee", "hindii", "hindy", "hndi", "indi", "indy",
        "hi", "hii", "hei", "hey", "hai", "हिंदी", "हिन्दी",
    }

    if any(token in english_direct_tokens or token.startswith("engli") or token.startswith("angre") for token in tokens):
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code="en",
            confidence=0.96,
            method="keyword",
            accepted=True,
            reprompt_text=reprompt_text,
        )

    if any(token in hindi_direct_tokens or token.startswith("hind") for token in tokens):
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code="hi",
            confidence=0.96,
            method="keyword",
            accepted=True,
            reprompt_text=reprompt_text,
        )

    for language_code, cfg in LANGUAGE_SELECTION_ALIASES.items():
        for item in cfg.get("exact", []):
            if normalized_text == normalize_text(item, corrections):
                return LanguageChoiceResult(
                    transcript=transcript,
                    normalized_text=normalized_text,
                    language_code=language_code,
                    confidence=1.0,
                    method="exact",
                    accepted=True,
                    reprompt_text=reprompt_text,
                )

    best_label = None
    best_phrase_score = 0.0
    for language_code, cfg in LANGUAGE_SELECTION_ALIASES.items():
        for item in cfg.get("phrases", []):
            score = _phrase_match_score(normalized_text, normalize_text(item, corrections))
            if score > best_phrase_score:
                best_phrase_score = score
                best_label = language_code
    if best_label is not None and best_phrase_score > 0.0:
        confidence = 0.95 if best_phrase_score >= 0.95 else best_phrase_score
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code=best_label,
            confidence=confidence,
            method="phrase",
            accepted=confidence >= 0.78,
            reprompt_text=reprompt_text,
                    reason="" if confidence >= 0.78 else "low_confidence",
        )

    if (
        detected_language == "hi"
        and detected_language_probability is not None
        and float(detected_language_probability) >= 0.30
    ):
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code="hi",
            confidence=max(0.8, float(detected_language_probability)),
            method="detected_language_bias",
            accepted=True,
            reprompt_text=reprompt_text,
        )

    best_label = None
    best_score = 0.0
    for language_code, cfg in LANGUAGE_SELECTION_ALIASES.items():
        for item in cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", []):
            score = similarity(normalized_text, normalize_text(item, corrections))
            if score > best_score:
                best_score = score
                best_label = language_code
    fuzzy_threshold = 0.88 if best_label == "en" else 0.72
    if best_label is not None and best_score >= fuzzy_threshold:
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code=best_label,
            confidence=best_score,
            method="fuzzy",
            accepted=True,
            reprompt_text=reprompt_text,
        )

    inferred_language = infer_response_language(
        transcript=transcript,
        detected_language=detected_language,
        detected_language_probability=detected_language_probability,
    )
    if inferred_language == "hi":
        return LanguageChoiceResult(
            transcript=transcript,
            normalized_text=normalized_text,
            language_code="hi",
            confidence=0.8,
            method="language_inference",
            accepted=True,
            reprompt_text=reprompt_text,
        )

    return LanguageChoiceResult(
        transcript=transcript,
        normalized_text=normalized_text,
        language_code=None,
        confidence=best_score,
        method="fallback",
        accepted=False,
        reprompt_text=reprompt_text,
        reason="low_confidence",
    )


def _context_rule_result(normalized_text: str, response_type: str) -> tuple[bool, str]:
    context_rules = FSM_AUDIO_RESPONSE_LIBRARY_V1["context_rules"]

    yes_terms = {
        "yes", "yeah", "yep", "yup", "ok", "okay",
        "haan", "haanji", "हाँ",
    }
    good_terms = {
        "good", "fine", "perfect", "comfortable",
        "theek", "thik", "ठीक", "sahi", "सही",
    }
    better_terms = {"better", "बेहतर", "innum better", "konjam adjust"}

    if normalized_text in yes_terms:
        if response_type in set(context_rules["yes_like_terms"]["disallowed_in"]) | {"duochrome_4way", "distance_bino_4way"}:
            return False, "context_disallowed"
    if normalized_text in good_terms:
        if response_type in set(context_rules["good_like_terms"]["disallowed_in"]) | {"duochrome_4way", "distance_bino_4way"}:
            return False, "context_disallowed"
    if normalized_text in better_terms:
        rule = context_rules["better_like_terms"]["allowed_in"].get(response_type)
        if rule == "needs_disambiguation":
            return False, "needs_disambiguation"

    return True, ""


def _state_match_config(state: str) -> tuple[dict, dict]:
    state_map = STATE_CONFIG_MAP.get(state)
    if state_map is None:
        raise KeyError(f"No audio response config for state: {state}")
    state_cfg = FSM_AUDIO_RESPONSE_LIBRARY_V1["states"][state_map["library_key"]]
    return state_map, state_cfg


def _compact_state_match_config(state: str) -> Optional[dict]:
    return COMPACT_INTERACTIVE_STATE_MAP.get(state)


def _extract_chart_letter_tokens(normalized_text: str) -> list[str]:
    allowed_letters = set(CHART_LETTER_ALIASES.keys())
    extracted: list[str] = []

    for token in normalized_text.split():
        cleaned = token.strip().upper()
        if not cleaned:
            continue
        if cleaned in allowed_letters:
            extracted.append(cleaned)
            continue
        if cleaned.isalpha() and len(cleaned) > 1 and set(cleaned).issubset(allowed_letters):
            extracted.extend(list(cleaned))
            continue

        matched = False
        lower_token = token.lower()
        for letter, aliases in CHART_LETTER_ALIASES.items():
            if lower_token in aliases:
                extracted.append(letter)
                matched = True
                break
        if matched:
            continue

    return extracted


def _lcs_length(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a or not seq_b:
        return 0

    dp = [[0] * (len(seq_b) + 1) for _ in range(len(seq_a) + 1)]
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _match_chart_letters(
    *,
    transcript: str,
    normalized_text: str,
    stimulus_letters: Optional[str],
) -> tuple[Optional[str], float, str]:
    if not stimulus_letters:
        return None, 0.0, ""

    target_letters: list[str] = []
    for token in str(stimulus_letters).upper().split():
        cleaned = "".join(ch for ch in token if ch.isalpha())
        if not cleaned:
            continue
        target_letters.extend(list(cleaned))
    spoken_letters = _extract_chart_letter_tokens(normalized_text)

    if len(target_letters) < 2 or len(spoken_letters) < 2:
        return None, 0.0, ""

    accuracy = _lcs_length(spoken_letters, target_letters) / float(len(target_letters))
    if accuracy >= 0.8:
        return "CLEAR", accuracy, "letter_reading"
    if accuracy >= 0.3:
        return "BLURRY", accuracy, "letter_reading_partial_blurry"
    return "REPEAT", accuracy, "letter_reading_partial_repeat"


def _compact_exact_phrase_fuzzy_match(
    *,
    normalized_text: str,
    family: str,
    corrections: dict[str, str],
) -> tuple[Optional[str], float, str]:
    alias_family = family
    if family in {"coarse_clarity", "near_clarity", "near_bino_3way"}:
        alias_family = "clarity_3way"
    family_cfg = COMPACT_MATCH_ALIASES[alias_family]

    for label, cfg in family_cfg.items():
        for item in cfg.get("exact", []):
            if normalized_text == normalize_text(item, corrections):
                return label, 1.0, "exact"

    best_phrase_label = None
    best_phrase_score = 0.0
    for label, cfg in family_cfg.items():
        for item in cfg.get("phrases", []):
            score = _phrase_match_score(normalized_text, normalize_text(item, corrections))
            if score > best_phrase_score:
                best_phrase_score = score
                best_phrase_label = label
    if best_phrase_label is not None and best_phrase_score > 0.0:
        return best_phrase_label, 0.95 if best_phrase_score >= 0.95 else best_phrase_score, "phrase"

    best_label = None
    best_score = 0.0
    for label, cfg in family_cfg.items():
        for item in cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", []):
            score = similarity(normalized_text, normalize_text(item, corrections))
            if score > best_score:
                best_score = score
                best_label = label
    if best_label is not None and best_score >= 0.78:
        return best_label, best_score, "fuzzy"

    return None, best_score, ""


def _comparison_token_label(normalized_text: str) -> Optional[str]:
    if _is_ambiguous_bare_option_text(normalized_text):
        return None
    tokens = set(normalized_text.split())
    one_terms = {"one", "first", "ek", "एक", "pehla", "पहला"}
    two_terms = {"two", "second", "do", "to", "too", "tu", "दो", "dusra", "दूसरा"}
    same_terms = {"same", "equal", "similar", "barabar", "बराबर"}

    has_one = bool(tokens & one_terms)
    has_two = bool(tokens & two_terms)
    has_same = bool(tokens & same_terms)

    if has_one and not has_two:
        return "BETTER_1"
    if has_two and not has_one:
        return "BETTER_2"
    if has_same and not has_one and not has_two:
        return "SAME"
    return None


def _compact_comparison_token_label(normalized_text: str) -> Optional[str]:
    if _is_ambiguous_bare_option_text(normalized_text):
        return None
    tokens = set(normalized_text.split())
    one_terms = {"one", "first", "ek", "एक", "pehla", "पहला"}
    two_terms = {"two", "second", "do", "to", "too", "tu", "दो", "dusra", "दूसरा"}
    same_terms = {"same", "equal", "similar", "both", "dono", "दोनों", "barabar", "बराबर"}
    repeat_terms = {"repeat", "again", "dubara", "दोबारा"}

    if tokens & repeat_terms:
        return "REPEAT"
    if tokens & same_terms and not (tokens & one_terms) and not (tokens & two_terms):
        return "SAME"
    if tokens & one_terms and not (tokens & two_terms):
        return "ONE"
    if tokens & two_terms and not (tokens & one_terms):
        return "TWO"
    return None


def _compact_comparison_intent_match(normalized_text: str) -> Optional[str]:
    if _is_ambiguous_bare_option_text(normalized_text):
        return None
    if _compact_repeat_intent(normalized_text):
        return "REPEAT"

    # Check for negated comparison intent (e.g., "not same" → CANT_TELL)
    negated = _detect_negated_intent(normalized_text, _COMPARISON_NEGATION_MAP)
    if negated is not None:
        return negated

    same_patterns = [
        r"\babout the same\b",
        r"\bboth (look )?same\b",
        r"\bthey( are|'re)? (about )?the same\b",
        r"\bno difference\b",
        r"\bequal\b",
        r"\bsame\b",
    ]
    better_one_patterns = [
        r"\b(first|one|ek|एक)\b.*\b(better|clearer|sharper|readable)\b",
        r"\b(better|clearer|sharper|readable)\b.*\b(first|one|ek|एक)\b",
    ]
    better_two_patterns = [
        r"\b(second|two|do|दो)\b.*\b(better|clearer|sharper|readable)\b",
        r"\b(better|clearer|sharper|readable)\b.*\b(second|two|do|दो)\b",
    ]

    if any(re.search(pattern, normalized_text) for pattern in same_patterns):
        return "SAME"
    if any(re.search(pattern, normalized_text) for pattern in better_two_patterns):
        return "TWO"
    if any(re.search(pattern, normalized_text) for pattern in better_one_patterns):
        return "ONE"
    return None


def _compact_direction_token_label(normalized_text: str, *, family: str) -> Optional[str]:
    tokens = set(normalized_text.split())
    same_terms = {"same", "equal", "similar", "both", "dono", "दोनों", "barabar", "बराबर"}
    repeat_terms = {"repeat", "again", "dubara", "दोबारा"}
    if tokens & repeat_terms:
        return "REPEAT"
    if tokens & same_terms:
        return "SAME"

    if family == "duochrome_4way":
        if tokens & {"red", "lal", "लाल"}:
            return "RED"
        if tokens & {"green", "hara", "हरा"}:
            return "GREEN"
    if family == "distance_bino_4way":
        if tokens & {"top", "upper", "upar", "ऊपर"}:
            return "TOP"
        if tokens & {"bottom", "lower", "neeche", "नीचे"}:
            return "BOTTOM"
    return None


_NEGATION_PREFIXES = re.compile(
    r"\b(?:not|no|nahi|nhi|nahin|isn'?t|don'?t|doesn'?t|cant|can'?t|cannot|without|hardly|barely)\b"
)

# Negation flips: if "not X" detected, return the opposite intent
_CLARITY_NEGATION_MAP = {
    "clear": "BLURRY", "sharp": "BLURRY", "readable": "BLURRY", "visible": "BLURRY",
    "good": "BLURRY", "fine": "BLURRY", "proper": "BLURRY", "crisp": "BLURRY",
    "blurry": "CLEAR", "blur": "CLEAR", "blurred": "CLEAR", "hazy": "CLEAR",
    "fuzzy": "CLEAR", "unclear": "CLEAR",
    "saaf": "BLURRY", "theek": "BLURRY", "sahi": "BLURRY",
    "dhundla": "CLEAR", "dhundhla": "CLEAR",
}

_COMPARISON_NEGATION_MAP = {
    "same": "CANT_TELL", "equal": "CANT_TELL", "similar": "CANT_TELL",
    "different": "SAME",
}


def _detect_negated_intent(normalized_text: str, intent_map: dict) -> Optional[str]:
    """If the text contains 'not <keyword>', return the flipped intent."""
    if not _NEGATION_PREFIXES.search(normalized_text):
        return None
    tokens = normalized_text.split()
    for i, tok in enumerate(tokens):
        if _NEGATION_PREFIXES.fullmatch(tok):
            # Check the next 1-2 tokens for a keyword
            for j in range(1, min(3, len(tokens) - i)):
                candidate = tokens[i + j]
                if candidate in intent_map:
                    return intent_map[candidate]
    return None


def _compact_clarity_intent_match(normalized_text: str) -> Optional[str]:
    # Check for negated intent first (e.g., "not blurry" → CLEAR, "not clear" → BLURRY)
    negated = _detect_negated_intent(normalized_text, _CLARITY_NEGATION_MAP)
    if negated is not None:
        return negated

    tokens = normalized_text.split()
    token_set = set(tokens)
    short_utterance = len(tokens) <= 2

    blurry_patterns = [
        r"\bblurr?y?\b",
        r"\bbloody\b",
        r"\bbluddy\b",
        r"\bbladi\b",
        r"\bbilari\b",
        r"\bbilari\b",
        r"\bblari\b",
        r"\bunclear\b",
        r"\bhazy\b",
        r"\bfuzzy\b",
        r"\bnot\s+(?:that\s+|very\s+|so\s+|fully\s+)?clear\b",
        r"\bnot\s+(?:that\s+|very\s+)?sharp\b",
        r"\bnot\s+(?:that\s+|very\s+)?readable\b",
        r"\bnot\s+(?:that\s+|fully\s+)?visible\b",
        r"\bnot\s+good\b",
        r"\bnot\s+proper\b",
        r"\bhard to read\b",
        r"\bdifficult to read\b",
        r"\bnahi\s+clear\b",
        r"\bclear nahi\b",
        r"धुंध",
        r"ब्लर",
        r"मंग",
        r"மங்க",
        r"தெளிவா இல்ல",
    ]
    clear_patterns = [
        r"\bclear\b",
        r"\breadable\b",
        r"\bvisible\b",
        r"\bsharp\b",
        r"\bcrisp\b",
        r"\bi can read\b",
        r"\bi can see\b",
        r"\ball clear\b",
        r"\blooks clear\b",
        r"\bvery clear\b",
        r"\bfully clear\b",
        r"साफ",
        r"ठीक",
        r"सही",
        r"पढ़",
        r"पढ",
        r"पड",
        r"dikh",
        r"dekh",
        r"padh",
        r"saaf",
        r"theriy",
    ]
    repeat_patterns = [
        r"\brepeat\b",
        r"\bagain\b",
        r"\bsay again\b",
        r"\bplease repeat\b",
        r"\bnot readable\b",
        r"\bcannot read\b",
        r"\bcan't read\b",
        r"\bcannot see\b",
        r"\bcan't see\b",
        r"\bnothing visible\b",
        r"\bnothing readable\b",
        r"\bnahi dikh\b",
        r"\bnahi padh\b",
        r"\bसमझ नहीं आ रहा\b",
        r"\bकुछ नहीं दिख\b",
        r"\bபடிக்க முடியல\b",
    ]

    comparative_blurry_patterns = [
        r"\bmore blurry\b",
        r"\bgot more blurry\b",
        r"\bbecame more blurry\b",
        r"\bblurrier\b",
        r"\bgot worse\b",
        r"\bbecame worse\b",
        r"\bworse now\b",
        r"\bnot better\b",
        r"\bdid not get better\b",
        r"\bdidn't get better\b",
        r"\bno better\b",
        r"\bnot improved\b",
    ]
    comparative_clear_patterns = [
        r"\bclearer\b",
        r"\bgot clearer\b",
        r"\bbecame clearer\b",
        r"\bgot clear\b",
        r"\bbecame clear\b",
        r"\bmore clear\b",
        r"\bless blurry\b",
        r"\bbetter now\b",
        r"\bbetter than before\b",
        r"\bgot better\b",
        r"\bbecame better\b",
        r"\bimproved\b",
    ]

    if short_utterance and token_set & {"yes", "haan", "haanji", "हाँ"}:
        return "CLEAR"
    if short_utterance and token_set & {"no", "nahin", "nahi", "नहीं", "nope"}:
        return "BLURRY"

    if re.search(r"(हाँ|हां|haan)\s+बेहतर", normalized_text):
        return "CLEAR"
    if re.search(r"(नहीं|nahin|nahi)\s+बेहतर", normalized_text):
        return "BLURRY"
    if "क्लियर" in normalized_text:
        return "CLEAR"

    if any(re.search(pattern, normalized_text) for pattern in comparative_blurry_patterns):
        return "BLURRY"

    if any(re.search(pattern, normalized_text) for pattern in comparative_clear_patterns):
        return "CLEAR"

    if any(re.search(pattern, normalized_text) for pattern in blurry_patterns):
        return "BLURRY"

    if any(re.search(pattern, normalized_text) for pattern in repeat_patterns):
        return "REPEAT"

    if short_utterance and token_set & {"here", "hear", "cheers", "cheer"}:
        return "CLEAR"

    if any(re.search(pattern, normalized_text) for pattern in clear_patterns):
        return "CLEAR"

    return None


def _line_phase_clarity_override(*, transcript: str, normalized_text: str) -> Optional[str]:
    combined = f"{normalized_text} {str(transcript or '').lower()}".strip()
    if not combined:
        return None

    clear_cues = (
        " clear",
        " readable",
        " visible",
        " sharp",
        " crisp",
        " saaf",
        " saf",
        "साफ",
        "ठीक",
        "सही",
    )
    blurry_cues = (
        " blurry",
        " blur",
        " fuzzy",
        " hazy",
        " unclear",
        " dhund",
        " dhundla",
        " dhundhla",
        "धुंध",
        "धुंद",
        "धुँध",
        "ब्लर",
        "thula",
        "hulahula",
        " not clear",
    )
    repeat_cues = (
        " repeat",
        " again",
        " fir se",
        " फिर से",
        " फिरसे",
        " दोबारा",
        " dobara",
    )

    has_clear = any(cue in combined for cue in clear_cues)
    has_blurry = any(cue in combined for cue in blurry_cues)
    has_repeat = any(cue in combined for cue in repeat_cues)

    if has_blurry and has_clear:
        return "BLURRY"
    if has_repeat:
        return "REPEAT"
    if has_blurry:
        return "BLURRY"
    return None


def _is_ambiguous_bare_option_text(normalized_text: str) -> bool:
    tokens = [token for token in normalized_text.split() if token]
    if not tokens:
        return False
    allowed = {"option", "options", "vikalp", "विकल्प", "hai", "hey", "is", "the"}
    return set(tokens).issubset(allowed) and any(token in {"option", "options", "vikalp", "विकल्प"} for token in tokens)


def _comparison_intent_match(normalized_text: str) -> Optional[str]:
    if _is_ambiguous_bare_option_text(normalized_text):
        return None
    cant_tell_patterns = [
        r"\b(can'?t tell|cannot tell|cant tell)\b",
        r"\bhard to tell\b",
        r"\bhard two tell\b",
        r"\btoo close to tell\b",
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\bno idea\b",
        r"\bdifficult to say\b",
    ]
    same_patterns = [
        r"\babout the same\b",
        r"\bboth (look )?same\b",
        r"\bthey( are|'re)? (about )?the same\b",
        r"\bno difference\b",
        r"\bequal\b",
        r"\bsame\b",
    ]
    better_one_patterns = [
        r"\b(first|one)\b.*\b(better|clearer|sharper|readable)\b",
        r"\b(better|clearer|sharper|readable)\b.*\b(first|one)\b",
    ]
    better_two_patterns = [
        r"\b(second|two|too|to|tu|do)\b.*\b(better|clearer|sharper|readable)\b",
        r"\b(better|clearer|sharper|readable)\b.*\b(second|two|too|to|tu|do)\b",
    ]

    if any(re.search(pattern, normalized_text) for pattern in cant_tell_patterns):
        return "CANT_TELL"
    if any(re.search(pattern, normalized_text) for pattern in same_patterns):
        return "SAME"
    if any(re.search(pattern, normalized_text) for pattern in better_two_patterns):
        return "BETTER_2"
    if any(re.search(pattern, normalized_text) for pattern in better_one_patterns):
        return "BETTER_1"
    return None


def _compact_repeat_intent(normalized_text: str) -> bool:
    repeat_patterns = [
        r"\brepeat\b",
        r"\bagain\b",
        r"\bsay again\b",
        r"\bplease repeat\b",
        r"\brepeat please\b",
        r"\bफिर से\b",
        r"\bदोबारा\b",
        r"\bcan't tell\b",
        r"\bcannot tell\b",
        r"\bhard to tell\b",
        r"\bnot sure\b",
    ]
    return any(re.search(pattern, normalized_text) for pattern in repeat_patterns)


def _multilingual_alias_match(
    normalized_text: str,
    response_type: str,
    corrections: dict[str, str],
) -> tuple[Optional[str], float, str]:
    alias_cfg = MULTILINGUAL_RESPONSE_ALIASES.get(response_type, {})

    for label, cfg in alias_cfg.items():
        for item in cfg.get("exact", []):
            if normalized_text == normalize_text(item, corrections):
                return label, 1.0, "multilingual_exact"

    best_label = None
    best_score = 0.0
    for label, cfg in alias_cfg.items():
        for item in cfg.get("phrases", []):
            normalized_item = normalize_text(item, corrections)
            score = _phrase_match_score(normalized_text, normalized_item)
            if score > best_score:
                best_score = score
                best_label = label

    if best_label is not None and best_score > 0.0:
        return best_label, 0.96 if best_score >= 0.95 else best_score, "multilingual_phrase"

    return None, 0.0, ""


def _multilingual_intent_match(
    normalized_text: str,
    response_type: str,
) -> tuple[Optional[str], float, str]:
    if response_type not in {"clarity_3way", "near_clarity_3way", "near_final_3way"}:
        return None, 0.0, ""

    not_visible_patterns = [
        r"(नहीं|नही|मत|not|cannot|can't).*(दिख|देख|पढ़|पढ|read|see|visible)",
        r"(दिख|देख|पढ़|पढ).*(नहीं|नही|not)",
        r"(nahi|nhi).*(dikh|dekh|padh|padh)",
        r"(dikh|dekh|padh).*(nahi|nhi)",
        r"(தெரிய|படிக்க).*(இல்ல|முடியல|ல்ல)",
        r"(not visible|not readable|cannot read|cannot see)",
    ]
    blurry_patterns = [
        r"धुंध",
        r"ब्लर",
        r"\bblur",
        r"\bblurr",
        r"मंग",
        r"fuzzy",
        r"hazy",
        r"தெளிவா இல்ல",
        r"மங்க",
    ]
    clear_patterns = [
        r"साफ",
        r"ठीक",
        r"सही",
        r"दिख",
        r"पढ़",
        r"पढ",
        r"पड",
        r"^ह($| )",
        r"saaf",
        r"saf",
        r"dikh",
        r"dekh",
        r"padh",
        r"padh",
        r"visible",
        r"readable",
        r"தெளி",
    ]

    if any(re.search(pattern, normalized_text) for pattern in not_visible_patterns):
        return ("NOT_VISIBLE" if response_type != "near_final_3way" else "NOT_COMFORTABLE"), 0.82, "multilingual_intent"
    if any(re.search(pattern, normalized_text) for pattern in blurry_patterns):
        return ("BLURRY" if response_type != "near_final_3way" else "NEED_ADJUSTMENT"), 0.82, "multilingual_intent"
    if any(re.search(pattern, normalized_text) for pattern in clear_patterns):
        return ("CLEAR" if response_type != "near_final_3way" else "TARGET_OK"), 0.82, "multilingual_intent"

    return None, 0.0, ""


def _best_multilingual_fuzzy_match(
    normalized_text: str,
    response_type: str,
    corrections: dict[str, str],
) -> tuple[Optional[str], float, str]:
    alias_cfg = MULTILINGUAL_RESPONSE_ALIASES.get(response_type, {})
    best_label = None
    best_score = 0.0

    for label, cfg in alias_cfg.items():
        candidates = cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", [])
        for item in candidates:
            normalized_item = normalize_text(item, corrections)
            score = similarity(normalized_text, normalized_item)
            if score > best_score:
                best_score = score
                best_label = label

    if best_label is not None and best_score >= 0.68:
        return best_label, best_score, "multilingual_fuzzy"

    return None, 0.0, ""


def _best_multilingual_semantic_match(
    normalized_text: str,
    response_type: str,
    corrections: dict[str, str],
) -> tuple[Optional[str], float, str]:
    alias_cfg = MULTILINGUAL_RESPONSE_ALIASES.get(response_type, {})
    best_label = None
    best_score = 0.0

    for label, cfg in alias_cfg.items():
        candidates = cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", [])
        for item in candidates:
            normalized_item = normalize_text(item, corrections)
            score = _semantic_similarity(normalized_text, normalized_item)
            if score > best_score:
                best_score = score
                best_label = label

    if best_label is not None and best_score >= 0.72:
        return best_label, best_score, "multilingual_semantic"

    return None, 0.0, ""


def match_response(
    *,
    transcript: str,
    state: str,
    available_options: list[str],
    stimulus_letters: Optional[str] = None,
    question: Optional[str] = None,
    language: Optional[str] = None,
) -> MatchResult:
    compact_cfg = _compact_state_match_config(state)
    if compact_cfg is not None:
        response_type = compact_cfg["response_type"]
        family = compact_cfg["family"]
        corrections = FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["normalization"]["common_asr_corrections"]
        normalized_text = normalize_text(transcript, corrections)
        reprompt_text = _reprompt_text(response_type)

        def build_compact_result(label: str, confidence: float, method: str) -> MatchResult:
            available_set = set(available_options)
            mapped_response_value = compact_cfg["label_to_response"].get(label)

            if label in available_set:
                response_value = label
            elif mapped_response_value in available_set:
                response_value = mapped_response_value
            elif label == "REPEAT":
                response_value = "REPEAT" if "REPEAT" in available_set else "__REPEAT__"
            else:
                response_value = mapped_response_value

            accepted = response_value in available_set or response_value in {"__REPEAT__", "REPEAT"}
            return MatchResult(
                transcript=transcript,
                normalized_text=normalized_text,
                response_type=response_type,
                canonical_label=label,
                response_value=response_value if accepted else None,
                confidence=confidence,
                method=method,
                reprompt_text=reprompt_text,
                accepted=accepted,
                reason="" if accepted else "option_not_available",
            )

        if not normalized_text:
            return MatchResult(
                transcript=transcript,
                normalized_text=normalized_text,
                response_type=response_type,
                canonical_label=None,
                response_value=None,
                confidence=0.0,
                method="fallback",
                reprompt_text=reprompt_text,
                accepted=False,
                reason="empty_transcript",
            )

        context_ok, context_reason = _context_rule_result(normalized_text, response_type)
        if not context_ok:
            return MatchResult(
                transcript=transcript,
                normalized_text=normalized_text,
                response_type=response_type,
                canonical_label=None,
                response_value=None,
                confidence=0.0,
                method="context_filter",
                reprompt_text=reprompt_text,
                accepted=False,
                reason=context_reason,
            )

        if family == "comparison_4way":
            intent_label = _compact_comparison_intent_match(normalized_text)
            if intent_label is not None:
                return build_compact_result(intent_label, 0.96 if intent_label != "REPEAT" else 0.95, "comparison_intent")
            token_label = _compact_comparison_token_label(normalized_text)
            if token_label is not None:
                return build_compact_result(token_label, 0.93, "token_intent")
        elif family in {"duochrome_4way", "distance_bino_4way"}:
            if _compact_repeat_intent(normalized_text):
                return build_compact_result("REPEAT", 0.95, "repeat_intent")
            token_label = _compact_direction_token_label(normalized_text, family=family)
            if token_label is not None:
                return build_compact_result(token_label, 0.93, "token_intent")
        elif family in {"coarse_clarity", "near_clarity", "near_bino_3way"}:
            if _is_line_clarity_phase(state=state, question=question):
                override_label = _line_phase_clarity_override(
                    transcript=transcript,
                    normalized_text=normalized_text,
                )
                if override_label is not None:
                    confidence = 0.96 if override_label != "REPEAT" else 0.95
                    return build_compact_result(override_label, confidence, "line_phase_clarity_override")
            clarity_label = _compact_clarity_intent_match(normalized_text)
            if clarity_label is not None:
                confidence = 0.96 if clarity_label != "REPEAT" else 0.95
                return build_compact_result(clarity_label, confidence, "clarity_intent")
            if _question_expects_line_reading(state=state, question=question, language=language):
                letter_label, letter_confidence, letter_method = _match_chart_letters(
                    transcript=transcript,
                    normalized_text=normalized_text,
                    stimulus_letters=stimulus_letters,
                )
                if letter_label is not None:
                    return build_compact_result(letter_label, letter_confidence, letter_method)

        matched_label, matched_confidence, matched_method = _compact_exact_phrase_fuzzy_match(
            normalized_text=normalized_text,
            family=family,
            corrections=corrections,
        )
        if matched_label is not None:
            return build_compact_result(matched_label, matched_confidence, matched_method)

        if state in COARSE_RESPONSE_STATES and _question_expects_line_reading(state=state, question=question, language=language):
            letter_label, letter_confidence, letter_method = _match_chart_letters(
                transcript=transcript,
                normalized_text=normalized_text,
                stimulus_letters=stimulus_letters,
            )
            if letter_label is not None:
                return build_compact_result(letter_label, letter_confidence, letter_method)

        best_score = 0.0
        alias_family = family
        if family in {"coarse_clarity", "near_clarity", "near_bino_3way"}:
            alias_family = "clarity_3way"
        for label, cfg in COMPACT_MATCH_ALIASES[alias_family].items():
            for item in cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", []):
                best_score = max(best_score, _semantic_similarity(normalized_text, normalize_text(item, corrections)))

        learned_label, learned_confidence, learned_method = _learned_vocab_match(normalized_text, family)
        if learned_label is not None:
            return build_compact_result(learned_label, learned_confidence, learned_method)

        return MatchResult(
            transcript=transcript,
            normalized_text=normalized_text,
            response_type=response_type,
            canonical_label=None,
            response_value=None,
            confidence=best_score,
            method="fallback",
            reprompt_text=reprompt_text,
            accepted=False,
            reason="low_confidence",
        )

    state_map, state_cfg = _state_match_config(state)
    response_type = state_cfg["response_type"]
    thresholds = FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["thresholds"]
    corrections = FSM_AUDIO_RESPONSE_LIBRARY_V1["matching_pipeline"]["normalization"]["common_asr_corrections"]

    normalized_text = normalize_text(transcript, corrections)
    reprompt_text = _reprompt_text(response_type)

    if not normalized_text:
        return MatchResult(
            transcript=transcript,
            normalized_text=normalized_text,
            response_type=response_type,
            canonical_label=None,
            response_value=None,
            confidence=0.0,
            method="fallback",
            reprompt_text=reprompt_text,
            accepted=False,
            reason="empty_transcript",
        )

    context_ok, context_reason = _context_rule_result(normalized_text, response_type)
    if not context_ok:
        return MatchResult(
            transcript=transcript,
            normalized_text=normalized_text,
            response_type=response_type,
            canonical_label=None,
            response_value=None,
            confidence=0.0,
            method="context_filter",
            reprompt_text=reprompt_text,
            accepted=False,
            reason=context_reason,
        )

    if response_type == "comparison_4way":
        comparison_intent_label = _comparison_intent_match(normalized_text)
        if comparison_intent_label is not None:
            response_value = state_map["label_to_response"].get(comparison_intent_label)
            if response_value in set(available_options):
                return MatchResult(
                    transcript=transcript,
                    normalized_text=normalized_text,
                    response_type=response_type,
                    canonical_label=comparison_intent_label,
                    response_value=response_value,
                    confidence=0.96,
                    method="comparison_intent",
                    reprompt_text=reprompt_text,
                    accepted=True,
                    reason="",
                )
        comparison_label = _comparison_token_label(normalized_text)
        if comparison_label is not None:
            response_value = state_map["label_to_response"].get(comparison_label)
            if response_value in set(available_options):
                return MatchResult(
                    transcript=transcript,
                    normalized_text=normalized_text,
                    response_type=response_type,
                    canonical_label=comparison_label,
                    response_value=response_value,
                    confidence=0.93,
                    method="token_intent",
                    reprompt_text=reprompt_text,
                    accepted=True,
                    reason="",
                )

    def build_result(label: str, confidence: float, method: str) -> MatchResult:
        response_value = state_map["label_to_response"].get(label)
        accepted = response_value in set(available_options)
        return MatchResult(
            transcript=transcript,
            normalized_text=normalized_text,
            response_type=response_type,
            canonical_label=label,
            response_value=response_value if accepted else None,
            confidence=confidence,
            method=method,
            reprompt_text=reprompt_text,
            accepted=accepted and confidence >= thresholds["fallback_reprompt_below"],
            reason="" if accepted else "option_not_available",
        )

    for label, cfg in state_cfg["labels"].items():
        for item in cfg.get("exact", []) + cfg.get("context_only", []):
            if normalized_text == normalize_text(item, corrections):
                return build_result(label, thresholds["exact_match_confidence"], "exact")

    best_phrase_label = None
    best_phrase_score = 0.0
    for label, cfg in state_cfg["labels"].items():
        for item in cfg.get("phrases", []):
            normalized_item = normalize_text(item, corrections)
            score = _phrase_match_score(normalized_text, normalized_item)
            if score > best_phrase_score:
                best_phrase_score = score
                best_phrase_label = label
    if best_phrase_label is not None and best_phrase_score > 0.0:
        confidence = thresholds["phrase_match_confidence"] if best_phrase_score >= 0.95 else best_phrase_score
        return build_result(best_phrase_label, confidence, "phrase")

    multilingual_label, multilingual_confidence, multilingual_method = _multilingual_alias_match(
        normalized_text,
        response_type,
        corrections,
    )
    if multilingual_label is not None:
        return build_result(multilingual_label, multilingual_confidence, multilingual_method)

    multilingual_intent_label, multilingual_intent_confidence, multilingual_intent_method = _multilingual_intent_match(
        normalized_text,
        response_type,
    )
    if multilingual_intent_label is not None:
        return build_result(
            multilingual_intent_label,
            multilingual_intent_confidence,
            multilingual_intent_method,
        )

    multilingual_fuzzy_label, multilingual_fuzzy_confidence, multilingual_fuzzy_method = _best_multilingual_fuzzy_match(
        normalized_text,
        response_type,
        corrections,
    )
    if multilingual_fuzzy_label is not None:
        return build_result(
            multilingual_fuzzy_label,
            multilingual_fuzzy_confidence,
            multilingual_fuzzy_method,
        )

    multilingual_semantic_label, multilingual_semantic_confidence, multilingual_semantic_method = _best_multilingual_semantic_match(
        normalized_text,
        response_type,
        corrections,
    )
    if multilingual_semantic_label is not None:
        return build_result(
            multilingual_semantic_label,
            multilingual_semantic_confidence,
            multilingual_semantic_method,
        )

    best_label = None
    best_score = 0.0
    for label, cfg in state_cfg["labels"].items():
        candidates = cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", [])
        for item in candidates:
            score = similarity(normalized_text, normalize_text(item, corrections))
            if score > best_score:
                best_score = score
                best_label = label
    if best_label is not None and best_score >= thresholds["fuzzy_match_confidence"]:
        return build_result(best_label, best_score, "fuzzy")

    best_label = None
    best_score = 0.0
    for label, cfg in state_cfg["labels"].items():
        candidates = cfg.get("exact", []) + cfg.get("phrases", []) + cfg.get("fuzzy", [])
        for item in candidates:
            score = _semantic_similarity(normalized_text, normalize_text(item, corrections))
            if score > best_score:
                best_score = score
                best_label = label
    if best_label is not None and best_score >= thresholds["semantic_match_confidence"]:
        return build_result(best_label, best_score, "semantic")

    learned_family = interactive_response_family(state)
    if learned_family is not None:
        learned_label, learned_confidence, learned_method = _learned_vocab_match(normalized_text, learned_family)
        if learned_label is not None:
            return build_result(learned_label, learned_confidence, learned_method)

    return MatchResult(
        transcript=transcript,
        normalized_text=normalized_text,
        response_type=response_type,
        canonical_label=None,
        response_value=None,
        confidence=best_score,
        method="fallback",
        reprompt_text=reprompt_text,
        accepted=False,
        reason="low_confidence",
    )
