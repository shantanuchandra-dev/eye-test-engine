from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Optional


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
                    "exact": ["blur", "blurry", "unclear", "hazy", "fuzzy"],
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
                    "exact": ["same", "equal"],
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
                    "exact": ["same", "equal"],
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
                    "exact": ["same", "equal"],
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
                    "exact": ["blur", "blurry", "unclear", "fuzzy", "hazy"],
                    "phrases": [
                        "not clear",
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
                "சரி", "seri", "sari", "தெளிவு", "thelivu",
            ],
            "phrases": [
                "साफ है", "साफ दिख रहा है", "दिख रहा है", "पढ़ पा रहा हूं", "पढ पा रहा हूं",
                "saaf dikh raha", "dikh raha", "padh pa raha", "i can read", "i can see", "தெளிவா இருக்கு", "நல்லா தெரியுது", "படிக்க முடிகிறது",
                "padikka mudiyuthu", "theriyuthu",
            ],
            "fuzzy": [
                "haan", "han", "haanjii", "thek", "thik hai", "saf", "saaf",
                "dikhrha", "dikhra", "padh pa raha", "pad pa raha", "theliva", "thelva", "theriyudhu",
            ],
        },
        "BLURRY": {
            "exact": ["धुंधला", "dhundla", "ब्लर", "blur", "மங்கலா", "mangala"],
            "phrases": [
                "ब्लर है", "धुंधला है", "साफ नहीं", "थोड़ा blur है", "थोड़ा धुंधला है",
                "மங்கலா இருக்கு", "தெளிவா இல்ல", "blur aa irukku", "konjam blur", "konjam mangala",
            ],
            "fuzzy": [
                "dhundhla", "dhudla", "blur hai", "thoda blur", "mangla", "mangla irukku", "theliva illa",
            ],
        },
        "NOT_VISIBLE": {
            "exact": ["தெரியல", "theriyala", "नहीं", "nahi"],
            "phrases": [
                "नहीं दिख रहा", "कुछ नहीं दिख रहा", "पढ़ नहीं पा रहा", "पढ नहीं पा रहा",
                "nahi dikh raha", "nahi padh pa raha", "nahi padh raha",
                "எதுவும் தெரியல", "படிக்க முடியல", "onnum theriyala", "padikka mudiyala",
                "not visible", "not readable",
            ],
            "fuzzy": [
                "nahi dikh raha", "nhi dikh raha", "kuch nahi dikh", "therila", "theryala", "mudiyala",
            ],
        },
    },
    "comparison_4way": {
        "BETTER_1": {
            "exact": ["एक", "ek", "पहला", "pehla", "ஒன்று", "onnu", "முதல்", "mudhal"],
            "phrases": ["पहला बेहतर", "एक बेहतर", "पहला साफ", "ஒன்று நல்லது", "முதல் நல்லது", "onnu nalladhu"],
            "fuzzy": ["ik", "yek", "pahla", "onu", "mudal"],
        },
        "BETTER_2": {
            "exact": ["दो", "do", "दूसरा", "dusra", "ரெண்டு", "rendu", "இரண்டு", "irandu", "இரண்டாவது", "irandavathu"],
            "phrases": ["दूसरा बेहतर", "दो बेहतर", "दूसरा साफ", "ரெண்டு நல்லது", "இரண்டு நல்லது", "rendu nalladhu"],
            "fuzzy": ["doo", "dusara", "rendu", "irand", "renduu"],
        },
        "SAME": {
            "exact": ["बराबर", "barabar", "बरावर", "समान", "ஒண்ணே", "same"],
            "phrases": ["दोनों same", "कोई फर्क नहीं", "एक जैसा", "ஒரே மாதிரி", "same தான்", "ரெண்டும் same"],
            "fuzzy": ["barabr", "barber", "ore madhiri", "same than"],
        },
        "CANT_TELL": {
            "exact": ["पता नहीं", "puriyala", "புரியல", "unsure"],
            "phrases": ["समझ नहीं आ रहा", "तय नहीं", "தெரியல", "சொல்ல முடியல", "not able to say", "purila"],
            "fuzzy": ["pata nai", "samaj nahi", "theriyala", "solla mudiyala"],
        },
    },
    "red_green_4way": {
        "RED_CLEARER": {
            "exact": ["लाल", "lal", "சிகப்பு", "sigappu"],
            "phrases": ["लाल साफ है", "लाल side", "சிகப்பு நல்லா இருக்கு", "sigappu side"],
            "fuzzy": ["laal", "sigapu", "sigapp"],
        },
        "GREEN_CLEARER": {
            "exact": ["हरा", "hara", "பச்சை", "pachai"],
            "phrases": ["हरा साफ है", "हरा side", "பச்சை நல்லா இருக்கு", "pachai side"],
            "fuzzy": ["haraa", "pacha", "pachay"],
        },
        "SAME": {
            "exact": ["बराबर", "barabar", "ஒண்ணே", "same"],
            "phrases": ["कोई फर्क नहीं", "दोनों same", "ஒரே மாதிரி", "same than"],
            "fuzzy": ["barabr", "ore madhiri"],
        },
        "CANT_TELL": {
            "exact": ["पता नहीं", "புரியல", "unsure"],
            "phrases": ["समझ नहीं आ रहा", "तय नहीं", "தெரியல", "solla mudiyala"],
            "fuzzy": ["pata nai", "theriyala", "purila"],
        },
    },
    "binocular_balance_4way": {
        "TOP_CLEARER": {
            "exact": ["ऊपर", "upar", "மேல்", "மேல", "mela"],
            "phrases": ["ऊपर साफ है", "ऊपर better", "மேல நல்லா இருக்கு", "mela nalla irukku"],
            "fuzzy": ["uparr", "mel", "melaa"],
        },
        "BOTTOM_CLEARER": {
            "exact": ["नीचे", "neeche", "கீழ", "கீழே", "keela"],
            "phrases": ["नीचे साफ है", "नीचे better", "கீழ நல்லா இருக்கு", "keela nalla irukku"],
            "fuzzy": ["niche", "nichay", "keel", "keelaa"],
        },
        "SAME": {
            "exact": ["बराबर", "barabar", "ஒண்ணே", "same"],
            "phrases": ["कोई फर्क नहीं", "दोनों same", "ஒரே மாதிரி"],
            "fuzzy": ["barabr", "ore madhiri"],
        },
        "CANT_TELL": {
            "exact": ["पता नहीं", "புரியல", "unsure"],
            "phrases": ["समझ नहीं आ रहा", "தெரியல", "solla mudiyala"],
            "fuzzy": ["pata nai", "theriyala", "purila"],
        },
    },
    "near_clarity_3way": {
        "CLEAR": {
            "exact": ["हाँ", "haan", "ठीक", "theek", "சரி", "seri", "தெளிவு", "thelivu"],
            "phrases": ["पढ़ पा रहा हूं", "साफ है", "clear hai", "தெளிவா இருக்கு", "படிக்க முடிகிறது", "padikka mudiyuthu"],
            "fuzzy": ["padh pa raha", "saf", "theliva", "theriyuthu"],
        },
        "BLURRY": {
            "exact": ["धुंधला", "dhundla", "மங்கலா", "mangala"],
            "phrases": ["पढ़ने में blur है", "थोड़ा blur है", "சற்று மங்கலா இருக்கு", "தெளிவா இல்ல", "konjam mangala"],
            "fuzzy": ["dhundhla", "blur hai", "mangla", "theliva illa"],
        },
        "NOT_VISIBLE": {
            "exact": ["தெரியல", "theriyala", "नहीं", "nahi"],
            "phrases": ["पढ़ नहीं पा रहा", "दिख नहीं रहा", "படிக்க முடியல", "எதுவும் தெரியல", "padikka mudiyala"],
            "fuzzy": ["nahi dikh", "therila", "mudiyala"],
        },
    },
    "near_final_3way": {
        "TARGET_OK": {
            "exact": ["ठीक", "theek", "सही", "sahi", "ஆம்", "aama", "சரி", "seri"],
            "phrases": ["ठीक है", "सही है", "आराम है", "comfortable है", "சரி இருக்கு", "comfortable aa irukku", "romba nalla irukku"],
            "fuzzy": ["thek", "sahi hai", "aamam", "seri irukku"],
        },
        "NEED_ADJUSTMENT": {
            "exact": ["adjust", "बेहतर", "better"],
            "phrases": ["adjust करना है", "और better", "थोड़ा adjust", "சரி இல்ல", "கொஞ்சம் adjust", "இன்னும் better", "konjam adjust"],
            "fuzzy": ["adjest", "better chahiye", "seri illa", "innum better"],
        },
        "NOT_COMFORTABLE": {
            "exact": ["असुविधा", "strain", "கஷ்டம்", "சுகமில்ல"],
            "phrases": ["आराम नहीं है", "strain है", "comfortable नहीं", "சுகமில்ல இருக்கு", "kashtama irukku"],
            "fuzzy": ["aaraam nahi", "comfort nahi", "sugam illa", "kashtam"],
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
    "ta": {
        "clarity_3way": {
            "initial": "எழுத்துகளை பார்த்து சொல்லுங்கள். தெளிவா இருக்கு, கொஞ்சம் மங்கலா இருக்கு, இல்ல படிக்க முடியலா?",
            "retry": "தயவுசெய்து இதிலொன்றை மட்டும் சொல்லுங்கள்: தெளிவு, மங்கலா, அல்லது படிக்க முடியல.",
        },
        "comparison_4way": {
            "initial": "ஒன்று, இரண்டு இதில் எது நல்லா இருக்கு? ஒன்று, இரண்டு, same, இல்ல சொல்ல கஷ்டமா?",
            "retry": "தயவுசெய்து இதிலொன்றை மட்டும் சொல்லுங்கள்: ஒன்று, இரண்டு, same, அல்லது சொல்ல முடியல.",
        },
        "red_green_4way": {
            "initial": "சிகப்பு, பச்சை இதில் எது தெளிவா இருக்கு? சிகப்பு, பச்சை, same, இல்ல சொல்ல முடியலா?",
            "retry": "தயவுசெய்து இதிலொன்றை மட்டும் சொல்லுங்கள்: சிகப்பு, பச்சை, same, அல்லது சொல்ல முடியல.",
        },
        "binocular_balance_4way": {
            "initial": "மேல், கீழ் இதில் எது நல்லா தெரிகிறது? மேல், கீழ், same, இல்ல சொல்ல முடியலா?",
            "retry": "தயவுசெய்து இதிலொன்றை மட்டும் சொல்லுங்கள்: மேல், கீழ், same, அல்லது சொல்ல முடியல.",
        },
        "near_clarity_3way": {
            "initial": "அருகிலுள்ள எழுத்துகளை பார்த்து சொல்லுங்கள். தெளிவா இருக்கு, மங்கலா இருக்கு, இல்ல படிக்க முடியலா?",
            "retry": "தயவுசெய்து இதிலொன்றை மட்டும் சொல்லுங்கள்: தெளிவு, மங்கலா, அல்லது படிக்க முடியல.",
        },
        "near_final_3way": {
            "initial": "இரண்டு கண்ணாலும் அருகிலுள்ள எழுத்து தெளிவாகவும் கம்ஃபர்ட்டாகவும் இருக்கிறதா, இல்ல இன்னும் சரி இல்லையா?",
            "retry": "தயவுசெய்து இதிலொன்றை மட்டும் சொல்லுங்கள்: சரி, adjust வேண்டும், அல்லது comfortable இல்லை.",
        },
    },
}


LANGUAGE_SCRIPT_PATTERNS = {
    "hi": re.compile(r"[\u0900-\u097F]"),
    "ta": re.compile(r"[\u0B80-\u0BFF]"),
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
    "ta": {
        "seri",
        "thelivu",
        "mangala",
        "theriyala",
        "puriyala",
        "onnu",
        "rendu",
        "irandu",
        "sigappu",
        "pachai",
        "mela",
        "keela",
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
        detected_language in {"en", "hi", "ta"}
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

    if detected_language in {"en", "hi", "ta"}:
        return detected_language

    return None


def _context_rule_result(normalized_text: str, response_type: str) -> tuple[bool, str]:
    context_rules = FSM_AUDIO_RESPONSE_LIBRARY_V1["context_rules"]

    yes_terms = {
        "yes", "yeah", "yep", "yup", "ok", "okay",
        "haan", "haanji", "हाँ", "ஆம்", "aama", "aamam",
    }
    good_terms = {
        "good", "fine", "perfect", "comfortable",
        "theek", "thik", "ठीक", "sahi", "सही",
        "seri", "sari", "சரி", "nalla", "நல்லா",
    }
    better_terms = {"better", "बेहतर", "innum better", "konjam adjust"}

    if normalized_text in yes_terms:
        if response_type in context_rules["yes_like_terms"]["disallowed_in"]:
            return False, "context_disallowed"
    if normalized_text in good_terms:
        if response_type in context_rules["good_like_terms"]["disallowed_in"]:
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


def _comparison_token_label(normalized_text: str) -> Optional[str]:
    tokens = set(normalized_text.split())
    one_terms = {"one", "first", "ek", "एक", "pehla", "पहला", "onnu", "ஒன்று", "mudhal", "முதல்"}
    two_terms = {"two", "second", "do", "to", "too", "tu", "दो", "dusra", "दूसरा", "rendu", "ரெண்டு", "irandu", "இரண்டு"}
    same_terms = {"same", "equal", "barabar", "बराबर", "ஒண்ணே"}

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


def _comparison_intent_match(normalized_text: str) -> Optional[str]:
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
        r"தெரியுது",
        r"படிக்க",
        r"சரி",
        r"theriyuthu",
        r"padikka",
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
) -> MatchResult:
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
