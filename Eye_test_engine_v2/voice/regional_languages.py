"""Regional language support for Eye Test Engine voice pipeline.

Provides translations, keyword maps, and TTS voice configuration
for Indian regional languages: Tamil, Telugu, Kannada, Marathi,
Malayalam, Gujarati, Bengali, Punjabi.

Each language provides:
    - FSM question translations (native script)
    - Fuzzy matcher keywords (native script + romanized)
    - Follow-up phrases for repeated questions
    - Rephrased questions for silence timeout
    - TTS voice configuration (Piper or Meta MMS model ID)
"""

# ── Language Registry ────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "script": "Latin", "tts_engine": "piper", "tts_model": "en_US-kusal-medium"},
    "hi": {"name": "Hindi", "script": "Devanagari", "tts_engine": "piper", "tts_model": "hi_IN-pratham-medium", "mms_model": "facebook/mms-tts-hin"},
    "te": {"name": "Telugu", "script": "Telugu", "tts_engine": "piper", "tts_model": "te_IN-venkatesh-medium", "mms_model": "facebook/mms-tts-tel"},
    "ml": {"name": "Malayalam", "script": "Malayalam", "tts_engine": "piper", "tts_model": "ml_IN-arjun-medium", "mms_model": "facebook/mms-tts-mal"},
    "ta": {"name": "Tamil", "script": "Tamil", "tts_engine": "mms", "mms_model": "facebook/mms-tts-tam"},
    "kn": {"name": "Kannada", "script": "Kannada", "tts_engine": "mms", "mms_model": "facebook/mms-tts-kan"},
    "mr": {"name": "Marathi", "script": "Devanagari", "tts_engine": "mms", "mms_model": "facebook/mms-tts-mar"},
    "gu": {"name": "Gujarati", "script": "Gujarati", "tts_engine": "mms", "mms_model": "facebook/mms-tts-guj"},
    "bn": {"name": "Bengali", "script": "Bengali", "tts_engine": "mms", "mms_model": "facebook/mms-tts-ben"},
    "pa": {"name": "Punjabi", "script": "Gurmukhi", "tts_engine": "mms", "mms_model": "facebook/mms-tts-pan"},
}

# Whisper language codes (ISO 639-1)
WHISPER_LANG_CODES = {
    "en": "en", "hi": "hi", "te": "te", "ml": "ml",
    "ta": "ta", "kn": "kn", "mr": "mr", "gu": "gu",
    "bn": "bn", "pa": "pa",
}


# ── Question Translations ────────────────────────────────────────────────
# Each dict maps the original English FSM question → translated version.

QUESTION_TRANSLATIONS = {
    "te": {  # Telugu
        "Looking at the letters, are they clear, slightly blurry, or not readable?":
            "అక్షరాలను చూడండి, అవి స్పష్టంగా కనిపిస్తున్నాయా, కొంచెం మసకగా ఉన్నాయా, లేదా చదవలేకపోతున్నారా?",
        "Looking at the letters now, are they clear, a bit blurry, or not readable?":
            "ఇప్పుడు అక్షరాలను చూడండి, స్పష్టంగా ఉన్నాయా, మసకగా ఉన్నాయా, లేదా చదవలేకపోతున్నారా?",
        "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
            "ఎరుపు మరియు ఆకుపచ్చ నేపథ్యంలో అక్షరాలను చూడండి. ఏ వైపు స్పష్టంగా ఉంది?",
        "Look at the two rows of letters. Which row looks clearer — the top row, the bottom row, or do they look about the same?":
            "రెండు వరుసలను చూడండి. ఏ వరుస స్పష్టంగా ఉంది — పైది, కిందది, లేదా రెండూ ఒకేలా?",
        "Looking at the near text, is it clear to read, a bit blurry, or not readable?":
            "దగ్గరి టెక్స్ట్ చూడండి. స్పష్టంగా ఉందా, మసకగా ఉందా, లేదా చదవలేకపోతున్నారా?",
        "Looking at the near text with both eyes, is it clear and comfortable, or still not clear?":
            "రెండు కళ్ళతో దగ్గరి టెక్స్ట్ చూడండి. స్పష్టంగా మరియు సౌకర్యంగా ఉందా?",
    },
    "ta": {  # Tamil
        "Looking at the letters, are they clear, slightly blurry, or not readable?":
            "எழுத்துக்களைப் பாருங்கள், அவை தெளிவாக இருக்கிறதா, சற்று மங்கலாக இருக்கிறதா, அல்லது படிக்க முடியவில்லையா?",
        "Looking at the letters now, are they clear, a bit blurry, or not readable?":
            "இப்போது எழுத்துக்களைப் பாருங்கள், தெளிவாக இருக்கிறதா, மங்கலாக இருக்கிறதா, அல்லது படிக்க முடியவில்லையா?",
        "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
            "சிவப்பு மற்றும் பச்சை பின்னணியில் எழுத்துக்களைப் பாருங்கள். எந்த பக்கம் தெளிவாக இருக்கிறது?",
        "Look at the two rows of letters. Which row looks clearer — the top row, the bottom row, or do they look about the same?":
            "இரண்டு வரிசைகளையும் பாருங்கள். எந்த வரிசை தெளிவாக இருக்கிறது — மேல், கீழ், அல்லது இரண்டும் ஒரே மாதிரி?",
        "Looking at the near text, is it clear to read, a bit blurry, or not readable?":
            "அருகிலுள்ள எழுத்தைப் பாருங்கள். தெளிவாக இருக்கிறதா, மங்கலாக இருக்கிறதா, அல்லது படிக்க முடியவில்லையா?",
        "Looking at the near text with both eyes, is it clear and comfortable, or still not clear?":
            "இரண்டு கண்களாலும் அருகிலுள்ள எழுத்தைப் பாருங்கள். தெளிவாகவும் வசதியாகவும் இருக்கிறதா?",
    },
    "kn": {  # Kannada
        "Looking at the letters, are they clear, slightly blurry, or not readable?":
            "ಅಕ್ಷರಗಳನ್ನು ನೋಡಿ, ಅವು ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುತ್ತಿವೆಯೇ, ಸ್ವಲ್ಪ ಮಸುಕಾಗಿವೆಯೇ, ಅಥವಾ ಓದಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲವೇ?",
        "Looking at the letters now, are they clear, a bit blurry, or not readable?":
            "ಈಗ ಅಕ್ಷರಗಳನ್ನು ನೋಡಿ, ಸ್ಪಷ್ಟವಾಗಿದೆಯೇ, ಮಸುಕಾಗಿದೆಯೇ, ಅಥವಾ ಓದಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲವೇ?",
        "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
            "ಕೆಂಪು ಮತ್ತು ಹಸಿರು ಹಿನ್ನೆಲೆಯಲ್ಲಿ ಅಕ್ಷರಗಳನ್ನು ನೋಡಿ. ಯಾವ ಕಡೆ ಸ್ಪಷ್ಟವಾಗಿದೆ?",
    },
    "mr": {  # Marathi
        "Looking at the letters, are they clear, slightly blurry, or not readable?":
            "अक्षरे बघा, ती स्पष्ट दिसत आहेत, थोडी धूसर आहेत, की वाचता येत नाहीत?",
        "Looking at the letters now, are they clear, a bit blurry, or not readable?":
            "आता अक्षरे बघा, स्पष्ट आहेत, धूसर आहेत, की वाचता येत नाहीत?",
        "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
            "लाल आणि हिरव्या पार्श्वभूमीवर अक्षरे बघा. कोणती बाजू स्पष्ट आहे?",
    },
}


# ── Keyword Maps (native script + romanized) ─────────────────────────────
# Maps per language for fuzzy matching of patient responses.

REGIONAL_KEYWORDS = {
    "te": {  # Telugu
        "READABILITY": {
            "READABLE": ["స్పష్టం", "కనిపిస్తోంది", "చదవగలను", "అవును", "బాగుంది", "clear", "yes"],
            "NOT_READABLE": ["కనిపించడం లేదు", "చదవలేను", "లేదు", "ఏమీ లేదు", "no"],
            "BLURRY": ["మసకం", "మసకగా", "కొంచెం", "blurry"],
        },
        "COMPARE_1_2": {
            "BETTER_1": ["మొదటిది", "ఒకటి", "first", "one"],
            "BETTER_2": ["రెండవది", "రెండు", "second", "two"],
            "SAME": ["ఒకేలా", "సమానం", "same"],
            "CANT_TELL": ["తెలియదు", "అర్థం కాలేదు", "can't tell"],
        },
        "COLOR_CHOICE": {
            "RED_CLEARER": ["ఎరుపు", "red"],
            "GREEN_CLEARER": ["ఆకుపచ్చ", "green"],
            "EQUAL": ["సమానం", "ఒకేలా", "same"],
        },
        "TOP_BOTTOM": {
            "TOP_CLEARER": ["పైది", "పైన", "top"],
            "BOTTOM_CLEARER": ["కిందది", "కింద", "bottom"],
            "SAME": ["ఒకేలా", "same"],
        },
        "NEAR_BINOC": {
            "TARGET_OK": ["స్పష్టం", "బాగుంది", "అవును", "clear", "yes"],
            "NOT_CLEAR": ["స్పష్టం కాదు", "లేదు", "not clear", "no"],
        },
    },
    "ta": {  # Tamil
        "READABILITY": {
            "READABLE": ["தெளிவு", "படிக்க முடிகிறது", "ஆம்", "நன்றாக", "clear", "yes"],
            "NOT_READABLE": ["தெரியவில்லை", "படிக்க முடியவில்லை", "இல்லை", "no"],
            "BLURRY": ["மங்கல்", "சற்று மங்கல்", "blurry"],
        },
        "COMPARE_1_2": {
            "BETTER_1": ["முதல்", "ஒன்று", "first", "one"],
            "BETTER_2": ["இரண்டாவது", "இரண்டு", "second", "two"],
            "SAME": ["ஒரே மாதிரி", "சமம்", "same"],
            "CANT_TELL": ["தெரியாது", "புரியவில்லை", "can't tell"],
        },
        "COLOR_CHOICE": {
            "RED_CLEARER": ["சிவப்பு", "red"],
            "GREEN_CLEARER": ["பச்சை", "green"],
            "EQUAL": ["சமம்", "ஒரே மாதிரி", "same"],
        },
        "TOP_BOTTOM": {
            "TOP_CLEARER": ["மேல்", "top"],
            "BOTTOM_CLEARER": ["கீழ்", "bottom"],
            "SAME": ["ஒரே மாதிரி", "same"],
        },
        "NEAR_BINOC": {
            "TARGET_OK": ["தெளிவு", "ஆம்", "clear", "yes"],
            "NOT_CLEAR": ["தெளிவில்லை", "இல்லை", "not clear", "no"],
        },
    },
    "kn": {  # Kannada
        "READABILITY": {
            "READABLE": ["ಸ್ಪಷ್ಟ", "ಕಾಣುತ್ತಿದೆ", "ಹೌದು", "clear", "yes"],
            "NOT_READABLE": ["ಕಾಣುತ್ತಿಲ್ಲ", "ಓದಲಾಗುತ್ತಿಲ್ಲ", "ಇಲ್ಲ", "no"],
            "BLURRY": ["ಮಸುಕು", "ಸ್ವಲ್ಪ ಮಸುಕು", "blurry"],
        },
        "COMPARE_1_2": {
            "BETTER_1": ["ಮೊದಲನೆಯದು", "ಒಂದು", "first", "one"],
            "BETTER_2": ["ಎರಡನೆಯದು", "ಎರಡು", "second", "two"],
            "SAME": ["ಒಂದೇ", "ಸಮಾನ", "same"],
            "CANT_TELL": ["ಗೊತ್ತಿಲ್ಲ", "ಅರ್ಥವಾಗಲಿಲ್ಲ"],
        },
        "COLOR_CHOICE": {
            "RED_CLEARER": ["ಕೆಂಪು", "red"],
            "GREEN_CLEARER": ["ಹಸಿರು", "green"],
            "EQUAL": ["ಸಮಾನ", "ಒಂದೇ", "same"],
        },
    },
    "mr": {  # Marathi
        "READABILITY": {
            "READABLE": ["स्पष्ट", "दिसतंय", "होय", "बरोबर", "clear", "yes"],
            "NOT_READABLE": ["दिसत नाही", "वाचता येत नाही", "नाही", "no"],
            "BLURRY": ["धूसर", "थोडं धूसर", "blurry"],
        },
        "COMPARE_1_2": {
            "BETTER_1": ["पहिला", "एक", "first", "one"],
            "BETTER_2": ["दुसरा", "दोन", "second", "two"],
            "SAME": ["सारखं", "समान", "same"],
            "CANT_TELL": ["कळत नाही", "समजत नाही"],
        },
        "COLOR_CHOICE": {
            "RED_CLEARER": ["लाल", "red"],
            "GREEN_CLEARER": ["हिरवा", "green"],
            "EQUAL": ["सारखं", "समान", "same"],
        },
    },
}


# ── Follow-up Phrases ────────────────────────────────────────────────────

REGIONAL_FOLLOWUPS = {
    "te": {
        "state_B": ["ఇప్పుడు ఎలా ఉంది?", "ఈ వరుస?", "ఏదైనా మార్పు?", "తరువాతి వరుస।"],
        "state_D": ["ఇప్పుడు ఎలా ఉంది?", "ఈ వరుస?", "ఏదైనా మార్పు?", "తరువాతి వరుస。"],
        "COMPARE_1_2": ["ఈసారి?", "ఏదైనా తేడా?", "ఇప్పుడు ఎలా?"],
    },
    "ta": {
        "state_B": ["இப்போது எப்படி?", "இந்த வரிசை?", "ஏதாவது மாற்றம்?", "அடுத்த வரிசை।"],
        "state_D": ["இப்போது எப்படி?", "இந்த வரிசை?", "ஏதாவது மாற்றம்?", "அடுத்த வரிசை।"],
        "COMPARE_1_2": ["இப்போது?", "ஏதாவது வித்தியாசம்?"],
    },
    "kn": {
        "state_B": ["ಈಗ ಹೇಗಿದೆ?", "ಈ ಸಾಲು?", "ಏನಾದರೂ ಬದಲಾವಣೆ?", "ಮುಂದಿನ ಸಾಲು।"],
        "state_D": ["ಈಗ ಹೇಗಿದೆ?", "ಈ ಸಾಲು?", "ಏನಾದರೂ ಬದಲಾವಣೆ?"],
    },
    "mr": {
        "state_B": ["आता कसं दिसतंय?", "ही ओळ?", "काही बदल?", "पुढची ओळ।"],
        "state_D": ["आता कसं दिसतंय?", "ही ओळ?", "काही बदल?"],
    },
}


# ── Rephrased Questions (silence timeout) ────────────────────────────────

REGIONAL_REPHRASED = {
    "te": {
        "READABILITY": "చదవగలరా?",
        "COMPARE_1_2": "మొదటిదా రెండవదా?",
        "COLOR_CHOICE": "ఏ వైపు స్పష్టం?",
        "TOP_BOTTOM": "ఏ వరుస స్పష్టం?",
        "NEAR_BINOC": "స్పష్టంగా ఉందా?",
    },
    "ta": {
        "READABILITY": "படிக்க முடிகிறதா?",
        "COMPARE_1_2": "முதலாவது அல்லது இரண்டாவது?",
        "COLOR_CHOICE": "எந்த பக்கம் தெளிவு?",
        "TOP_BOTTOM": "எந்த வரிசை தெளிவு?",
        "NEAR_BINOC": "தெளிவாக இருக்கிறதா?",
    },
    "kn": {
        "READABILITY": "ಓದಲು ಸಾಧ್ಯವಾಗುತ್ತಿದೆಯೇ?",
        "COMPARE_1_2": "ಮೊದಲನೆಯದು ಅಥವಾ ಎರಡನೆಯದು?",
        "COLOR_CHOICE": "ಯಾವ ಕಡೆ ಸ್ಪಷ್ಟ?",
        "TOP_BOTTOM": "ಯಾವ ಸಾಲು ಸ್ಪಷ್ಟ?",
        "NEAR_BINOC": "ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣುತ್ತಿದೆಯೇ?",
    },
    "mr": {
        "READABILITY": "वाचता येतंय का?",
        "COMPARE_1_2": "पहिला की दुसरा?",
        "COLOR_CHOICE": "कोणती बाजू स्पष्ट?",
        "TOP_BOTTOM": "कोणती ओळ स्पष्ट?",
        "NEAR_BINOC": "स्पष्ट दिसतंय का?",
    },
}


# ── System Messages ──────────────────────────────────────────────────────

REGIONAL_MESSAGES = {
    "te": {
        "test_complete": "పరీక్ష పూర్తయింది. ధన్యవాదాలు.",
        "no_match": "అర్థం కాలేదు. దయచేసి మళ్ళీ చెప్పండి.",
        "exit_confirm": "పరీక్ష ఆపాలా? అవును లేదా కాదు చెప్పండి.",
        "exit_yes": "సరే, పరీక్ష ఆపుతున్నాం.",
        "exit_no": "సరే, కొనసాగిద్దాం.",
        "flip1": "ఇది ఒకటి.",
        "flip2": "ఇది రెండు. ఏది మెరుగు?",
    },
    "ta": {
        "test_complete": "பரிசோதனை முடிந்தது. நன்றி.",
        "no_match": "புரியவில்லை. மீண்டும் சொல்லுங்கள்.",
        "exit_confirm": "பரிசோதனையை நிறுத்த வேண்டுமா? ஆம் அல்லது இல்லை சொல்லுங்கள்.",
        "exit_yes": "சரி, நிறுத்துகிறோம்.",
        "exit_no": "சரி, தொடர்வோம்.",
        "flip1": "இது ஒன்று.",
        "flip2": "இது இரண்டு. எது சிறந்தது?",
    },
    "kn": {
        "test_complete": "ಪರೀಕ್ಷೆ ಮುಗಿಯಿತು. ಧನ್ಯವಾದಗಳು.",
        "no_match": "ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಹೇಳಿ.",
        "exit_confirm": "ಪರೀಕ್ಷೆ ನಿಲ್ಲಿಸಬೇಕಾ? ಹೌದು ಅಥವಾ ಇಲ್ಲ ಹೇಳಿ.",
        "exit_yes": "ಸರಿ, ನಿಲ್ಲಿಸುತ್ತೇವೆ.",
        "exit_no": "ಸರಿ, ಮುಂದುವರಿಸೋಣ.",
        "flip1": "ಇದು ಒಂದು.",
        "flip2": "ಇದು ಎರಡು. ಯಾವುದು ಉತ್ತಮ?",
    },
    "mr": {
        "test_complete": "तपासणी पूर्ण झाली. धन्यवाद.",
        "no_match": "समजलं नाही. कृपया पुन्हा सांगा.",
        "exit_confirm": "तपासणी थांबवायची का? होय किंवा नाही सांगा.",
        "exit_yes": "ठीक आहे, थांबवतो.",
        "exit_no": "ठीक आहे, चालू ठेवू.",
        "flip1": "हा पहिला.",
        "flip2": "हा दुसरा. कोणता चांगला?",
    },
}


# ── Additional translations for ml, gu, bn, pa ──────────────────────────
QUESTION_TRANSLATIONS["ml"] = {
    "Looking at the letters, are they clear, slightly blurry, or not readable?":
        "അക്ഷരങ്ങൾ നോക്കൂ, അവ വ്യക്തമായി കാണുന്നുണ്ടോ, അല്പം മങ്ങിയതാണോ, അതോ വായിക്കാൻ കഴിയുന്നില്ലേ?",
    "Looking at the letters now, are they clear, a bit blurry, or not readable?":
        "ഇപ്പോൾ അക്ഷരങ്ങൾ നോക്കൂ, വ്യക്തമാണോ, മങ്ങിയതാണോ, അതോ വായിക്കാൻ കഴിയുന്നില്ലേ?",
    "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
        "ചുവപ്പും പച്ചയും പശ്ചാത്തലത്തിൽ അക്ഷരങ്ങൾ നോക്കൂ. ഏത് ഭാഗം വ്യക്തമാണ്?",
    "Look at the two rows of letters. Which row looks clearer — the top row, the bottom row, or do they look about the same?":
        "രണ്ട് വരികൾ നോക്കൂ. ഏത് വരി വ്യക്തമാണ് — മുകളിലെ, താഴെയുള്ള, അതോ രണ്ടും ഒരുപോലെ?",
}
QUESTION_TRANSLATIONS["gu"] = {
    "Looking at the letters, are they clear, slightly blurry, or not readable?":
        "અક્ષરો જુઓ, શું તે સ્પષ્ટ દેખાય છે, થોડા ઝાંખા છે, કે વાંચી શકાતા નથી?",
    "Looking at the letters now, are they clear, a bit blurry, or not readable?":
        "હવે અક્ષરો જુઓ, સ્પષ્ટ છે, ઝાંખા છે, કે વાંચી શકાતા નથી?",
    "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
        "લાલ અને લીલા બેકગ્રાઉન્ડ પર અક્ષરો જુઓ. કઈ બાજુ વધુ સ્પષ્ટ છે?",
}
QUESTION_TRANSLATIONS["bn"] = {
    "Looking at the letters, are they clear, slightly blurry, or not readable?":
        "অক্ষরগুলো দেখুন, সেগুলো কি পরিষ্কার দেখাচ্ছে, একটু ঝাপসা, নাকি পড়া যাচ্ছে না?",
    "Looking at the letters now, are they clear, a bit blurry, or not readable?":
        "এখন অক্ষরগুলো দেখুন, পরিষ্কার, ঝাপসা, নাকি পড়া যাচ্ছে না?",
    "Looking at the letters on the red and green backgrounds, which side looks clearer — red, green, or do they look about the same?":
        "লাল ও সবুজ পটভূমিতে অক্ষরগুলো দেখুন। কোন দিকটা বেশি পরিষ্কার?",
}
QUESTION_TRANSLATIONS["pa"] = {
    "Looking at the letters, are they clear, slightly blurry, or not readable?":
        "ਅੱਖਰ ਦੇਖੋ, ਕੀ ਇਹ ਸਾਫ਼ ਦਿਖਾਈ ਦੇ ਰਹੇ ਹਨ, ਥੋੜੇ ਧੁੰਦਲੇ ਹਨ, ਜਾਂ ਪੜ੍ਹੇ ਨਹੀਂ ਜਾ ਰਹੇ?",
    "Looking at the letters now, are they clear, a bit blurry, or not readable?":
        "ਹੁਣ ਅੱਖਰ ਦੇਖੋ, ਸਾਫ਼ ਹਨ, ਧੁੰਦਲੇ ਹਨ, ਜਾਂ ਪੜ੍ਹੇ ਨਹੀਂ ਜਾ ਰਹੇ?",
}


def get_translation(lang: str, english_question: str) -> str:
    """Get translated question for a language. Falls back to English."""
    translations = QUESTION_TRANSLATIONS.get(lang, {})
    return translations.get(english_question, english_question)


def get_keywords(lang: str) -> dict:
    """Get keyword map for a language. Falls back to English in fuzzy_matcher."""
    return REGIONAL_KEYWORDS.get(lang, {})


def get_followup(lang: str, response_type: str, state: str = "") -> str:
    """Get a random follow-up phrase for repeated questions."""
    import random
    followups = REGIONAL_FOLLOWUPS.get(lang, {})
    pool = followups.get(f"state_{state}") or followups.get(response_type)
    if pool:
        return random.choice(pool)
    return ""


def get_rephrased(lang: str, response_type: str) -> str:
    """Get rephrased question for silence timeout."""
    rephrased = REGIONAL_REPHRASED.get(lang, {})
    return rephrased.get(response_type, "")


def get_message(lang: str, key: str) -> str:
    """Get a system message in the given language."""
    messages = REGIONAL_MESSAGES.get(lang, {})
    return messages.get(key, "")
