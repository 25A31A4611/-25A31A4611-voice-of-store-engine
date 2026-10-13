"""
Multilingual, Code-Mixed, and Noisy Retail Text Preprocessing
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

import re
import unicodedata
from typing import Dict, Any, Tuple

# Common Retail Abbreviations Expansion
RETAIL_ABBREVIATIONS = {
    r"\boos\b": "out of stock",
    r"\bpos\b": "point of sale checkout counter",
    r"\bmrp\b": "maximum retail price",
    r"\bupi\b": "upi digital payment",
    r"\bcc\b": "customer care",
    r"\bmgr\b": "store manager",
    r"\bexp\b": "expired",
    r"\bac\b": "air conditioning",
    r"\bdept\b": "department",
    r"\bveg\b": "vegetable",
    r"\bnon-veg\b": "non vegetarian",
    r"\bqty\b": "quantity",
    r"\bpkt\b": "packet",
    r"\bchq\b": "cheque",
    r"\bwt\b": "weight",
    r"\bdeliv\b": "delivery",
    r"\bapp\b": "mobile application",
    r"\bpromo\b": "promotional discount"
}

# Emoji to Semantic Operational Tokens
EMOJI_SEMANTIC_MAP = {
    "🪳": " [pest cockroach] ",
    "🐀": " [pest rat rodent] ",
    "🐁": " [pest mouse] ",
    "🤢": " [unhygienic nauseating food] ",
    "🤮": " [food poisoning vomiting] ",
    "😡": " [angry customer service] ",
    "🤬": " [extremely angry abusive] ",
    "👎": " [dissatisfied bad experience] ",
    "❄️": " [cooling air conditioning refrigerator] ",
    "🧊": " [freezer refrigeration cold storage] ",
    "🔥": " [fire hazard heat] ",
    "⏳": " [long wait delay queue] ",
    "⌛": " [long waiting queue] ",
    "💸": " [overcharged extra money billed] ",
    "💳": " [card payment pos swipe] ",
    "⚠️": " [warning safety hazard] ",
    "🚨": " [emergency hazard alert] "
}

# Transliterated Hindi / Hinglish phrases to normalized English semantic tokens
HINGLISH_MAP = {
    r"\bkaam nahi kar raha\b": "not working dysfunctional",
    r"\bkam nahi kar raha\b": "not working",
    r"\bchal nahi raha\b": "not working broken",
    r"\bkharab hai\b": "defective spoiled bad condition",
    r"\bkharab tha\b": "spoiled defective",
    r"\bsada hua\b": "rotten decayed spoiled food",
    r"\bbadboo aa rahi\b": "foul smell stink unhygienic",
    r"\bbadboo\b": "foul odor stench",
    r"\bbheed bohot hai\b": "excessive crowd long queues",
    r"\bbheed\b": "crowded queue delay",
    r"\bpaise kaat liye\b": "money debited overcharged billing issue",
    r"\bpaise zyada liye\b": "overcharged above mrp billing issue",
    r"\bpaisa zyada\b": "overcharged billing issue",
    r"\bpaise kat gaye\b": "money deducted twice billing",
    r"\bnahi mila\b": "item not found out of stock unavailable",
    r"\bstock khatam\b": "out of stock empty shelves",
    r"\bchori\b": "theft stolen misconduct",
    r"\bkida\b": "insect bug pest in product",
    r"\bkide\b": "insects bugs worms pest",
    r"\blifter\b": "elevator lift facilities",
    r"\btamiz nahi hai\b": "rude disrespectful staff behavior",
    r"\btameez nahi\b": "rude misbehaving staff",
    r"\bchilla raha tha\b": "shouting aggressively staff misconduct",
    r"\bladai\b": "argument altercation fight",
    r"\bchhota bacha gir gaya\b": "child slipped physical injury safety hazard"
}

# Transliterated Tamil phrases
TAMIL_MAP = {
    r"\bromba mosam\b": "extremely bad terrible condition",
    r"\bmosama irukku\b": "terrible condition",
    r"\bpanam eduthutanga\b": "money debited wrongly billing",
    r"\bvela seiyala\b": "not working out of order",
    r"\bazhukku\b": "dirty filthy unhygienic",
    r"\bkedaikala\b": "not available out of stock",
    r"\bmariathai illa\b": "no respect rude staff conduct",
    r"\beppo paarthalum kootam\b": "always overcrowded long wait checkout",
    r"\bpuzhu irukku\b": "worms insects food contamination pest"
}

# Transliterated Marathi phrases
MARATHI_MAP = {
    r"\bpaise jaast ghetle\b": "overcharged more money taken billing",
    r"\bkharab aahe\b": "spoiled defective rotten",
    r"\bkahi milat nahi\b": "nothing available out of stock",
    r"\bsaaf nahi\b": "not clean dirty unhygienic",
    r"\bbhandan zala\b": "altercation fight staff misconduct",
    r"\bkamgar vait vagle\b": "staff behaved badly rude conduct"
}

# Common Spelling Typos in retail feedback
TYPO_CORRECTIONS = {
    r"\bexpird\b": "expired",
    r"\bexpirey\b": "expiry",
    r"\brefrigarator\b": "refrigerator",
    r"\bfrige\b": "fridge",
    r"\bcashiar\b": "cashier",
    r"\bcashir\b": "cashier",
    r"\bmanagr\b": "manager",
    r"\bmanger\b": "manager",
    r"\bunhygenic\b": "unhygienic",
    r"\bunhygenic\b": "unhygienic",
    r"\bcockroch\b": "cockroach",
    r"\bque\b": "queue",
    r"\bq\b": "queue",
    r"\boos\b": "out of stock",
    r"\bdescount\b": "discount"
}


def detect_language_characteristics(text: str) -> Dict[str, Any]:
    """
    Detect language characteristics, scripts, and code-mixing indicators.
    Identifies English, Devanagari, Tamil, Hinglish, or Transliterated Regional.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "primary_language": "Unknown",
            "is_code_mixed": False,
            "has_emojis": False,
            "confidence": 0.0
        }

    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))
    has_emojis = any(char in EMOJI_SEMANTIC_MAP for char in text)

    # Check for Hinglish / Romanized vernacular patterns
    hinglish_hit = any(re.search(pat, text, re.IGNORECASE) for pat in HINGLISH_MAP.keys())
    tamil_trans_hit = any(re.search(pat, text, re.IGNORECASE) for pat in TAMIL_MAP.keys())
    marathi_trans_hit = any(re.search(pat, text, re.IGNORECASE) for pat in MARATHI_MAP.keys())

    if has_devanagari:
        lang = "Hindi / Marathi (Devanagari)"
        code_mixed = bool(re.search(r'[A-Za-z]', text))
        conf = 0.92
    elif has_tamil:
        lang = "Tamil (Script)"
        code_mixed = bool(re.search(r'[A-Za-z]', text))
        conf = 0.94
    elif hinglish_hit:
        lang = "Hinglish (Code-Mixed Roman Hindi)"
        code_mixed = True
        conf = 0.88
    elif tamil_trans_hit:
        lang = "Tamil (Transliterated Roman)"
        code_mixed = True
        conf = 0.85
    elif marathi_trans_hit:
        lang = "Marathi (Transliterated Roman)"
        code_mixed = True
        conf = 0.85
    else:
        lang = "English"
        code_mixed = False
        conf = 0.95

    return {
        "primary_language": lang,
        "is_code_mixed": code_mixed,
        "has_emojis": has_emojis,
        "confidence": conf
    }


def normalize_feedback_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Normalizes feedback text while preserving the original verbatim.
    Expands emojis, translates Hinglish / transliterated vernacular idioms,
    corrects retail typos, and expands operational abbreviations.
    
    Returns:
        normalized_text: str (clean tokenized representation for ML & pattern matching)
        metadata: Dict[str, Any] (clarity score, language, human review flag)
    """
    if not isinstance(text, str) or not text.strip():
        return "", {
            "primary_language": "Empty",
            "is_code_mixed": False,
            "has_emojis": False,
            "clarity_score": 0.0,
            "needs_human_review": True,
            "review_reason": "Missing or empty feedback text"
        }

    original = text.strip()
    lang_info = detect_language_characteristics(original)

    # 1. Emoji normalization to operational semantic tokens
    normalized = original
    for emoji_char, token in EMOJI_SEMANTIC_MAP.items():
        if emoji_char in normalized:
            normalized = normalized.replace(emoji_char, token)

    # 2. Transliterated phrase mappings
    for pattern, replacement in HINGLISH_MAP.items():
        normalized = re.sub(pattern, f" {replacement} ", normalized, flags=re.IGNORECASE)

    for pattern, replacement in TAMIL_MAP.items():
        normalized = re.sub(pattern, f" {replacement} ", normalized, flags=re.IGNORECASE)

    for pattern, replacement in MARATHI_MAP.items():
        normalized = re.sub(pattern, f" {replacement} ", normalized, flags=re.IGNORECASE)

    # 3. Abbreviations expansion
    for pattern, replacement in RETAIL_ABBREVIATIONS.items():
        normalized = re.sub(pattern, f" {replacement} ", normalized, flags=re.IGNORECASE)

    # 4. Common spelling typos
    for pattern, replacement in TYPO_CORRECTIONS.items():
        normalized = re.sub(pattern, f" {replacement} ", normalized, flags=re.IGNORECASE)

    # 5. Clean whitespace and non-standard characters
    normalized = re.sub(r'[\r\n\t]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # 6. Quality & clarity heuristic
    word_count = len(normalized.split())
    char_count = len(normalized)

    # Evaluate whether human review is needed
    needs_review = False
    review_reasons = []

    if word_count < 3 and not any(tag in normalized for tag in ["[pest", "[unhygienic", "[fire", "[warning"]):
        needs_review = True
        review_reasons.append("Very short text (< 3 words) without specific operational indicator")

    if lang_info["confidence"] < 0.80:
        needs_review = True
        review_reasons.append("Low language identification confidence")

    if re.fullmatch(r'^[!?.\s\d]+$', original):
        needs_review = True
        review_reasons.append("Punctuation or digits only")

    clarity_score = min(1.0, max(0.2, (word_count / 15.0) * lang_info["confidence"]))

    metadata = {
        "primary_language": lang_info["primary_language"],
        "is_code_mixed": lang_info["is_code_mixed"],
        "has_emojis": lang_info["has_emojis"],
        "word_count": word_count,
        "clarity_score": round(clarity_score, 2),
        "needs_human_review": needs_review,
        "review_reason": "; ".join(review_reasons) if review_reasons else "Normal"
    }

    return normalized, metadata
