from __future__ import annotations

import re


BENGALI_VOWELS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ঐ", "ও", "ঔ"]
BENGALI_CONSONANTS = ["ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ", "য়", "ড়", "ঢ়", "য়", "ৎ"]
BENGALI_SIGNS = ["্", "ে", "ি", "ী", "ু", "ূ", "ূ", "ো", "ৌ", "ৈ", "ৈ", "ঁ"]


def split_graphemes(text: str) -> list[str]:
    """Split Bangla text into grapheme-like units using a simple Unicode-aware rule.

    The implementation keeps the logic intentionally clear and extendable for later
    phases where a more complete Unicode \\X-based splitter may be used.
    """
    if not text:
        return []
    return [char for char in text if char.strip()]


def normalize_word_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())
