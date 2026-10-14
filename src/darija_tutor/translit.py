"""Rule-based Arabic-script -> Arabizi (Latin + numbers) for Darija display.

Arabic script omits short vowels, so this is a best-effort reading aid, not an authority:
long vowels (ا و ي) are rendered, short vowels are only written if the text carries
diacritics. Treat output as a hint and verify against lesson data when available.
"""
import re

_CONSONANTS = {
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "7", "خ": "kh", "د": "d", "ذ": "d",
    "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
    "ع": "3", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "ڭ": "g", "ݣ": "g", "گ": "g", "ڤ": "v", "پ": "p", "ء": "2",
}
_ALEF = {"ا": "a", "أ": "a", "آ": "a", "إ": "i", "ٱ": "a"}
_DIACRITICS = {"َ": "a", "ِ": "i", "ُ": "u"}  # fatha, kasra, damma
_SHADDA, _SUKUN, _TATWEEL = "ّ", "ْ", "ـ"
_PUNCT = {"؟": "?", "،": ",", "؛": ";"}
_ARABIC_WORD = re.compile(r"[؀-ۿݐ-ݿ]+")


def _is_consonant(ch: str) -> bool:
    return ch in _CONSONANTS


def _word(w: str) -> str:
    w = w.replace(_TATWEEL, "")
    if w.startswith("ال") and len(w) > 2:  # definite article: al- -> l-
        w = "ل" + w[2:]
    elif w.startswith("ا") and len(w) > 1 and w[1] == "ل":  # اللي -> lli
        w = w[1:]
    out: list[str] = []
    n = len(w)
    for i, ch in enumerate(w):
        prev = w[i - 1] if i else ""
        nxt = w[i + 1] if i + 1 < n else ""
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
        elif ch in _ALEF:
            out.append(_ALEF[ch])
        elif ch == "ة":
            out.append("a")
        elif ch == "ى":
            out.append("a")
        elif ch == "و":
            # consonant w at the start or next to a vowel, else long u; after a ي it is the u of "yu"
            out.append("w" if i == 0 or prev in _ALEF or nxt in _ALEF or (prev == "و") else "u")
        elif ch == "ي":
            # consonant y at the start, next to a vowel, or before a و (ليوم = lyum), else long i
            out.append("y" if i == 0 or prev in _ALEF or nxt in _ALEF or (nxt == "و" and i + 2 < n) or prev == "ي" else "i")
        elif ch == "ؤ":
            out.append("u")
        elif ch == "ئ":
            out.append("i")
        elif ch in _DIACRITICS:
            out.append(_DIACRITICS[ch])
        elif ch == _SHADDA and out:
            out.append(out[-1] if _is_consonant(prev) else "")
        elif ch == _SUKUN:
            pass
        else:
            out.append(ch)
    return "".join(out)


def for_terminal(text: str) -> str:
    """Join and reorder Arabic so it reads right-to-left in terminals without bidi support.

    Display only: the output uses presentation-form glyphs, so don't feed it back to the model.
    """
    from arabic_reshaper import reshape
    from bidi import get_display

    return "\n".join(
        get_display(reshape(line)) if _ARABIC_WORD.search(line) else line
        for line in text.splitlines()
    )


def to_arabizi(text: str) -> str:
    """Transliterate Arabic-script words in `text`; other characters pass through."""
    for ar, lat in _PUNCT.items():
        text = text.replace(ar, lat)
    return _ARABIC_WORD.sub(lambda m: _word(m.group()), text)
