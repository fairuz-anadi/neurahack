"""Conservative, configurable text normalization.

Design rules (see docs/TEAM_CONTRACT.md):
- raw_text is never modified; normalized_text is a separate view.
- Punctuation, emojis, digits, URLs, phone masks and Bangla combining marks are kept.
- ZWJ / ZWNJ are kept when they sit between Bangla characters (they shape conjuncts);
  elsewhere they are treated as adversarial invisible characters and removed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

BANGLA_RE = re.compile(r"[ঀ-৿]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Invisible characters that never carry meaning in this data.
_ALWAYS_INVISIBLE = "​⁠﻿­᠎‎‏‪‫‬‭‮"
_INVISIBLE_RE = re.compile(f"[{_ALWAYS_INVISIBLE}]")
_ZWJ_ZWNJ = "‌‍"

# Cyrillic / Greek letters that look like Latin letters (seen in training data, e.g. 'sаlаry').
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ո": "n", "ս": "u", "ν": "v", "ԝ": "w",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "τ": "t", "υ": "u",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_HOMO_TABLE = str.maketrans(HOMOGLYPHS)

_REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)  # 3+ identical letters -> 2
_WS_RE = re.compile(r"[ \t\r\f\v  - 　]+")
_DEFANG = [(re.compile(r"\[\.\]"), "."), (re.compile(r"\bhxxp(s?)://", re.I), r"http\1://")]


@dataclass(frozen=True)
class NormConfig:
    unicode_form: str = "NFC"        # NFC keeps Bangla vowel signs intact; NFKC also folds fullwidth
    remove_invisible: bool = True
    homoglyphs: bool = True
    lowercase: bool = True           # Latin only; Bangla has no case
    squeeze_repeats: bool = False    # optional view: 'PIIIN' -> 'PIIN'
    refang_urls: bool = False        # optional view: 'hxxps://x[.]top' -> 'https://x.top'
    empty_token: str = ""            # what a fully-empty record becomes (e.g. '<empty>' for tokenizers)


DEFAULT = NormConfig()
AGGRESSIVE = NormConfig(unicode_form="NFKC", squeeze_repeats=True, refang_urls=True)


def _strip_zw_outside_bangla(text: str) -> str:
    if not any(c in text for c in _ZWJ_ZWNJ):
        return text
    out = []
    for i, ch in enumerate(text):
        if ch in _ZWJ_ZWNJ:
            prev_b = i > 0 and BANGLA_RE.match(text[i - 1])
            next_b = i + 1 < len(text) and BANGLA_RE.match(text[i + 1])
            if prev_b and next_b:
                out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def _fold_homoglyphs(text: str) -> str:
    # Only fold inside tokens that also contain real Latin letters, so genuine Cyrillic/Greek words
    # (not expected here, but possible) are left alone.
    def fix(m: re.Match) -> str:
        tok = m.group(0)
        return tok.translate(_HOMO_TABLE) if LATIN_RE.search(tok) else tok

    return re.sub(r"\S+", fix, text)


def normalize(text, cfg: NormConfig = DEFAULT) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return cfg.empty_token
    s = unicodedata.normalize(cfg.unicode_form, str(text))
    if cfg.remove_invisible:
        s = _INVISIBLE_RE.sub("", s)
        s = _strip_zw_outside_bangla(s)
    if cfg.homoglyphs:
        s = _fold_homoglyphs(s)
    if cfg.refang_urls:
        for pat, rep in _DEFANG:
            s = pat.sub(rep, s)
    if cfg.lowercase:
        s = s.lower()  # str.lower leaves Bangla untouched
    if cfg.squeeze_repeats:
        s = _REPEAT_RE.sub(r"\1\1", s)
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\n+", " \n ", s).strip()
    s = re.sub(r" {2,}", " ", s)
    return s if s else cfg.empty_token


def script_profile(text: str) -> dict:
    """Script shares. NOTE: 'latin' cannot tell English from Banglish; only the script is detected."""
    s = "" if text is None else str(text)
    n_b = len(BANGLA_RE.findall(s))
    n_l = len(LATIN_RE.findall(s))
    tot = n_b + n_l
    if tot == 0:
        script = "none"
    elif n_b / tot >= 0.9:
        script = "bangla"
    elif n_l / tot >= 0.9:
        script = "latin"
    else:
        script = "mixed"
    non_latin_letters = sum(1 for c in s if c.isalpha() and not LATIN_RE.match(c) and not BANGLA_RE.match(c))
    invis = len(_INVISIBLE_RE.findall(s)) + sum(s.count(c) for c in _ZWJ_ZWNJ)
    return {
        "script": script,
        "bangla_chars": n_b,
        "latin_chars": n_l,
        "other_letters": non_latin_letters,      # homoglyph signal (Cyrillic/Greek)
        "invisible_chars": invis,
        "has_url": bool(re.search(r"(https?|hxxps?)://|\[\.\]|\bwww\.", s, re.I)),
    }


def is_noise_only(text: str) -> bool:
    """Heuristic flag: no Bangla, no digits/URL, and no Latin token with a vowel pattern of a real word.
    Used only for EDA/diagnostics, never to drop rows."""
    s = normalize(text)
    if not s or BANGLA_RE.search(s) or re.search(r"\d|://|\[\.\]", s):
        return False
    toks = re.findall(r"[a-z]+", s)
    if not toks:
        return True
    return all(is_junk_token(t) for t in toks)


def is_junk_token(tok: str) -> bool:
    """Keyboard-mash token: a run of 5+ consonants, or 4+ letters with no vowel at all."""
    t = tok.lower()
    return bool(re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", t)) or (len(t) >= 4 and not re.search(r"[aeiouy]", t))
