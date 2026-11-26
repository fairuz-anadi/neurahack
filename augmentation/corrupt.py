"""Deterministic adversarial text corruption, modelled on the noise types visible in the TRAINING data
(homoglyphs like 'sаlаry', zero-width spaces, injected junk tokens like 'xkcdpl', letter repeats like
'PIIIN', digit-for-letter swaps like 'P1N', typos, emoji injection).

Used by:
  - common/stress.py  -> robustness proxy (corrupt VALIDATION text only, never the training fold)
  - Samprity's adversarial augmentation (corrupt TRAINING-fold text only)
"""
from __future__ import annotations

import random
import string

_HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і", "y": "у"}
_LEET = {"i": "1", "o": "0", "e": "3", "a": "4", "s": "5", "l": "1"}
_EMOJI = ["🎁", "🔥", "💰", "✅", "⚠️", "👉"]
_ZW = ["​", "‌", "‍"]


def _junk(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 10)))


def corrupt(text: str, seed: int, strength: float = 0.15) -> str:
    """Apply random character/token corruptions. Same (text, seed) -> same output."""
    rng = random.Random(f"{seed}:{text}")
    chars = list(text)
    out = []
    for ch in chars:
        low = ch.lower()
        r = rng.random()
        if ch.isalpha() and ch.isascii() and r < strength:
            op = rng.randint(0, 5)
            if op == 0 and low in _HOMO:
                out.append(_HOMO[low]); continue
            if op == 1 and low in _LEET:
                out.append(_LEET[low]); continue
            if op == 2:
                out.append(ch * rng.randint(2, 4)); continue
            if op == 3:
                continue  # drop char
            if op == 4:
                out.append(ch.swapcase()); continue
            out.append(ch + rng.choice(_ZW)); continue
        out.append(ch)
    s = "".join(out)
    toks = s.split(" ")
    if rng.random() < 0.6:
        toks.insert(rng.randint(0, len(toks)), _junk(rng))
    if rng.random() < 0.3:
        toks.insert(rng.randint(0, len(toks)), rng.choice(_EMOJI))
    return " ".join(toks)
