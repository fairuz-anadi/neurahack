"""Label constants and technique encoding. Single source of truth for column order."""
from __future__ import annotations

import numpy as np

TASK_A_CLASSES = [0, 1, 2, 3, 4, 5]
TASK_A_NAMES = {
    0: "Legitimate",
    1: "MFS/OTP scam",
    2: "Prize/lottery scam",
    3: "Fake job/investment",
    4: "Delivery/parcel phishing",
    5: "Promotional spam",
}
TECHNIQUES = ["T1", "T2", "T3", "T4", "T5", "T6"]
TECHNIQUE_NAMES = {
    "T1": "Authority or organization impersonation",
    "T2": "Credential or sensitive-information harvesting",
    "T3": "Advance-fee or payment deception",
    "T4": "Prize or reward baiting",
    "T5": "Urgency, pressure, trust-building or social engineering",
    "T6": "Task-based earning, fake job or investment manipulation",
}
TASK_C_CLASSES = [0, 1, 2, 3, 4, 5]
TASK_C_NAMES = {
    0: "No suspicious action",
    1: "Disclose confidential information",
    2: "Transfer money or pay a fee",
    3: "Visit a link or website",
    4: "Install an application, join a group or complete a task",
    5: "Contact, call or reply",
}
NONE_TOKEN = "NONE"
SEP = "|"
N_A, N_B, N_C = len(TASK_A_CLASSES), len(TECHNIQUES), len(TASK_C_CLASSES)


class LabelError(ValueError):
    pass


def parse_techniques(value) -> list[str]:
    """'T1|T5' -> ['T1','T5']; 'NONE' -> []. Raises LabelError on unknown codes or duplicates."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise LabelError("missing techniques value (expected 'NONE' or T-codes)")
    s = str(value).strip()
    if s.upper() == NONE_TOKEN:
        return []
    codes = [c.strip().upper() for c in s.split(SEP)]
    bad = [c for c in codes if c not in TECHNIQUES]
    if bad:
        raise LabelError(f"unknown technique code(s) {bad} in {value!r}")
    if len(set(codes)) != len(codes):
        raise LabelError(f"duplicate technique code in {value!r}")
    return codes


def techniques_to_binary(values) -> np.ndarray:
    """Iterable of technique strings -> (n, 6) int8 matrix in T1..T6 order."""
    values = list(values)
    out = np.zeros((len(values), N_B), dtype=np.int8)
    for i, v in enumerate(values):
        for c in parse_techniques(v):
            out[i, TECHNIQUES.index(c)] = 1
    return out


def binary_to_techniques(row) -> str:
    """Binary/boolean vector in T1..T6 order -> canonical string ('T1|T5' or 'NONE')."""
    codes = [TECHNIQUES[j] for j in range(N_B) if row[j]]
    return SEP.join(codes) if codes else NONE_TOKEN
