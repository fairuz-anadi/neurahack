"""Builds and validates submission.csv: id,label,techniques,action,confidence.

Predictions only come from probability matrices produced by model code. Nothing here edits a
prediction by hand; the only post-processing is the official consistency rule
(non-fraudulent -> NONE) and optional, config-controlled rules documented in configs/base.yaml.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .labels import NONE_TOKEN, TECHNIQUES, binary_to_techniques
from .prediction_io import validate_probs
from .schema import SUBMISSION_COLS

_TECH_RE = re.compile(r"^(NONE|T[1-6](\|T[1-6])*)$")


def build_submission(test_ids, pA, pB, pC, confidence=None, technique_threshold=0.5,
                     nonfraud_classes=(0,), fraud_requires_technique=False) -> pd.DataFrame:
    n = len(test_ids)
    for t, P in (("A", pA), ("B", pB), ("C", pC)):
        validate_probs(t, P, n, "submission input")
    label = np.asarray(pA).argmax(1).astype(int)
    action = np.asarray(pC).argmax(1).astype(int)
    thr = np.broadcast_to(np.asarray(technique_threshold, float), (len(TECHNIQUES),))
    techB = np.asarray(pB) >= thr
    if fraud_requires_technique:
        fraud_empty = ~np.isin(label, nonfraud_classes) & ~techB.any(1)
        techB[fraud_empty, np.asarray(pB)[fraud_empty].argmax(1)] = True
    techB[np.isin(label, nonfraud_classes)] = False
    conf = np.asarray(pA).max(1) if confidence is None else np.asarray(confidence, float)
    conf = np.clip(conf, 0.0, 1.0)
    return pd.DataFrame({
        "id": np.asarray(test_ids),
        "label": label,
        "techniques": [binary_to_techniques(r) for r in techB],
        "action": action,
        "confidence": conf.round(6),
    })


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame | None = None,
                        nonfraud_classes=(0,)) -> list[str]:
    """Returns a list of problems (empty list = valid)."""
    p = []
    if list(sub.columns) != SUBMISSION_COLS:
        return [f"columns must be exactly {SUBMISSION_COLS}, got {list(sub.columns)}"]
    if sub.isna().any().any():
        p.append(f"missing values in columns {sub.columns[sub.isna().any()].tolist()}")
    if sub["id"].duplicated().any():
        p.append(f"{int(sub['id'].duplicated().sum())} duplicate ids")
    for col in ("label", "action"):
        v = pd.to_numeric(sub[col], errors="coerce")
        if not pd.api.types.is_integer_dtype(sub[col]):
            p.append(f"{col} must be integer dtype, got {sub[col].dtype}")
        if v.isna().any() or not v.between(0, 5).all():
            p.append(f"{col} has values outside 0..5")
    bad_t = ~sub["techniques"].astype(str).str.match(_TECH_RE)
    if bad_t.any():
        p.append(f"{int(bad_t.sum())} malformed techniques values, e.g. {sub.loc[bad_t, 'techniques'].head(3).tolist()}")
    else:
        dup = sub["techniques"].astype(str).map(lambda s: len(s.split("|")) != len(set(s.split("|"))))
        if dup.any():
            p.append(f"{int(dup.sum())} techniques values repeat a code")
    nf = sub["label"].isin(nonfraud_classes) & (sub["techniques"] != NONE_TOKEN)
    if nf.any():
        p.append(f"{int(nf.sum())} non-fraud rows (label in {list(nonfraud_classes)}) have techniques != NONE")
    c = pd.to_numeric(sub["confidence"], errors="coerce")
    if c.isna().any() or not np.isfinite(c).all() or not c.between(0, 1).all():
        p.append("confidence must be finite and within [0, 1]")
    if sample is not None:
        if list(sample.columns) != SUBMISSION_COLS:
            p.append(f"sample_submission columns differ from expected: {list(sample.columns)}")
        if len(sub) != len(sample):
            p.append(f"row count {len(sub)} != sample_submission {len(sample)}")
        if set(sub["id"].tolist()) != set(sample["id"].tolist()):
            missing = len(set(sample["id"]) - set(sub["id"]))
            extra = len(set(sub["id"]) - set(sample["id"]))
            p.append(f"id set differs from sample_submission: {missing} missing, {extra} extra")
    return p


def write_submission(sub: pd.DataFrame, path, sample: pd.DataFrame | None = None) -> pd.DataFrame:
    """Order rows like sample_submission (if given), validate, then write. Raises on any problem."""
    if sample is not None and set(sub["id"].tolist()) == set(sample["id"].tolist()):
        sub = sample[["id"]].merge(sub, on="id", how="left", validate="one_to_one")
    problems = validate_submission(sub, sample)
    if problems:
        raise ValueError("submission invalid:\n  " + "\n  ".join(problems))
    sub.to_csv(path, index=False, lineterminator="\n")
    return sub
