"""Load a synthetic labeled CSV (id,text,label,techniques,action) into the canonical training frame.

Synthetic rows are generated from the official class definitions only (models/finetune/llm.py), never from
sample.csv rows, so the same file can be added to every training fold. They are never used for evaluation.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from common.data_loader import _canonicalize, load_config
from common.labels import TECHNIQUES, binary_to_techniques, techniques_to_binary
from common.preprocessing import DEFAULT
from common.schema import check_train_labels


@lru_cache(maxsize=4)
def load_synthetic(path: str) -> pd.DataFrame:
    cfg = load_config()
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"text": str, "id": str})
    raw = raw.dropna(subset=["text"]).reset_index(drop=True)
    problems = check_train_labels(raw)
    if problems:
        raise ValueError(f"synthetic file {path} has invalid labels: {problems[:5]}")
    out = _canonicalize(raw, cfg, "text", DEFAULT)
    out["label"] = raw["label"].astype(int).to_numpy()
    out["action"] = raw["action"].astype(int).to_numpy()
    tb = techniques_to_binary(raw["techniques"])
    out["techniques"] = [binary_to_techniques(r) for r in tb]
    for j, t in enumerate(TECHNIQUES):
        out[t] = tb[:, j]
    out["is_synthetic"] = True
    return out


@lru_cache(maxsize=2)
def _test_frame() -> pd.DataFrame:
    from common.data_loader import load_test
    return load_test(load_config())


@lru_cache(maxsize=4)
def load_pseudo(path: str, min_conf: float = 0.0) -> pd.DataFrame:
    """Teacher pseudo-labels (id,label,techniques,action[,conf] for TEST ids) joined to the test text by code.
    Rows whose teacher label confidence is below min_conf are dropped (quality filter).
    Kaggle only (needs test.csv). Pseudo rows are training data only; they are never scored."""
    lab = pd.read_csv(path, keep_default_na=False)
    if "conf" in lab.columns and min_conf > 0:
        lab = lab[lab["conf"].astype(float) >= min_conf]
    lab = lab[["id", "label", "techniques", "action"]]
    test = _test_frame()
    merged = test.merge(lab, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(lab):
        raise ValueError(f"{len(lab) - len(merged)} pseudo-label ids not found in test.csv")
    out = merged.copy()
    out["label"] = out["label"].astype(int)
    out["action"] = out["action"].astype(int)
    tb = techniques_to_binary(out["techniques"])
    out["techniques"] = [binary_to_techniques(r) for r in tb]
    for j, tt in enumerate(TECHNIQUES):
        out[tt] = tb[:, j]
    out["is_synthetic"] = False
    out["is_pseudo"] = True
    return out.reset_index(drop=True)


@lru_cache(maxsize=4)
def load_external(path: str) -> pd.DataFrame:
    """External public data (models/finetune/external.py output). label=-1 means partial: Task A is only known to be
    in `allowed`; techniques=='' / action==-1 mean Task B / C unknown (masked in the loss)."""
    cfg = load_config()
    raw = pd.read_csv(path, keep_default_na=False, dtype={"id": str, "text": str, "allowed": str, "techniques": str})
    raw = raw[raw["text"].str.len() > 0].reset_index(drop=True)
    out = _canonicalize(raw, cfg, "text", DEFAULT)
    lab = raw["label"].astype(int).to_numpy()
    out["label"] = lab
    out["action"] = raw["action"].astype(int).clip(lower=0).to_numpy()
    tech = raw["techniques"].fillna("")
    b_known = (tech != "").to_numpy()
    tb = techniques_to_binary(tech.where(b_known, "NONE"))
    out["techniques"] = [binary_to_techniques(r) for r in tb]
    for j, tt in enumerate(TECHNIQUES):
        out[tt] = tb[:, j]
    masks = []
    for y_, allowed in zip(lab, raw["allowed"].fillna("")):
        mk = np.zeros(6, dtype=np.float32)
        if y_ >= 0:
            mk[y_] = 1.0
        else:
            for a in str(allowed).split("|"):
                mk[int(a)] = 1.0
        masks.append(mk)
    out["a_mask"] = masks
    out["b_known"] = b_known
    out["c_known"] = (raw["action"].astype(int) >= 0).to_numpy()
    out["is_external"] = True
    return out
