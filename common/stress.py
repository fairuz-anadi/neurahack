"""Shared CV runner + local stress tests (proxies for the hidden F_domain / F_robust / F_novel subsets).

A model family plugs in with ONE function:

    def fit_predict(train_df, valid_df) -> {"A": (n_valid,6), "B": (n_valid,6), "C": (n_valid,6)}

train_df / valid_df are slices of the canonical DataFrame from common.data_loader.load_train
(use raw_text or normalized_text as you like; labels are in label / T1..T6 / action).

Proxy definitions (LOCAL, approximate):
  F_robust_proxy  Task A macro F1 when VALIDATION text is corrupted by augmentation.corrupt (main folds)
  F_novel_proxy   Task A macro F1 on folds_novel.csv (looser grouping, whole scenarios held out)
  F_domain_proxy  Task A macro F1 on folds_domain.csv (all rows of a brand held out together)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from augmentation.corrupt import corrupt
from .data_loader import get_targets
from .folds import load_folds, load_repeat_folds
from .labels import N_A, N_B, N_C
from .metrics import evaluate, macro_f1
from .preprocessing import normalize


def corrupt_df(df: pd.DataFrame, seed: int = 42, strength: float = 0.15) -> pd.DataFrame:
    out = df.copy()
    out["raw_text"] = [corrupt(t, seed, strength) for t in df["raw_text"]]
    out["normalized_text"] = [normalize(t) for t in out["raw_text"]]
    return out


def run_cv(fit_predict, train: pd.DataFrame, fold: np.ndarray, valid_transform=None) -> dict:
    oof = {"A": np.zeros((len(train), N_A)), "B": np.zeros((len(train), N_B)), "C": np.zeros((len(train), N_C))}
    for k in sorted(np.unique(fold)):
        tr, va = train[fold != k], train[fold == k]
        if valid_transform is not None:
            va = valid_transform(va)
        pred = fit_predict(tr, va)
        for t in oof:
            oof[t][fold == k] = pred[t]
    return oof


def slice_scores(train: pd.DataFrame, predA: np.ndarray) -> dict:
    yA = train["label"].to_numpy()
    out = {}
    for name, mask in {
        **{f"script={s}": (train.script == s).to_numpy() for s in sorted(train.script.unique())},
        "conversation": train.is_conversation.to_numpy(),
        "single_message": (~train.is_conversation).to_numpy(),
    }.items():
        if mask.sum() >= 3:
            out[name] = {"n": int(mask.sum()), "accuracy_A": round(float((predA[mask] == yA[mask]).mean()), 4),
                         "macro_f1_A_present_classes": round(macro_f1(yA[mask], predA[mask],
                                                                      sorted(np.unique(yA[mask]))), 4)}
    return out


def run_stress(fit_predict, train: pd.DataFrame, seed: int = 42, repeats: bool = True) -> tuple[dict, dict]:
    """Returns (report, oof_main). report holds official-style OOF metrics + proxy scores."""
    t0 = time.time()
    yA, yB, yC = get_targets(train)
    fold = load_folds(train, "folds")
    oof = run_cv(fit_predict, train, fold)
    report = {"main": evaluate(yA, yB, yC, oof["A"], oof["B"], oof["C"])}

    rob = run_cv(fit_predict, train, fold, valid_transform=lambda d: corrupt_df(d, seed))
    report["F_robust_proxy"] = round(macro_f1(yA, rob["A"].argmax(1)), 4)

    nov = run_cv(fit_predict, train, load_folds(train, "folds_novel"))
    report["F_novel_proxy"] = round(macro_f1(yA, nov["A"].argmax(1)), 4)

    dom = run_cv(fit_predict, train, load_folds(train, "folds_domain"))
    report["F_domain_proxy"] = round(macro_f1(yA, dom["A"].argmax(1)), 4)

    report["slices_A"] = slice_scores(train, oof["A"].argmax(1))

    if repeats:
        fa = [macro_f1(yA, run_cv(fit_predict, train, f)["A"].argmax(1)) for f in load_repeat_folds(train)]
        report["repeated_F_A"] = {"mean": round(float(np.mean(fa)), 4), "std": round(float(np.std(fa)), 4),
                                  "values": [round(x, 4) for x in fa]}
    report["runtime_sec"] = round(time.time() - t0, 1)
    return report, oof
