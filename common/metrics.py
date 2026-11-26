"""Official-style metrics and the LOCAL composite proxy.

OFFICIAL weights (problem statement):
  S = 0.40 F_A + 0.15 F_B + 0.10 F_C + 0.10 F_domain + 0.10 F_robust + 0.05 Calibration + 0.10 F_novel

F_A, F_B, F_C and Calibration can be computed locally on OOF predictions with the same definitions.
F_domain / F_robust / F_novel are scored by the organizers on HIDDEN subsets. Locally we only have
PROXIES (see common/stress.py). A composite that uses proxies is a LOCAL PROXY, never a Kaggle score.

Assumptions to verify against the official evaluation page:
  - Task B: per-technique binary F1 with zero_division=0, averaged over T1..T6.
  - Calibration: 1 - mean((confidence - 1[pred_A == y_A])^2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .labels import TASK_A_CLASSES, TASK_C_CLASSES, TECHNIQUES

WEIGHTS = {"F_A": 0.40, "F_B": 0.15, "F_C": 0.10, "F_domain": 0.10, "F_robust": 0.10,
           "Calibration": 0.05, "F_novel": 0.10}


def macro_f1(y_true, y_pred, labels=TASK_A_CLASSES) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def task_b_f1(Y_true, Y_pred) -> float:
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    return float(np.mean([f1_score(Y_true[:, j], Y_pred[:, j], zero_division=0) for j in range(Y_true.shape[1])]))


def brier(confidence, correct) -> float:
    c = np.asarray(confidence, float)
    return float(np.mean((c - np.asarray(correct, float)) ** 2))


def calibration_score(confidence, correct) -> float:
    return 1.0 - brier(confidence, correct)


def per_class(y_true, y_pred, labels) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {str(l): {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                     "f1": round(float(f[i]), 4), "support": int(s[i])} for i, l in enumerate(labels)}


def per_technique(Y_true, Y_pred) -> dict:
    out = {}
    for j, t in enumerate(TECHNIQUES):
        p, r, f, _ = precision_recall_fscore_support(Y_true[:, j], Y_pred[:, j], average="binary", zero_division=0)
        out[t] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
                  "support": int(np.asarray(Y_true)[:, j].sum())}
    return out


def evaluate(yA, yB, yC, pA, pB, pC, confidence=None, technique_threshold=0.5) -> dict:
    """Full evaluation from probability matrices (contract shapes, see prediction_io.py)."""
    predA, predC = pA.argmax(1), pC.argmax(1)
    predB = (pB >= technique_threshold).astype(int)
    conf = pA.max(1) if confidence is None else np.asarray(confidence)
    correct = (predA == yA).astype(int)
    return {
        "F_A": round(macro_f1(yA, predA), 4),
        "F_B": round(task_b_f1(yB, predB), 4),
        "F_C": round(macro_f1(yC, predC, TASK_C_CLASSES), 4),
        "Brier": round(brier(conf, correct), 4),
        "Calibration": round(calibration_score(conf, correct), 4),
        "accuracy_A": round(float(correct.mean()), 4),
        "per_class_A": per_class(yA, predA, TASK_A_CLASSES),
        "per_technique_B": per_technique(np.asarray(yB), predB),
        "per_action_C": per_class(yC, predC, TASK_C_CLASSES),
        "confusion_A": confusion_matrix(yA, predA, labels=TASK_A_CLASSES).tolist(),
        "confusion_C": confusion_matrix(yC, predC, labels=TASK_C_CLASSES).tolist(),
    }


def composite_proxy(scores: dict) -> dict:
    """Weighted composite using whatever components are present. Missing hidden-subset proxies are
    replaced by F_A and listed under 'imputed' so nobody mistakes the number for a measured one."""
    used, imputed = {}, []
    for k in WEIGHTS:
        if scores.get(k) is not None:
            used[k] = float(scores[k])
        else:
            used[k] = float(scores["F_A"])
            imputed.append(k)
    value = sum(WEIGHTS[k] * used[k] for k in WEIGHTS)
    return {"composite_proxy": round(value, 4), "components": {k: round(v, 4) for k, v in used.items()},
            "imputed_with_F_A": imputed,
            "note": "LOCAL PROXY. F_domain/F_robust/F_novel come from local stress tests, not hidden subsets."}


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
