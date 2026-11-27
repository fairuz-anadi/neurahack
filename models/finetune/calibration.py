"""Task D confidence = calibrated max Task A probability.

Temperature scaling (argmax unchanged). The temperature is chosen to minimize the Brier score of
confidence vs. correctness, which is exactly the competition's Calibration component (1 - Brier).
`crossfit_confidence` fits T on the other folds' OOF rows and applies it to each fold, so reported
calibration is never fit on the rows it is scored on.
"""
from __future__ import annotations

import numpy as np

T_GRID = np.exp(np.linspace(np.log(0.2), np.log(5.0), 81))


def apply_temperature(pA: np.ndarray, T: float) -> np.ndarray:
    logp = np.log(np.clip(pA, 1e-12, 1.0)) / T
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def brier_conf(pA: np.ndarray, y: np.ndarray) -> float:
    correct = (pA.argmax(1) == y).astype(float)
    return float(np.mean((pA.max(1) - correct) ** 2))


def fit_temperature(pA: np.ndarray, y: np.ndarray) -> float:
    scores = [brier_conf(apply_temperature(pA, T), y) for T in T_GRID]
    return float(T_GRID[int(np.argmin(scores))])


def crossfit_confidence(pA: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, dict]:
    conf = np.zeros(len(y))
    temps = {}
    for k in sorted(np.unique(fold)):
        tr, va = fold != k, fold == k
        T = fit_temperature(pA[tr], y[tr])
        temps[int(k)] = round(T, 3)
        conf[va] = apply_temperature(pA[va], T).max(1)
    return conf, temps
