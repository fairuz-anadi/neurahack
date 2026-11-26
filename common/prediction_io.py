"""Mandatory prediction contract for every model family.

artifacts/<model_name>/
    oof_A_<model>.npy  oof_B_<model>.npy  oof_C_<model>.npy     (n_train, 6), train row order
    test_A_<model>.npy test_B_<model>.npy test_C_<model>.npy    (n_test, 6),  test row order
    oof_ids.npy  test_ids.npy  metadata.json

A and C rows are categorical distributions (sum to 1). B columns are independent probabilities (T1..T6).
Test files are optional locally (test.csv only exists on Kaggle).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import REPO_ROOT
from .labels import N_A, N_B, N_C, TASK_A_CLASSES, TASK_C_CLASSES, TECHNIQUES

SHAPES = {"A": N_A, "B": N_B, "C": N_C}
COLUMN_ORDER = {"A": TASK_A_CLASSES, "B": TECHNIQUES, "C": TASK_C_CLASSES}


class ContractError(ValueError):
    pass


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (sha or "no-commit") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def validate_probs(task: str, P, n_rows: int | None = None, name: str = "") -> None:
    P = np.asarray(P)
    where = f"{name} task {task}"
    if P.ndim != 2 or P.shape[1] != SHAPES[task]:
        raise ContractError(f"{where}: expected shape (n, {SHAPES[task]}), got {P.shape}")
    if n_rows is not None and P.shape[0] != n_rows:
        raise ContractError(f"{where}: expected {n_rows} rows, got {P.shape[0]}")
    if not np.isfinite(P).all():
        raise ContractError(f"{where}: contains NaN/inf")
    if (P < -1e-6).any() or (P > 1 + 1e-6).any():
        raise ContractError(f"{where}: values outside [0, 1]")
    if task in ("A", "C"):
        s = P.sum(1)
        if not np.allclose(s, 1.0, atol=1e-3):
            raise ContractError(f"{where}: rows must sum to 1 (min {s.min():.4f}, max {s.max():.4f})")


def save_predictions(model_name: str, out_root: Path, oof: dict, oof_ids, metadata: dict,
                     test: dict | None = None, test_ids=None) -> Path:
    d = Path(out_root) / model_name
    d.mkdir(parents=True, exist_ok=True)
    oof_ids = np.asarray(oof_ids)
    for t in "ABC":
        validate_probs(t, oof[t], len(oof_ids), f"oof {model_name}")
        np.save(d / f"oof_{t}_{model_name}.npy", np.asarray(oof[t], dtype=np.float32))
    np.save(d / "oof_ids.npy", oof_ids)
    if test is not None:
        test_ids = np.asarray(test_ids)
        for t in "ABC":
            validate_probs(t, test[t], len(test_ids), f"test {model_name}")
            np.save(d / f"test_{t}_{model_name}.npy", np.asarray(test[t], dtype=np.float32))
        np.save(d / "test_ids.npy", test_ids)
    meta = {
        "model_name": model_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "column_order": {k: [str(c) for c in v] for k, v in COLUMN_ORDER.items()},
        "has_test": test is not None,
        **metadata,
    }
    for req in ("config", "seed", "dataset_fingerprint", "fold_fingerprint", "metrics"):
        if req not in meta:
            raise ContractError(f"metadata missing required key {req!r}")
    with open(d / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return d


def load_predictions(model_dir: Path, split: str = "oof", expected_ids=None) -> dict:
    """Load one model's matrices. If expected_ids is given, rows are re-ordered to match it by id
    (never by position) and a mismatch in the id set raises ContractError."""
    d = Path(model_dir)
    name = json.loads((d / "metadata.json").read_text(encoding="utf-8"))["model_name"]
    ids = np.load(d / f"{split}_ids.npy", allow_pickle=False)
    if len(np.unique(ids)) != len(ids):
        raise ContractError(f"{name}: duplicate ids in {split}_ids.npy")
    out = {t: np.load(d / f"{split}_{t}_{name}.npy") for t in "ABC"}
    for t in "ABC":
        validate_probs(t, out[t], len(ids), f"{split} {name}")
    if expected_ids is not None:
        expected_ids = np.asarray(expected_ids)
        if len(expected_ids) != len(ids) or set(expected_ids.tolist()) != set(ids.tolist()):
            raise ContractError(f"{name}: {split} ids do not match the reference ids")
        pos = {v: i for i, v in enumerate(ids.tolist())}
        order = np.array([pos[v] for v in expected_ids.tolist()])
        out = {t: out[t][order] for t in out}
        ids = expected_ids
    out["ids"] = ids
    return out


def check_metadata_compatible(metas: list[dict]) -> None:
    """Models can only be ensembled if they used the same data and folds."""
    for key in ("dataset_fingerprint", "fold_fingerprint"):
        vals = {m.get(key) for m in metas}
        if len(vals) > 1:
            raise ContractError(f"incompatible {key} across models: {vals}")
