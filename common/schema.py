"""Schema checks for the official files. Columns come from the real files (inspected 26 Nov):
    sample.csv            id,text,label,techniques,action
    sample_submission.csv id,label,techniques,action,confidence
    test.csv              not available locally; expected id + one text column (auto-detected)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import TASK_A_CLASSES, TASK_C_CLASSES, LabelError, parse_techniques

TRAIN_LABEL_COLS = ["label", "techniques", "action"]
SUBMISSION_COLS = ["id", "label", "techniques", "action", "confidence"]


class SchemaError(ValueError):
    pass


def find_text_col(df: pd.DataFrame, candidates, id_col="id") -> str:
    for c in candidates:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if c != id_col and not pd.api.types.is_numeric_dtype(df[c])]
    if len(obj) == 1:
        return obj[0]
    raise SchemaError(
        f"cannot identify the text column. Columns: {list(df.columns)}. "
        f"Add the right name to data.text_col_candidates in configs/base.yaml."
    )


def check_ids(df: pd.DataFrame, id_col: str, name: str) -> list[str]:
    if id_col not in df.columns:
        raise SchemaError(f"{name}: missing id column {id_col!r}; columns are {list(df.columns)}")
    problems = []
    if df[id_col].isna().any():
        problems.append(f"{name}: {int(df[id_col].isna().sum())} missing ids")
    dup = df[id_col][df[id_col].duplicated(keep=False)]
    if len(dup):
        problems.append(f"{name}: duplicate ids {sorted(dup.unique().tolist())[:10]}")
    return problems


def check_train_labels(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty = valid). Never drops rows."""
    problems = []
    missing = [c for c in TRAIN_LABEL_COLS if c not in df.columns]
    if missing:
        raise SchemaError(f"train file missing label columns {missing}; columns are {list(df.columns)}")
    for col, allowed in (("label", TASK_A_CLASSES), ("action", TASK_C_CLASSES)):
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = df.loc[vals.isna() | ~vals.isin(allowed), "id"].tolist()
        if bad:
            problems.append(f"{col}: invalid or missing values at ids {bad[:10]}")
        elif not np.allclose(vals, vals.round()):
            problems.append(f"{col}: non-integer values")
    for rid, v in zip(df["id"], df["techniques"]):
        try:
            parse_techniques(v)
        except LabelError as e:
            problems.append(f"techniques at id {rid}: {e}")
    return problems
