import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_data(tmp_path):
    """Synthetic fixture files written for tests only (no real test data exists locally)."""
    rows = [
        (1, "Your order #1 is out for delivery today.", 0, "NONE", 0),
        (2, "bKash: apnar PIN ta pathan, account block hobe", 1, "T1|T2|T5", 1),
        (3, "You won Tk 5000! Pay Tk 200 fee to claim", 2, "T4|T3", 2),
        (4, "Part-time job: like videos, earn Tk 500. Join Telegram", 3, "T6", 4),
        (5, "Parcel held. Pay Tk 45 fee: hxxps://x[.]top", 4, "T1|T3", 2),
        (6, "Eid sale 30% off all items", 5, "NONE", 0),
    ] * 3
    df = pd.DataFrame(rows, columns=["id", "text", "label", "techniques", "action"])
    df["id"] = np.arange(1, len(df) + 1)
    df.to_csv(tmp_path / "sample.csv", index=False)
    test = pd.DataFrame({"id": np.arange(101, 107), "text": df.text[:6].tolist()})
    test.to_csv(tmp_path / "test.csv", index=False)
    ss = pd.DataFrame({"id": test.id, "label": 0, "techniques": "NONE", "action": 0, "confidence": 0.5})
    ss.to_csv(tmp_path / "sample_submission.csv", index=False)
    cfg = {
        "seed": 42,
        "data": {"dir": str(tmp_path), "fallback_dirs": [], "train_file": "sample.csv", "test_file": "test.csv",
                 "sample_submission_file": "sample_submission.csv", "id_col": "id",
                 "text_col_candidates": ["text"], "turn_separator": "|||"},
        "output": {"artifacts_dir": str(tmp_path / "art"), "outputs_dir": str(tmp_path / "out")},
        "folds": {"max_splits": 3, "near_dup_threshold": 0.8, "novel_rows_per_cluster": 3, "n_repeats": 2},
        "submission": {"technique_threshold": 0.5, "nonfraud_classes": [0], "fraud_requires_technique": False},
    }
    return tmp_path, cfg
