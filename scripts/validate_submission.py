"""Validate a submission file against sample_submission.csv and the format rules.

    python scripts/validate_submission.py outputs/submission.csv
"""
import _bootstrap  # noqa: F401

import sys

import pandas as pd

from common.data_loader import load_config, load_sample_submission
from common.submission import validate_submission


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/submission.csv"
    cfg = load_config()
    sub = pd.read_csv(path, keep_default_na=False)
    sample = load_sample_submission(cfg)
    problems = validate_submission(sub, sample, cfg["submission"]["nonfraud_classes"])
    if problems:
        print("INVALID:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"VALID: {len(sub)} rows, columns {list(sub.columns)}")
    print("label share:", sub["label"].value_counts(normalize=True).sort_index().round(4).to_dict())
    print("action share:", sub["action"].value_counts(normalize=True).sort_index().round(4).to_dict())
    print("NONE share:", round(float((sub["techniques"] == "NONE").mean()), 4),
          "| confidence mean:", round(float(sub["confidence"].mean()), 4))


if __name__ == "__main__":
    main()
