"""Turn a run's saved test arrays into a validated submission.csv (Kaggle only: needs test arrays).

    python -m models.finetune.make_submission --name ft_e5s_corrupt [--out /kaggle/working/submission.csv]

Uses common.submission (official 0/5 convention, technique threshold from configs/base.yaml) and the run's
calibrated Task D confidence (test_conf_<name>.npy). Prints aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from common.data_loader import load_config, load_sample_submission, output_dirs
from common.submission import build_submission, validate_submission, write_submission


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    art_dir, out_dir = output_dirs(cfg)
    d, n = art_dir / args.name, args.name
    if not (d / "test_ids.npy").exists():
        sys.exit(f"{d} has no test arrays; run the opt-in test inference first")
    L = lambda k: np.load(d / f"{k}.npy")
    conf = L(f"test_conf_{n}") if (d / f"test_conf_{n}.npy").exists() else None
    sc = cfg["submission"]
    sub = build_submission(L("test_ids"), L(f"test_A_{n}"), L(f"test_B_{n}"), L(f"test_C_{n}"), confidence=conf,
                           technique_threshold=sc["technique_threshold"], nonfraud_classes=sc["nonfraud_classes"],
                           fraud_requires_technique=sc["fraud_requires_technique"])
    sample = load_sample_submission(cfg)
    out = Path(args.out) if args.out else out_dir / "submission.csv"
    sub = write_submission(sub, out, sample)
    print(json.dumps({"file": str(out), "rows": len(sub), "columns": list(sub.columns),
                      "problems": validate_submission(sub, sample, sc["nonfraud_classes"]),
                      "label_counts": sub["label"].value_counts().sort_index().to_dict(),
                      "action_counts": sub["action"].value_counts().sort_index().to_dict(),
                      "none_share": round(float((sub["techniques"] == "NONE").mean()), 4),
                      "mean_confidence": round(float(sub["confidence"].mean()), 4)}))


if __name__ == "__main__":
    main()
