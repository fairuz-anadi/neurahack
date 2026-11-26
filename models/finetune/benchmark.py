"""Inference throughput benchmark on STAND-IN rows (training texts, optionally stitched into long
conversations). Never reads test.csv. Run on the target hardware (Kaggle GPU) before 500k inference.

    python -m models.finetune.benchmark --rows 2000 --batch 64 128 --max-len 128 256

Weights are the pretrained backbone + untrained heads: speed does not depend on the head weights.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from common.data_loader import load_train, split_turns
from common.seed import device_info
from models.finetune.multitask import FTConfig, MultiTaskFT, MultiTaskNet, load_backbone

N_TEST = 500_000


def standin_rows(train: pd.DataFrame, n: int, conv_share: float, seed: int = 0) -> pd.DataFrame:
    """Mix of single messages and multi-turn conversations (3-12 turns) built from training texts."""
    rng = np.random.default_rng(seed)
    texts = train["raw_text"].tolist()
    rows = []
    for _ in range(n):
        if rng.random() < conv_share:
            k = int(rng.integers(3, 13))
            parts = [f"{'SR'[i % 2]}: {texts[int(rng.integers(len(texts)))]}" for i in range(k)]
            rows.append(" ||| ".join(parts))
        else:
            rows.append(texts[int(rng.integers(len(texts)))])
    df = pd.DataFrame({"raw_text": rows})
    df["turns"] = [split_turns(t) for t in df["raw_text"]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--batch", type=int, nargs="+", default=[64])
    ap.add_argument("--max-len", type=int, nargs="+", default=[128])
    ap.add_argument("--conv-share", type=float, default=0.1, help="share of long conversations in stand-in rows")
    ap.add_argument("--model", default=FTConfig.model_name)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    train = load_train()
    df = standin_rows(train, args.rows, args.conv_share)
    results = []
    for ml in args.max_len:
        for bs in args.batch:
            cfg = FTConfig(model_name=args.model, max_len=ml, eval_batch_size=bs, local_files_only=args.offline)
            m = MultiTaskFT(cfg)
            m.tok, enc = load_backbone(cfg)
            m.net = MultiTaskNet(enc, 0.0).to(m.device).eval()
            m.predict(df.head(min(64, len(df))))  # warm-up
            if m.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            m.predict(df)
            if m.device.type == "cuda":
                torch.cuda.synchronize()
            sec = time.time() - t
            rps = len(df) / sec
            r = {"max_len": ml, "batch": bs, "rows": len(df), "seconds": round(sec, 2), "rows_per_sec": round(rps, 1),
                 "est_500k_minutes": round(N_TEST / rps / 60, 1), "device": str(m.device)}
            results.append(r)
            print(json.dumps(r))
    print("device:", device_info())


if __name__ == "__main__":
    main()
