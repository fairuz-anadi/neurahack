"""Aggregate token-length and truncation statistics (tokenizer only, no model weights).

    python -m models.finetune.token_stats                 # test.csv if present, else training rows
    python -m models.finetune.token_stats --split train

Prints/saves ONLY aggregates (counts, shares, quantiles). No individual record is displayed or written.
Texts are built exactly as the model sees them (prefix + speaker-tagged turns).
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

from common.data_loader import load_config, load_test, load_train, output_dirs, test_available
from common.metrics import save_json
from models.finetune.multitask import FTConfig, build_text

LIMITS = (64, 128, 192, 256, 384, 512)


def token_lengths(df, cfg: FTConfig, tok, chunk: int = 20_000) -> np.ndarray:
    sep = getattr(tok, "sep_token", None) or "|"
    texts = [build_text(t, r, cfg, sep) for t, r in zip(df["turns"], df["raw_text"])]
    out = []
    for i in range(0, len(texts), chunk):
        enc = tok(texts[i:i + chunk], add_special_tokens=True, truncation=False)
        out.extend(len(x) for x in enc["input_ids"])
    return np.asarray(out)


def summarize(lengths: np.ndarray, n_turns: np.ndarray, scripts) -> dict:
    q = np.percentile(lengths, [50, 75, 90, 95, 99, 99.9]).round(1)
    rep = {"rows": int(len(lengths)),
           "quantiles": dict(zip(["p50", "p75", "p90", "p95", "p99", "p99.9"], q.tolist())),
           "max": int(lengths.max()), "mean": round(float(lengths.mean()), 1),
           "truncated_share": {f">{L}": round(float((lengths > L).mean()), 5) for L in LIMITS},
           "truncated_count": {f">{L}": int((lengths > L).sum()) for L in LIMITS}}
    buckets = {"1 turn": n_turns == 1, "2-4 turns": (n_turns >= 2) & (n_turns <= 4),
               "5-8 turns": (n_turns >= 5) & (n_turns <= 8), "9+ turns": n_turns >= 9}
    rep["by_turns"] = {k: {"rows": int(m.sum()), "mean_tokens": round(float(lengths[m].mean()), 1) if m.any() else None,
                           "share_over_128": round(float((lengths[m] > 128).mean()), 4) if m.any() else None,
                           "share_over_256": round(float((lengths[m] > 256).mean()), 4) if m.any() else None}
                       for k, m in buckets.items()}
    scripts = np.asarray(scripts)
    rep["by_script"] = {s: {"rows": int((scripts == s).sum()), "mean_tokens": round(float(lengths[scripts == s].mean()), 1)}
                        for s in sorted(set(scripts.tolist()))}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["auto", "train", "test"], default="auto")
    ap.add_argument("--model", default=FTConfig.model_name)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    cfg = load_config()
    _, out_dir = output_dirs(cfg)
    split = args.split
    if split == "auto":
        split = "test" if test_available(cfg) else "train"
    if split == "test" and not test_available(cfg):
        sys.exit("test.csv not available here (expected outside Kaggle)")
    df = load_test(cfg) if split == "test" else load_train(cfg)
    ft = FTConfig(model_name=args.model, local_files_only=args.offline)
    tok = AutoTokenizer.from_pretrained(ft.model_name, local_files_only=ft.local_files_only)
    rep = {"split": split, "model": ft.model_name, **summarize(token_lengths(df, ft, tok), df["n_turns"].to_numpy(),
                                                              df["script"].tolist())}
    save_json(rep, out_dir / f"token_stats_{split}.json")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
