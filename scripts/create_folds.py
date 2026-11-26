"""Create the shared fold manifests in folds/. Run once by Anadi; everyone else pulls them from git.

    python scripts/create_folds.py
"""
import _bootstrap  # noqa: F401

import json

from common.data_loader import load_config, load_train
from common.folds import _fdir, make_all_folds
from common.seed import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    train = load_train(cfg)
    meta = make_all_folds(train, cfg)
    m = meta["main"]
    print(f"main folds: {m['n_splits']} splits, {m['n_groups']} groups ({m['multi_row_groups']} multi-row), "
          f"fingerprint {m['fingerprint']}")
    for k, c in m["coverage"].items():
        missing_a = [cl for cl, n in c["task_a"].items() if n == 0]
        missing_c = [cl for cl, n in c["task_c"].items() if n == 0]
        missing_t = [t for t, n in c["techniques"].items() if n == 0]
        print(f"  fold {k}: n={c['n']} A={list(c['task_a'].values())} C={list(c['task_c'].values())} "
              f"T={list(c['techniques'].values())}  missing A:{missing_a} C:{missing_c} T:{missing_t}")
    print(f"novel folds: {meta['novel']['n_splits']} splits, {meta['novel']['n_groups']} groups")
    print(f"domain folds: {meta['domain']['n_splits']} splits, brand rows {meta['domain']['n_brand_rows']}")
    print(f"cross-fold pairs with cosine >= 0.6 (leakage watch): {len(m['leakage_risk_pairs'])}")
    print(json.dumps(m["leakage_risk_pairs"][:5]))
    print(f"wrote {_fdir(None)}")


if __name__ == "__main__":
    main()
