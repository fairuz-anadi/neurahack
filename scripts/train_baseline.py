"""Train + evaluate the classical baseline on the shared folds; save artifacts per the contract.
If test.csv exists (Kaggle), also fit on all labeled rows, predict test and write submission.csv.

    python scripts/train_baseline.py [--no-repeats] [--model-name tfidf_lr]
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time

from common.data_loader import (load_config, load_sample_submission, load_test, load_train, output_dirs,
                                test_available)
from common.folds import fold_fingerprint
from common.metrics import composite_proxy, save_json
from common.prediction_io import save_predictions
from common.seed import device_info, set_seed
from common.stress import run_stress
from common.submission import build_submission, write_submission
from models.classical.tfidf_lr import TfidfLR, TfidfLRConfig, make_fit_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="tfidf_lr")
    ap.add_argument("--no-repeats", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed = set_seed(cfg["seed"])
    art_dir, out_dir = output_dirs(cfg)
    train = load_train(cfg)
    mcfg = TfidfLRConfig(seed=seed)

    t0 = time.time()
    report, oof = run_stress(make_fit_predict(mcfg), train, seed=seed, repeats=not args.no_repeats)
    main_m = report["main"]
    comp = composite_proxy({"F_A": main_m["F_A"], "F_B": main_m["F_B"], "F_C": main_m["F_C"],
                            "Calibration": main_m["Calibration"], "F_robust": report["F_robust_proxy"],
                            "F_novel": report["F_novel_proxy"], "F_domain": report["F_domain_proxy"]})
    report["composite"] = comp

    test_pred, test_ids = None, None
    if test_available(cfg):
        test = load_test(cfg)
        model = TfidfLR(mcfg).fit(train)
        test_pred, test_ids = model.predict(test), test["id"].to_numpy()
        sample = load_sample_submission(cfg)
        sc = cfg["submission"]
        sub = build_submission(test_ids, test_pred["A"], test_pred["B"], test_pred["C"],
                               technique_threshold=sc["technique_threshold"],
                               nonfraud_classes=sc["nonfraud_classes"],
                               fraud_requires_technique=sc["fraud_requires_technique"])
        write_submission(sub, out_dir / "submission.csv", sample)
        report["test_prediction_distribution_A"] = sub["label"].value_counts(normalize=True).sort_index().round(4).to_dict()
        print(f"wrote {out_dir / 'submission.csv'} ({len(sub)} rows, validated)")
    else:
        print("test.csv not found -> OOF artifacts only (no test predictions, no submission.csv)")

    meta = {"config": TfidfLR(mcfg).config_dict(), "seed": seed,
            "dataset_fingerprint": train.attrs["fingerprint"], "fold_fingerprint": fold_fingerprint("folds"),
            "metrics": {k: report[k] for k in ("F_robust_proxy", "F_novel_proxy", "F_domain_proxy")}
                       | {k: main_m[k] for k in ("F_A", "F_B", "F_C", "Brier", "Calibration")}
                       | {"composite_proxy": comp["composite_proxy"]},
            "runtime_sec": round(time.time() - t0, 1), "device": device_info()}
    d = save_predictions(args.model_name, art_dir, oof, train["id"].to_numpy(), meta, test_pred, test_ids)
    save_json(report, out_dir / f"eval_{args.model_name}.json")

    print(f"artifacts -> {d}")
    print(json.dumps({k: main_m[k] for k in ("F_A", "F_B", "F_C", "Brier", "Calibration", "accuracy_A")}))
    print(json.dumps({k: report[k] for k in ("F_robust_proxy", "F_novel_proxy", "F_domain_proxy")}))
    if "repeated_F_A" in report:
        print("repeated F_A:", report["repeated_F_A"])
    print("LOCAL composite proxy (not a Kaggle score):", comp["composite_proxy"], "imputed:", comp["imputed_with_F_A"])
    print("slices:", json.dumps(report["slices_A"]))


if __name__ == "__main__":
    main()
