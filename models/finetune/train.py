"""Train + evaluate the multi-task transformer on the shared folds; save artifacts per the contract.

    python -m models.finetune.train --name ft_e5s_base                 # FT0 baseline (main + stress)
    python -m models.finetune.train --name ft_e5s_corrupt --augment corrupt
    python -m models.finetune.train --name quick --main-only --epochs 5

If test.csv is present (Kaggle only), also fits on all labeled rows and writes test_{A,B,C} arrays.
Run `python -m models.finetune.benchmark` on Kaggle first to size the 500k inference.
Local tip (Windows): set HF_HOME to a drive with space, e.g. $env:HF_HOME = "E:\\neurahack\\hf_cache".
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
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from common.data_loader import get_targets, load_config, load_test, load_train, output_dirs, test_available
from common.folds import fold_fingerprint, load_folds
from common.metrics import calibration_score, composite_proxy, evaluate, save_json
from common.prediction_io import save_predictions
from common.seed import device_info, set_seed
from common.stress import run_cv, run_stress
from models.finetune.calibration import crossfit_confidence, fit_temperature, apply_temperature
from models.finetune.multitask import FTConfig, MultiTaskFT, make_fit_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ft_e5s_base")
    ap.add_argument("--model", default=FTConfig.model_name)
    ap.add_argument("--epochs", type=int, default=FTConfig.epochs)
    ap.add_argument("--max-len", type=int, default=FTConfig.max_len)
    ap.add_argument("--lr", type=float, default=FTConfig.lr_encoder)
    ap.add_argument("--augment", default="none", choices=["none", "corrupt"])
    ap.add_argument("--copies", type=int, default=FTConfig.augment_copies)
    ap.add_argument("--main-only", action="store_true", help="skip robust/novel/domain proxies")
    ap.add_argument("--offline", action="store_true", help="only use cached model files")
    args = ap.parse_args()

    cfg = load_config()
    seed = set_seed(cfg["seed"])
    art_dir, out_dir = output_dirs(cfg)
    train = load_train(cfg)
    yA, yB, yC = get_targets(train)
    fold = load_folds(train)
    ft = FTConfig(model_name=args.model, epochs=args.epochs, max_len=args.max_len, lr_encoder=args.lr,
                  augment=args.augment, augment_copies=args.copies, seed=seed, local_files_only=args.offline)
    timings: list = []
    fp = make_fit_predict(ft, timings)
    t0 = time.time()
    if args.main_only:
        oof = run_cv(fp, train, fold)
        report = {"main": evaluate(yA, yB, yC, oof["A"], oof["B"], oof["C"])}
    else:
        report, oof = run_stress(fp, train, seed=seed, repeats=False)

    conf, temps = crossfit_confidence(oof["A"], yA, fold)
    correct = (oof["A"].argmax(1) == yA).astype(int)
    report["task_d"] = {"raw_calibration": round(calibration_score(oof["A"].max(1), correct), 4),
                        "crossfit_temperature_calibration": round(calibration_score(conf, correct), 4),
                        "fold_temperatures": temps,
                        "full_temperature": round(fit_temperature(oof["A"], yA), 3)}
    m = report["main"]
    comp = composite_proxy({"F_A": m["F_A"], "F_B": m["F_B"], "F_C": m["F_C"],
                            "Calibration": report["task_d"]["crossfit_temperature_calibration"],
                            "F_robust": report.get("F_robust_proxy"), "F_novel": report.get("F_novel_proxy"),
                            "F_domain": report.get("F_domain_proxy")})
    report["composite"] = comp
    report["fit_seconds_per_fold"] = timings
    report["total_seconds"] = round(time.time() - t0, 1)
    report["device"] = device_info()

    test_pred, test_ids = None, None
    if test_available(cfg):
        test = load_test(cfg)
        model = MultiTaskFT(ft).fit(train)
        t1 = time.time()
        test_pred = model.predict(test)
        report["test_inference_seconds"] = round(time.time() - t1, 1)
        test_ids = test["id"].to_numpy()
        report["test_prediction_distribution_A"] = np.bincount(test_pred["A"].argmax(1), minlength=6).tolist()
    else:
        print("test.csv not found -> OOF artifacts only (expected outside Kaggle)")

    meta = {"config": MultiTaskFT(ft).config_dict(), "seed": seed,
            "dataset_fingerprint": train.attrs["fingerprint"], "fold_fingerprint": fold_fingerprint("folds"),
            "metrics": {k: m[k] for k in ("F_A", "F_B", "F_C", "Brier", "Calibration")}
                       | {k: report.get(k) for k in ("F_robust_proxy", "F_novel_proxy", "F_domain_proxy")}
                       | {"calibrated_Calibration": report["task_d"]["crossfit_temperature_calibration"],
                          "composite_proxy": comp["composite_proxy"]},
            "task_d_temperature": report["task_d"]["full_temperature"],
            "runtime_sec": report["total_seconds"], "device": report["device"]}
    d = save_predictions(args.name, art_dir, oof, train["id"].to_numpy(), meta, test_pred, test_ids)
    np.save(d / f"oof_conf_{args.name}.npy", conf.astype(np.float32))
    if test_pred is not None:
        np.save(d / f"test_conf_{args.name}.npy",
                apply_temperature(test_pred["A"], report["task_d"]["full_temperature"]).max(1).astype(np.float32))
    save_json(report, out_dir / f"eval_{args.name}.json")

    print(f"artifacts -> {d}")
    print(json.dumps({k: m[k] for k in ("F_A", "F_B", "F_C", "Brier", "Calibration", "accuracy_A")}))
    print("Task D:", json.dumps(report["task_d"]))
    print(json.dumps({k: report.get(k) for k in ("F_robust_proxy", "F_novel_proxy", "F_domain_proxy")}))
    print("LOCAL composite proxy (not a Kaggle score):", comp["composite_proxy"], "imputed:", comp["imputed_with_F_A"])
    if "slices_A" in report:
        print("slices:", json.dumps(report["slices_A"]))
    print("fit seconds per fold:", timings, "| total", report["total_seconds"])


if __name__ == "__main__":
    main()
