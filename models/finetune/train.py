"""Train + evaluate the multi-task transformer on the shared folds; save artifacts per the contract.

Validation (no test inference — the default):
    python -m models.finetune.train --name ft_e5s_base                     # FT0: main CV + proxies
    python -m models.finetune.train --name ft_e5s_corrupt --augment corrupt   # FT1 ablation
    python -m models.finetune.train --name quick --main-only --epochs 5

Full test inference is OPT-IN (Kaggle only, needs test.csv):
    python -m models.finetune.train --name ft_e5s_base --no-cv --predict-test
        reuses artifacts/<name>/ (OOF arrays + Task D temperature) from the validation run of the same
        name, fits on all 86 labeled rows, predicts every test row, re-saves the contract with test arrays.
    python -m models.finetune.train --name ft_e5s_base --predict-test       # validation + test in one go

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
from common.prediction_io import load_predictions, save_predictions
from common.seed import device_info, set_seed
from common.stress import run_cv, run_stress
from models.finetune.calibration import apply_temperature, crossfit_confidence, fit_temperature
from models.finetune.multitask import FTConfig, MultiTaskFT, make_fit_predict


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ft_e5s_base")
    ap.add_argument("--model", default=FTConfig.model_name)
    ap.add_argument("--epochs", type=int, default=FTConfig.epochs)
    ap.add_argument("--max-len", type=int, default=FTConfig.max_len)
    ap.add_argument("--lr", type=float, default=FTConfig.lr_encoder)
    ap.add_argument("--batch-size", type=int, default=FTConfig.batch_size, help="training batch size")
    ap.add_argument("--eval-batch", type=int, default=FTConfig.eval_batch_size, help="inference batch size")
    ap.add_argument("--no-fp16", action="store_true", help="disable CUDA mixed precision")
    ap.add_argument("--augment", default="none", choices=["none", "corrupt"])
    ap.add_argument("--copies", type=int, default=FTConfig.augment_copies)
    ap.add_argument("--synthetic", default=None, help="synthetic labeled CSV added to every training fit")
    ap.add_argument("--synthetic-max", type=int, default=None, help="cap synthetic rows per fit")
    ap.add_argument("--real-repeat", type=int, default=1, help="repeat real rows k times per fit")
    ap.add_argument("--pseudo", default=None, help="teacher pseudo-label CSV for test ids (Kaggle only)")
    ap.add_argument("--pseudo-max", type=int, default=None, help="cap pseudo-labeled rows per fit")
    ap.add_argument("--pseudo-min-conf", type=float, default=0.0, help="teacher label-confidence filter")
    ap.add_argument("--external", default=None, help="external public data CSV (models/finetune/external.py)")
    ap.add_argument("--external-max", type=int, default=None, help="cap external rows per fit")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed (FT4 seed check)")
    ap.add_argument("--main-only", action="store_true", help="skip robust/novel/domain proxies")
    ap.add_argument("--no-cv", action="store_true", help="skip validation; needs a previous run with --name")
    ap.add_argument("--predict-test", action="store_true", help="OPT-IN: predict every test row (Kaggle)")
    ap.add_argument("--offline", action="store_true", help="only use cached model files")
    return ap.parse_args(argv)


def run_validation(args, ft, train, seed, out_dir):
    yA, yB, yC = get_targets(train)
    fold = load_folds(train)
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
    report["composite"] = composite_proxy({
        "F_A": m["F_A"], "F_B": m["F_B"], "F_C": m["F_C"],
        "Calibration": report["task_d"]["crossfit_temperature_calibration"],
        "F_robust": report.get("F_robust_proxy"), "F_novel": report.get("F_novel_proxy"),
        "F_domain": report.get("F_domain_proxy")})
    report["fit_seconds_per_fold"] = timings
    report["validation_seconds"] = round(time.time() - t0, 1)
    return report, oof, conf


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config()
    seed = set_seed(args.seed if args.seed is not None else cfg["seed"])
    art_dir, out_dir = output_dirs(cfg)
    train = load_train(cfg)
    ft = FTConfig(model_name=args.model, epochs=args.epochs, max_len=args.max_len, lr_encoder=args.lr,
                  batch_size=args.batch_size, eval_batch_size=args.eval_batch, fp16=not args.no_fp16,
                  augment=args.augment, augment_copies=args.copies, seed=seed, local_files_only=args.offline,
                  synthetic_path=args.synthetic, synthetic_max=args.synthetic_max, real_repeat=args.real_repeat,
                  pseudo_path=args.pseudo, pseudo_max=args.pseudo_max, pseudo_min_conf=args.pseudo_min_conf,
                  external_path=args.external, external_max=args.external_max)
    if args.no_cv and not args.predict_test:
        sys.exit("--no-cv without --predict-test does nothing")
    if args.predict_test and not test_available(cfg):
        sys.exit("--predict-test needs test.csv (only available on Kaggle). Nothing was run.")

    run_dir = art_dir / args.name
    eval_path = out_dir / f"eval_{args.name}.json"
    if args.no_cv:
        if not (run_dir / "metadata.json").exists():
            sys.exit(f"--no-cv needs a previous validation run saved in {run_dir}")
        prev_meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if prev_meta["config"] != MultiTaskFT(ft).config_dict():
            print("WARNING: config differs from the validation run of the same name; OOF and test may not match.")
        oof = load_predictions(run_dir, "oof", expected_ids=train["id"].to_numpy())
        conf = np.load(run_dir / f"oof_conf_{args.name}.npy")
        report = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        temperature = prev_meta["task_d_temperature"]
        metrics = prev_meta["metrics"]
    else:
        report, oof, conf = run_validation(args, ft, train, seed, out_dir)
        temperature = report["task_d"]["full_temperature"]
        m = report["main"]
        metrics = ({k: m[k] for k in ("F_A", "F_B", "F_C", "Brier", "Calibration")}
                   | {k: report.get(k) for k in ("F_robust_proxy", "F_novel_proxy", "F_domain_proxy")}
                   | {"calibrated_Calibration": report["task_d"]["crossfit_temperature_calibration"],
                      "composite_proxy": report["composite"]["composite_proxy"]})
    report["device"] = device_info()

    test_pred, test_ids, test_conf = None, None, None
    if args.predict_test:
        test = load_test(cfg)
        model = MultiTaskFT(ft).fit(train)                       # final model: all labeled rows
        t1 = time.time()
        test_pred = model.predict(test)
        sec = time.time() - t1
        model.release()
        test_ids = test["id"].to_numpy()
        test_conf = apply_temperature(test_pred["A"], temperature).max(1)
        report["test_inference"] = {"rows": len(test), "seconds": round(sec, 1),
                                    "rows_per_sec": round(len(test) / max(sec, 1e-9), 1),
                                    "final_fit_seconds": model.fit_seconds, "peak_gpu_mb": model.peak_gpu_mb,
                                    "eval_batch_size_used": model.last_eval_batch_size,
                                    "predicted_A_counts": np.bincount(test_pred["A"].argmax(1), minlength=6).tolist(),
                                    "mean_confidence": round(float(test_conf.mean()), 4)}
        print("test inference:", json.dumps(report["test_inference"]))
    else:
        print("test inference skipped (opt-in: --predict-test)")

    meta = {"config": MultiTaskFT(ft).config_dict(), "seed": seed,
            "dataset_fingerprint": train.attrs["fingerprint"], "fold_fingerprint": fold_fingerprint("folds"),
            "metrics": metrics, "task_d_temperature": temperature, "device": report["device"],
            "test_inference": report.get("test_inference")}
    d = save_predictions(args.name, art_dir, {k: oof[k] for k in "ABC"}, train["id"].to_numpy(), meta,
                         test_pred, test_ids)
    np.save(d / f"oof_conf_{args.name}.npy", np.asarray(conf, dtype=np.float32))
    if test_conf is not None:
        np.save(d / f"test_conf_{args.name}.npy", test_conf.astype(np.float32))
    save_json(report, eval_path)

    print(f"artifacts -> {d}")
    if "main" in report:
        m = report["main"]
        print(json.dumps({k: m[k] for k in ("F_A", "F_B", "F_C", "Brier", "Calibration", "accuracy_A")}))
        print("Task D:", json.dumps(report["task_d"]))
        print(json.dumps({k: report.get(k) for k in ("F_robust_proxy", "F_novel_proxy", "F_domain_proxy")}))
        print("LOCAL composite proxy (not a Kaggle score):", report["composite"]["composite_proxy"])
        if "slices_A" in report:
            print("slices:", json.dumps(report["slices_A"]))
        print("fit seconds per fold:", report.get("fit_seconds_per_fold"))


if __name__ == "__main__":
    main()
