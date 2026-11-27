"""Unattended end-to-end pipeline for Kaggle: data -> candidates -> validation -> selection -> test -> submission.

    python -m models.finetune.autotrain --work /kaggle/working [--hours 3.5]

Reuses the existing modules (loader, shared folds, metrics, stress tests, MultiTaskFT, calibration, external data,
Qwen synthetic/teacher, submission builder, prediction contract). Every optional stage is wrapped so a failure is
logged and the run continues; critical data-integrity problems (fold fingerprint mismatch, invalid submission)
stop the run instead of producing a bad file. A runtime budget skips optional work so the final model and
submission always get their reserved time.

Overfitting safeguards (see validation_report.html):
  * the shared fold manifest only (fingerprint checked); scores only on official held-out rows;
  * external and synthetic rows with char-TF-IDF cosine >= 0.8 to ANY sample.csv row are removed;
  * synthetic data comes from official definitions only (no sample rows in prompts);
  * external rows capped per source/label and per fit; real rows repeated so they keep weight;
  * fixed, short training schedules (no early stopping on the held-out fold; inner splits of ~14 rows are too
    small to be reliable, so schedules are predetermined) + weight decay, dropout, clipping, noise augmentation;
  * calibration temperatures are cross-fitted; class weights are computed inside each training fold;
  * teacher shot rows are excluded from the comparison metrics whenever the teacher candidate is ranked.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd

from common.data_loader import get_targets, load_config, load_sample_submission, load_test, load_train, test_available
from common.folds import fold_fingerprint, load_folds
from common.metrics import calibration_score, composite_proxy, evaluate, macro_f1, save_json
from common.prediction_io import save_predictions
from common.seed import device_info, set_seed
from common.stress import corrupt_df, run_cv
from common.submission import build_submission, validate_submission, write_submission
from models.finetune.calibration import apply_temperature, crossfit_confidence, fit_temperature
from models.finetune.multitask import FTConfig, MultiTaskFT, make_fit_predict

EXPECTED_FOLD_FP = "c357ab8ee361835a"


@dataclass
class AutoConfig:
    work: str = "/kaggle/working"
    hours: float = 3.5                  # total planned runtime
    final_reserve_min: float = 30.0     # always kept for final training + 500k inference + packaging
    seed: int = 42
    second_seed: int = 43
    test_max_len: int = 256
    run_external: bool = True
    include_nc: bool = False            # Bangalabarta (CC BY-NC-SA) stays out unless explicitly allowed
    external_cap: int = 300
    external_reuse_path: str | None = None
    external_snapshot_path: str | None = None   # embedded CC BY 4.0 snapshot, used if the source download is refused
    run_synth: bool = True
    synth_n: int = 900
    synth_max_min: float = 45.0
    synth_reuse_path: str | None = None
    verify_llm: bool = True
    run_teacher: bool = True
    teacher_n: int = 6000
    teacher_max_min: float = 30.0
    pseudo_min_conf: float = 0.8
    stress_top_k: int = 2
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model: str = FTConfig.model_name
    offline: bool = False
    expected_fold_fp: str | None = EXPECTED_FOLD_FP
    notebook_copy: str | None = None    # path of the notebook file to put in final_package.zip
    extra: dict = field(default_factory=dict)


class Budget:
    def __init__(self, hours: float, reserve_min: float):
        self.t0, self.limit, self.reserve = time.time(), hours * 3600, reserve_min * 60

    def elapsed(self) -> float:
        return time.time() - self.t0

    def left(self) -> float:
        return self.limit - self.elapsed()

    def can(self, est_s: float) -> bool:
        return self.left() - self.reserve >= est_s


class Run:
    """Holds the report; every stage writes into it and it is saved after each step."""

    def __init__(self, cfg: AutoConfig):
        self.cfg, self.work = cfg, Path(cfg.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.budget = Budget(cfg.hours, cfg.final_reserve_min)
        self.report = {"config": asdict(cfg), "stages": [], "candidates": {}, "device": device_info()}

    def log(self, msg: str):
        line = f"[{self.budget.elapsed() / 60:6.1f} min] {msg}"
        print(line, flush=True)
        with open(self.work / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def stage(self, name: str, status: str, **info):
        self.report["stages"].append({"stage": name, "status": status, "minute": round(self.budget.elapsed() / 60, 1),
                                      **info})
        self.save()
        self.log(f"{name}: {status} {json.dumps(info, default=str)[:300]}")

    def save(self):
        save_json(self.report, self.work / "experiment_report.json")


# ------------------------------------------------------------------ helpers
def sh(args: list, timeout_s: float | None, cwd=ROOT) -> int:
    """Run a repo module in a fresh process (frees GPU memory when it exits); stream output; bounded by timeout."""
    p = subprocess.Popen([sys.executable, "-u", "-m", *map(str, args)], cwd=cwd, env=os.environ.copy(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t0 = time.time()
    try:
        for line in p.stdout:
            if "Loading weights" not in line and "it/s]" not in line:
                print("   " + line, end="", flush=True)
            if timeout_s and time.time() - t0 > timeout_s:
                p.kill()
                print(f"   [timeout after {timeout_s / 60:.0f} min - process stopped]", flush=True)
                return -9
        return p.wait()
    finally:
        if p.poll() is None:
            p.kill()


def filter_near_sample(path_in: str, path_out: str, sample_texts) -> dict:
    from models.finetune.external import drop_near_sample

    df = pd.read_csv(path_in, keep_default_na=False)
    kept, n_drop = drop_near_sample(df, sample_texts)
    kept.to_csv(path_out, index=False)
    return {"rows_in": len(df), "removed_near_sample": n_drop, "rows_out": len(kept)}


def scores(train, oof, fold, rows=None) -> dict:
    """Official-style metrics on (a subset of) the OOF rows + cross-fitted calibration + screening composite."""
    yA, yB, yC = get_targets(train)
    idx = np.arange(len(train)) if rows is None else np.asarray(rows)
    m = evaluate(yA[idx], yB[idx], yC[idx], oof["A"][idx], oof["B"][idx], oof["C"][idx])
    conf, temps = crossfit_confidence(oof["A"], yA, fold)
    correct = (oof["A"].argmax(1) == yA).astype(int)
    m["Calibration_crossfit"] = round(calibration_score(conf[idx], correct[idx]), 4)
    m["screen_composite"] = composite_proxy({"F_A": m["F_A"], "F_B": m["F_B"], "F_C": m["F_C"],
                                             "Calibration": m["Calibration_crossfit"]})["composite_proxy"]
    return m


def candidate_specs(paths: dict, base: dict) -> dict:
    """Small set of high-value configurations. A = FT1 fallback."""
    syn, ext, ps = paths.get("synthetic"), paths.get("external"), paths.get("pseudo")
    c = {"A_ft1_fallback": dict(base, augment="corrupt", augment_copies=2)}
    if ext:
        c["B_external"] = dict(base, external_path=ext, external_max=1200, real_repeat=3, epochs=4, batch_size=32)
    if syn:
        c["C_synthetic"] = dict(base, synthetic_path=syn, real_repeat=3)
    if syn and ext:
        c["D_syn_external"] = dict(base, synthetic_path=syn, external_path=ext, external_max=1200, real_repeat=3,
                                   epochs=6, batch_size=32)
        c["E_balanced"] = dict(base, synthetic_path=syn, external_path=ext, external_max=600, real_repeat=6,
                               epochs=6, batch_size=32, augment="corrupt", augment_copies=1)
    if ps:
        c["F_teacher_student"] = dict(base, pseudo_path=ps, pseudo_min_conf=paths.get("pseudo_min_conf", 0.8),
                                      synthetic_path=syn, real_repeat=3, epochs=3, batch_size=32)
    return c


def cv_screen(run: Run, name: str, kw: dict, train, fold, seed: int) -> dict:
    """Main-fold CV. Also measures training-fit F_A on the real training rows (train-validation gap)."""
    yA = train["label"].to_numpy()
    oof = {k: np.zeros((len(train), 6)) for k in "ABC"}
    fit_s, gaps, peaks = [], [], []
    for k in sorted(np.unique(fold)):
        tr, va = train[fold != k], train[fold == k]
        m = MultiTaskFT(FTConfig(**dict(kw, seed=seed))).fit(tr)
        pv, pt = m.predict(va), m.predict(tr)
        m.release()
        for t in "ABC":
            oof[t][fold == k] = pv[t]
        fit_s.append(m.fit_seconds)
        peaks.append(m.peak_gpu_mb)
        gaps.append(macro_f1(tr["label"].to_numpy(), pt["A"].argmax(1)))
    res = {"oof": oof, "fit_seconds": fit_s, "train_F_A": round(float(np.mean(gaps)), 4),
           "peak_gpu_mb": max([p for p in peaks if p] or [0]), "seed": seed}
    return res


def stress(run: Run, kw: dict, train, seed: int) -> dict:
    fp = make_fit_predict(FTConfig(**dict(kw, seed=seed)))
    yA = train["label"].to_numpy()
    out = {}
    rob = run_cv(fp, train, load_folds(train, "folds"), valid_transform=lambda d: corrupt_df(d, seed))
    out["F_robust_proxy"] = round(macro_f1(yA, rob["A"].argmax(1)), 4)
    nov = run_cv(fp, train, load_folds(train, "folds_novel"))
    out["F_novel_proxy"] = round(macro_f1(yA, nov["A"].argmax(1)), 4)
    dom = run_cv(fp, train, load_folds(train, "folds_domain"))
    out["F_domain_proxy"] = round(macro_f1(yA, dom["A"].argmax(1)), 4)
    return out


def final_fit_predict(kw: dict, seed: int, train, test, max_len: int) -> tuple[dict, dict]:
    m = MultiTaskFT(FTConfig(**dict(kw, seed=seed, max_len=max_len))).fit(train)
    t0 = time.time()
    pred = m.predict(test)
    info = {"fit_seconds": m.fit_seconds, "inference_seconds": round(time.time() - t0, 1)}
    m.release()
    info["peak_gpu_mb"] = m.peak_gpu_mb
    return pred, info


# ------------------------------------------------------------------ pipeline
def run_pipeline(cfg: AutoConfig) -> dict:
    run = Run(cfg)
    W = run.work
    os.environ.setdefault("SCAMSHIELD_ARTIFACTS_DIR", str(W / "artifacts"))
    os.environ.setdefault("SCAMSHIELD_OUTPUTS_DIR", str(W))
    set_seed(cfg.seed)

    # ---- 0. data + integrity (critical)
    dcfg = load_config()
    train = load_train(dcfg)
    fold = load_folds(train)
    ffp = fold_fingerprint()
    if cfg.expected_fold_fp and ffp != cfg.expected_fold_fp:
        raise RuntimeError(f"fold manifest fingerprint {ffp} != expected {cfg.expected_fold_fp}; refusing to continue")
    has_test = test_available(dcfg)
    run.report["data"] = {"train_rows": len(train), "dataset_fingerprint": train.attrs["fingerprint"],
                          "fold_fingerprint": ffp, "test_available": has_test}
    run.stage("setup", "ok", **run.report["data"])
    sample_texts = train["raw_text"].tolist()
    paths: dict = {}

    # ---- 1. external data (CC BY 4.0 sources by default)
    if cfg.external_reuse_path and Path(cfg.external_reuse_path).exists():
        info = filter_near_sample(cfg.external_reuse_path, str(W / "external_v1.csv"), sample_texts)
        paths["external"] = str(W / "external_v1.csv")
        run.stage("external", "reused", **info)
    elif cfg.run_external:
        try:
            out = W / "external_v1.csv"
            args = ["models.finetune.external", "--out", out, "--cap", cfg.external_cap]
            if cfg.include_nc:
                args.append("--include-nc")
            if sh(args, timeout_s=15 * 60) == 0 and out.exists():
                paths["external"] = str(out)
                run.stage("external", "ok", rows=int(len(pd.read_csv(out))))
            elif cfg.external_snapshot_path and Path(cfg.external_snapshot_path).exists():
                info = filter_near_sample(cfg.external_snapshot_path, str(out), sample_texts)
                paths["external"] = str(out)
                run.stage("external", "snapshot", note="source download refused - using the embedded CC BY 4.0 "
                                                        "snapshot built by the same code", **info)
            else:
                run.stage("external", "failed", note="download/build failed - continuing without external data")
        except Exception as e:
            run.stage("external", "failed", error=repr(e))

    # ---- 2. synthetic data (Qwen, official definitions only)
    syn_raw = None
    if cfg.synth_reuse_path and Path(cfg.synth_reuse_path).exists():
        syn_raw = cfg.synth_reuse_path
        run.stage("synthetic", "reused", path=syn_raw)
    elif cfg.run_synth and run.budget.can(cfg.synth_max_min * 60 + 300):
        try:
            out = W / "synth_v1.csv"
            rc = sh(["models.finetune.llm", "synth", "--model", cfg.llm_model, "--n", cfg.synth_n, "--per-prompt", 5,
                     "--batch", 16, "--max-minutes", cfg.synth_max_min, "--out", out],
                    timeout_s=(cfg.synth_max_min + 15) * 60)
            partial = Path(str(out) + ".partial.csv")
            if rc == 0 and out.exists():
                syn_raw = str(out)
            elif partial.exists() and len(pd.read_csv(partial)) > 0:  # time-limited / crashed: keep finished rows
                from models.finetune.llm import quality_filter
                df, rep = quality_filter(pd.read_csv(partial, keep_default_na=False).to_dict("records"))
                df.insert(0, "id", [f"synp_{i:05d}" for i in range(len(df))])
                df.to_csv(out, index=False)
                syn_raw = str(out)
            run.stage("synthetic", "ok" if syn_raw else "failed", return_code=rc)
        except Exception as e:
            run.stage("synthetic", "failed", error=repr(e))
    else:
        run.stage("synthetic", "skipped", reason="disabled or not enough budget")
    if syn_raw:
        try:
            info = filter_near_sample(syn_raw, str(W / "synth_filtered.csv"), sample_texts)
            paths["synthetic"] = str(W / "synth_filtered.csv")
            run.stage("synthetic_filter", "ok", **info)
        except Exception as e:
            run.stage("synthetic_filter", "failed", error=repr(e))

    # ---- 3. screen candidates A-E on the main folds
    base = dict(model_name=cfg.student_model, local_files_only=cfg.offline)
    specs = candidate_specs(paths, base)
    results: dict = {}

    def screen(name, kw, rows=None):
        est = 5 * (results.get("_per_fit_est", 60))
        if name != "A_ft1_fallback" and not run.budget.can(est):
            run.stage(f"screen:{name}", "skipped", reason="budget")
            return
        try:
            r = cv_screen(run, name, kw, train, fold, cfg.seed)
            r["scores"] = scores(train, r["oof"], fold, rows)
            r["kw"] = kw
            results[name] = r
            results["_per_fit_est"] = max(results.get("_per_fit_est", 0), float(np.mean(r["fit_seconds"])))
            run.report["candidates"][name] = {k: v for k, v in r.items() if k not in ("oof",)}
            run.stage(f"screen:{name}", "ok", **{k: r["scores"][k] for k in ("F_A", "F_B", "F_C", "Calibration_crossfit",
                                                                            "screen_composite")},
                      train_F_A=r["train_F_A"], mean_fit_s=round(float(np.mean(r["fit_seconds"])), 1))
            save_predictions(name, Path(os.environ["SCAMSHIELD_ARTIFACTS_DIR"]), r["oof"], train["id"].to_numpy(),
                             {"config": {k: v for k, v in kw.items()}, "seed": cfg.seed,
                              "dataset_fingerprint": train.attrs["fingerprint"], "fold_fingerprint": ffp,
                              "metrics": r["scores"]})
        except Exception as e:
            run.stage(f"screen:{name}", "failed", error=repr(e), trace=traceback.format_exc()[-1500:])

    for name in ["A_ft1_fallback", "B_external", "D_syn_external", "E_balanced", "C_synthetic"]:
        if name in specs:
            screen(name, specs[name])

    # ---- 4. optional: verify the LLM and run the teacher -> student candidate
    shot_ids = None
    if cfg.verify_llm and run.budget.can(8 * 60):
        try:
            rc = sh(["models.finetune.llm", "eval", "--model", cfg.llm_model, "--shots", 2, "--batch", 8],
                    timeout_s=20 * 60)
            ev = W / "eval_llm_fewshot.json"
            if rc == 0 and ev.exists():
                m = json.loads(ev.read_text(encoding="utf-8"))
                run.report["llm_eval"] = {k: m.get(k) for k in ("F_A", "F_B", "F_C", "accuracy_A", "unparseable",
                                                                "seconds")}
                run.report["llm_eval"]["protocol"] = ("shared 5 folds, 86 official rows scored out-of-fold; 2 examples "
                                                      "per class drawn only from the other 4 folds; greedy decoding; "
                                                      "prompt = official class definitions + priority + conventions")
                run.stage("llm_eval", "ok", **run.report["llm_eval"])
            else:
                run.stage("llm_eval", "failed", return_code=rc)
        except Exception as e:
            run.stage("llm_eval", "failed", error=repr(e))
    llm_fa = (run.report.get("llm_eval") or {}).get("F_A") or 0
    best_fa = max([r["scores"]["F_A"] for n, r in results.items() if not n.startswith("_")] or [0])
    teacher_est = (cfg.teacher_max_min + 6) * 60 + 5 * results.get("_per_fit_est", 60) * 2
    if cfg.run_teacher and has_test and llm_fa > best_fa + 0.03 and run.budget.can(teacher_est):
        try:
            out = W / "teacher_labels.csv"
            rc = sh(["models.finetune.llm", "teacher", "--model", cfg.llm_model, "--n", cfg.teacher_n, "--shots", 2,
                     "--backend", "auto", "--batch", 32, "--max-minutes", cfg.teacher_max_min, "--out", out],
                    timeout_s=(cfg.teacher_max_min + 15) * 60)
            rep_path = W / "teacher_report.json"
            if out.exists() and len(pd.read_csv(out)) > 0:
                trep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
                shot_ids = set(trep.get("shot_ids", []))
                paths["pseudo"], paths["pseudo_min_conf"] = str(out), cfg.pseudo_min_conf
                run.stage("teacher", "ok", return_code=rc, **{k: trep.get(k) for k in
                                                              ("labeled", "rows_per_sec", "share_conf_ge_0.8", "minutes")})
            else:
                run.stage("teacher", "failed", return_code=rc)
        except Exception as e:
            run.stage("teacher", "failed", error=repr(e))
    else:
        run.stage("teacher", "skipped", reason=f"run_teacher={cfg.run_teacher}, test={has_test}, llm_F_A={llm_fa}, "
                                               f"best_student_F_A={best_fa}, budget_ok={run.budget.can(teacher_est)}")
    if "pseudo" in paths:
        specs = candidate_specs(paths, base)
        screen("F_teacher_student", specs["F_teacher_student"])

    # comparison rows: exclude teacher shot rows when the teacher candidate competes (they were in its prompts)
    cand = {n: r for n, r in results.items() if not n.startswith("_")}
    if not cand:
        raise RuntimeError("no candidate finished - not even the FT1 fallback")
    rows = None
    if shot_ids and "F_teacher_student" in cand:
        rows = np.where(~train["id"].astype(str).isin(shot_ids))[0]
        for n, r in cand.items():
            r["scores_noshot"] = scores(train, r["oof"], fold, rows)
            run.report["candidates"][n]["scores_noshot"] = r["scores_noshot"]
    key = "scores_noshot" if rows is not None else "scores"
    ranked = sorted(cand, key=lambda n: cand[n][key]["screen_composite"], reverse=True)
    run.report["screen_ranking"] = [(n, cand[n][key]["screen_composite"]) for n in ranked]
    run.stage("ranking", "ok", order=run.report["screen_ranking"])

    # ---- 5. stress tests on the shortlist
    per_fit = results.get("_per_fit_est", 60)
    for n in ranked[:cfg.stress_top_k]:
        if not run.budget.can(15 * per_fit * 1.3):
            run.stage(f"stress:{n}", "skipped", reason="budget")
            continue
        try:
            st = stress(run, cand[n]["kw"], train, cfg.seed)
            cand[n]["stress"] = st
            s = cand[n][key]
            cand[n]["full_composite"] = composite_proxy({"F_A": s["F_A"], "F_B": s["F_B"], "F_C": s["F_C"],
                                                         "Calibration": s["Calibration_crossfit"],
                                                         "F_robust": st["F_robust_proxy"], "F_novel": st["F_novel_proxy"],
                                                         "F_domain": st["F_domain_proxy"]})["composite_proxy"]
            run.report["candidates"][n].update(stress=st, full_composite=cand[n]["full_composite"])
            run.stage(f"stress:{n}", "ok", **st, full_composite=cand[n]["full_composite"])
        except Exception as e:
            run.stage(f"stress:{n}", "failed", error=repr(e))
    stressed = [n for n in ranked if "full_composite" in cand[n]]
    best = max(stressed, key=lambda n: cand[n]["full_composite"]) if stressed else ranked[0]

    # ---- 6. seed stability + optional 2-model average (only kept if OOF improves)
    members = [(best, cfg.seed)]
    oof_members = [cand[best]["oof"]]
    best_score = cand[best][key]["screen_composite"]
    if run.budget.can(6 * per_fit * 1.3):
        try:
            r2 = cv_screen(run, best, cand[best]["kw"], train, fold, cfg.second_seed)
            avg = {t: (cand[best]["oof"][t] + r2["oof"][t]) / 2 for t in "ABC"}
            s_avg = scores(train, avg, fold, rows)["screen_composite"]
            run.stage("seed_check", "ok", seed42=best_score, seed2_avg=s_avg)
            if s_avg >= best_score + 0.005:
                members.append((best, cfg.second_seed))
                oof_members.append(r2["oof"])
                best_score = s_avg
        except Exception as e:
            run.stage("seed_check", "failed", error=repr(e))
    if len(ranked) > 1:
        second = next((n for n in ranked if n != best), None)
        avg2 = {t: (np.mean([o[t] for o in oof_members], 0) + cand[second]["oof"][t]) / 2 for t in "ABC"}
        s2 = scores(train, avg2, fold, rows)["screen_composite"]
        run.stage("pair_check", "ok", best=best, second=second, single=best_score, averaged=s2)
        if s2 >= best_score + 0.01:
            members.append((second, cfg.seed))
            oof_members.append(cand[second]["oof"])
            best_score = s2
    oof_final = {t: np.mean([o[t] for o in oof_members], 0) for t in "ABC"}
    final_scores = scores(train, oof_final, fold)
    run.report["selected"] = {"best_candidate": best, "members": members, "oof_scores": final_scores,
                              "ft1_fallback_scores": cand.get("A_ft1_fallback", {}).get("scores")}
    run.stage("selection", "ok", best=best, members=members, F_A=final_scores["F_A"],
              screen_composite=final_scores["screen_composite"])

    # ---- 7. final training on all labeled + permitted extra data, test inference, submission (critical)
    if not has_test:
        run.stage("final", "skipped", reason="test.csv not available (expected outside Kaggle)")
        write_html(run)
        run.save()
        return run.report
    test = load_test(dcfg)
    sample = load_sample_submission(dcfg)
    preds, infos = [], []
    for n, sd in members:
        try:
            p, info = final_fit_predict(cand[n]["kw"], sd, train, test, cfg.test_max_len)
            preds.append(p)
            infos.append(dict(info, member=n, seed=sd))
            run.stage(f"final_fit:{n}:{sd}", "ok", **info)
        except Exception as e:
            run.stage(f"final_fit:{n}:{sd}", "failed", error=repr(e))
    if not preds and best != "A_ft1_fallback" and "A_ft1_fallback" in cand:      # fallback to FT1
        p, info = final_fit_predict(cand["A_ft1_fallback"]["kw"], cfg.seed, train, test, cfg.test_max_len)
        preds, members, oof_final = [p], [("A_ft1_fallback", cfg.seed)], cand["A_ft1_fallback"]["oof"]
        infos.append(dict(info, member="A_ft1_fallback", seed=cfg.seed))
        run.stage("final_fallback", "ok", **info)
    if not preds:
        raise RuntimeError("final training failed for every member - no submission written")
    test_avg = {t: np.mean([p[t] for p in preds], 0) for t in "ABC"}
    yA = train["label"].to_numpy()
    T = fit_temperature(oof_final["A"], yA)
    conf = apply_temperature(test_avg["A"], T).max(1)
    sc = dcfg["submission"]
    sub = build_submission(test["id"].to_numpy(), test_avg["A"], test_avg["B"], test_avg["C"], confidence=conf,
                           technique_threshold=sc["technique_threshold"], nonfraud_classes=sc["nonfraud_classes"],
                           fraud_requires_technique=sc["fraud_requires_technique"])
    sub = write_submission(sub, W / "submission.csv", sample)            # raises if anything is invalid
    problems = validate_submission(sub, sample, sc["nonfraud_classes"])
    run.report["final"] = {"members": members, "temperature": round(T, 3), "problems": problems, "rows": len(sub),
                           "label_counts": sub["label"].value_counts().sort_index().to_dict(),
                           "mean_confidence": round(float(sub["confidence"].mean()), 4), "runs": infos}
    art = Path(os.environ["SCAMSHIELD_ARTIFACTS_DIR"])
    save_predictions("autotrain_final", art, oof_final, train["id"].to_numpy(),
                     {"config": {"members": members, "kw": {n: cand[n]["kw"] for n, _ in members}}, "seed": cfg.seed,
                      "dataset_fingerprint": train.attrs["fingerprint"], "fold_fingerprint": ffp,
                      "metrics": final_scores, "task_d_temperature": T}, test_avg, test["id"].to_numpy())
    np.save(art / "autotrain_final" / "test_conf_autotrain_final.npy", conf.astype(np.float32))
    run.stage("submission", "ok", rows=len(sub), problems=problems)

    # ---- 8. packaging
    try:
        shutil.make_archive(str(W / "prediction_artifacts"), "zip", str(art.parent), art.name)
        with zipfile.ZipFile(W / "final_package.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(W / "submission.csv", "submission.csv")
            nb = cfg.notebook_copy
            if nb and Path(nb).exists():
                z.write(nb, Path(nb).name)
            z.writestr(".env", b"")
        run.stage("package", "ok", note="candidate package (team-wide selection still pending)",
                  includes_notebook=bool(cfg.notebook_copy and Path(cfg.notebook_copy).exists()))
    except Exception as e:
        run.stage("package", "failed", error=repr(e))
    run.report["total_minutes"] = round(run.budget.elapsed() / 60, 1)
    write_html(run)
    run.save()
    return run.report


# ------------------------------------------------------------------ report
def write_html(run: Run):
    r = run.report
    rows = []
    for n, c in r["candidates"].items():
        s = c.get("scores", {})
        st = c.get("stress", {})
        rows.append(f"<tr><td>{n}</td><td>{s.get('F_A')}</td><td>{s.get('F_B')}</td><td>{s.get('F_C')}</td>"
                    f"<td>{s.get('Calibration_crossfit')}</td><td>{s.get('screen_composite')}</td>"
                    f"<td>{c.get('train_F_A')}</td><td>{st.get('F_robust_proxy', '')}</td>"
                    f"<td>{st.get('F_novel_proxy', '')}</td><td>{st.get('F_domain_proxy', '')}</td>"
                    f"<td>{c.get('full_composite', '')}</td><td>{round(float(np.mean(c.get('fit_seconds', [0]))), 1)}</td>"
                    f"<td>{c.get('peak_gpu_mb')}</td></tr>")
    sel = r.get("selected", {})
    best = sel.get("best_candidate")
    detail = ""
    if best and best in r["candidates"]:
        s = r["candidates"][best]["scores"]
        detail = (f"<h2>Selected: {best}</h2><p>Members: {sel.get('members')}</p>"
                  f"<h3>Per-class Task A</h3><pre>{json.dumps(s.get('per_class_A'), indent=1)}</pre>"
                  f"<h3>Confusion Task A (rows true, cols predicted)</h3><pre>{s.get('confusion_A')}</pre>"
                  f"<h3>Per-technique Task B</h3><pre>{json.dumps(s.get('per_technique_B'), indent=1)}</pre>"
                  f"<h3>Per-action Task C</h3><pre>{json.dumps(s.get('per_action_C'), indent=1)}</pre>")
    stages = "".join(f"<li>{x['minute']} min - {x['stage']}: <b>{x['status']}</b> "
                     f"{json.dumps({k: v for k, v in x.items() if k not in ('stage', 'status', 'minute', 'trace')}, default=str)[:400]}</li>"
                     for x in r["stages"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ScamShield AutoTrain report</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}pre{{background:#f5f5f5;padding:8px;font-size:12px}}</style>
</head><body><h1>ScamShield BD - AutoTrain validation report</h1>
<p><b>All numbers are local out-of-fold scores on the 86 official labeled rows (shared 5 folds). They are NOT Kaggle
leaderboard scores.</b> With 86 rows, differences below ~0.03 are within noise.</p>
<p>Data: {json.dumps(r.get('data'))}</p><p>LLM check: {json.dumps(r.get('llm_eval'))}</p>
<h2>Candidates</h2><table><tr><th>candidate</th><th>F_A</th><th>F_B</th><th>F_C</th><th>calib</th><th>screen comp.</th>
<th>train F_A (gap)</th><th>robust</th><th>novel</th><th>domain</th><th>full comp.</th><th>fit s</th><th>GPU MB</th></tr>
{''.join(rows)}</table>{detail}
<h2>Final</h2><pre>{json.dumps(r.get('final'), indent=1, default=str)}</pre>
<h2>Overfitting safeguards</h2><ul><li>shared fold manifest only, fingerprint checked; metrics only on official rows</li>
<li>external/synthetic rows near any sample.csv row removed (cosine &gt;= 0.8); synthetic from definitions only</li>
<li>external capped per source/label and per fit; real rows repeated</li><li>fixed short schedules (no early stopping on
the held-out fold; inner splits too small), weight decay, dropout, gradient clipping, noise augmentation</li>
<li>cross-fitted calibration; teacher shot rows excluded from comparisons when the teacher candidate is ranked</li></ul>
<h2>Limitations</h2><ul><li>86 labeled rows: high variance; proxies are local stand-ins for hidden subsets</li>
<li>synthetic and pseudo labels come from an LLM and can be wrong</li><li>official leaderboard evaluation still required</li></ul>
<h2>Stages</h2><ol>{stages}</ol><p>Total minutes: {r.get('total_minutes')}</p></body></html>"""
    (run.work / "validation_report.html").write_text(html, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", default=None, help="JSON file with AutoConfig fields")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--hours", type=float, default=None)
    args = ap.parse_args(argv)
    kw = json.loads(Path(args.config_json).read_text(encoding="utf-8")) if args.config_json else {}
    kw.setdefault("work", args.work)
    if args.hours:
        kw["hours"] = args.hours
    cfg = AutoConfig(**{k: v for k, v in kw.items() if k in AutoConfig.__dataclass_fields__})
    try:
        rep = run_pipeline(cfg)
        print("AUTOTRAIN DONE", json.dumps({"selected": rep.get("selected", {}).get("members"),
                                            "total_minutes": rep.get("total_minutes")}, default=str))
    except Exception:
        traceback.print_exc()
        Path(cfg.work, "AUTOTRAIN_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
