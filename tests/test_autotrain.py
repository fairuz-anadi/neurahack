"""End-to-end smoke test of the unattended pipeline on fixture data with a tiny offline model.
No downloads, no LLM: synthetic and external files are reused fixtures; the LLM stages are disabled."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from common.data_loader import load_train
from common.folds import make_all_folds


def test_autotrain_smoke(tiny_data, tmp_path, monkeypatch):
    from models.finetune import autotrain, multitask as mt
    from models.finetune.synthetic import load_external, load_synthetic
    from tests.test_finetune import _tiny_backbone

    d, cfg = tiny_data
    monkeypatch.setattr(mt, "load_backbone", _tiny_backbone)
    monkeypatch.setenv("SCAMSHIELD_DATA_DIR", str(d))
    fold_dir = d / "folds"
    make_all_folds(load_train(cfg), cfg, fold_dir)
    monkeypatch.setenv("SCAMSHIELD_FOLDS_DIR", str(fold_dir))
    work = tmp_path / "work"
    monkeypatch.setenv("SCAMSHIELD_ARTIFACTS_DIR", str(work / "artifacts"))
    monkeypatch.setenv("SCAMSHIELD_OUTPUTS_DIR", str(work))

    syn = tmp_path / "syn.csv"
    pd.DataFrame({"id": [f"s{i}" for i in range(12)],
                  "text": [f"synthetic record {i} about a parcel fee and bkash pin {i * 7}" for i in range(12)],
                  "label": [i % 6 for i in range(12)],
                  "techniques": ["NONE" if i % 6 in (0, 5) else "T1|T5" for i in range(12)],
                  "action": [0 if i % 6 in (0, 5) else 2 for i in range(12)]}).to_csv(syn, index=False)
    ext = tmp_path / "ext.csv"
    pd.DataFrame({"id": [f"e{i}" for i in range(8)], "text": [f"external sms number {i} offer win {i}" for i in range(8)],
                  "label": [0, -1, -1, 5, 0, -1, 5, -1], "allowed": ["", "1|2|3|4", "1|2|3|4|5", "", "", "1|2|3|4", "", "1|2|3|4|5"],
                  "techniques": ["NONE", "", "", "NONE", "NONE", "", "NONE", ""], "action": [0, -1, -1, 0, 0, -1, 0, -1],
                  "source": ["x"] * 8}).to_csv(ext, index=False)
    nb = tmp_path / "ScamShield_AutoTrain.ipynb"
    nb.write_text("{}", encoding="utf-8")
    load_synthetic.cache_clear(); load_external.cache_clear()

    conf = autotrain.AutoConfig(work=str(work), hours=1.0, final_reserve_min=0.1, run_external=False,
                                external_reuse_path=str(ext), run_synth=False, synth_reuse_path=str(syn),
                                verify_llm=False, run_teacher=False, expected_fold_fp=None, test_max_len=32,
                                notebook_copy=str(nb), stress_top_k=1)
    rep = autotrain.run_pipeline(conf)

    names = set(rep["candidates"])
    assert {"A_ft1_fallback", "B_external", "C_synthetic", "D_syn_external", "E_balanced"} <= names
    assert rep["selected"]["best_candidate"] in names
    sub = pd.read_csv(work / "submission.csv", keep_default_na=False)
    sample = pd.read_csv(d / "sample_submission.csv")
    assert list(sub.columns) == ["id", "label", "techniques", "action", "confidence"]
    assert sub["id"].tolist() == sample["id"].tolist()
    assert ((sub.label.isin([0, 5])) <= ((sub.techniques == "NONE") & (sub.action == 0))).all()
    with zipfile.ZipFile(work / "final_package.zip") as z:
        assert sorted(z.namelist()) == sorted(["submission.csv", "ScamShield_AutoTrain.ipynb", ".env"])
    with zipfile.ZipFile(work / "prediction_artifacts.zip") as z:
        names_z = z.namelist()
        assert any(n.endswith("test_A_autotrain_final.npy") for n in names_z)
        assert any(n.endswith("oof_A_A_ft1_fallback.npy") for n in names_z)
    assert (work / "validation_report.html").exists() and (work / "progress.log").exists()
    saved = json.loads((work / "experiment_report.json").read_text(encoding="utf-8"))
    assert any(s["stage"].startswith("stress:") for s in saved["stages"])
    load_synthetic.cache_clear(); load_external.cache_clear()


def test_budget_skips_optional_work():
    from models.finetune.autotrain import Budget
    b = Budget(hours=0.01, reserve_min=0.5)   # 36 s total, 30 s reserved
    assert not b.can(60) and b.left() <= 36
