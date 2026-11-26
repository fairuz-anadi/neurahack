import numpy as np
import pandas as pd
import pytest

from common.data_loader import load_train
from common.folds import build_groups, load_folds, make_all_folds, template_key
from common.metrics import WEIGHTS, calibration_score, composite_proxy, evaluate, task_b_f1
from common.prediction_io import ContractError, load_predictions, save_predictions, validate_probs


def test_template_grouping_merges_number_variants():
    texts = ["Your order #4471 is out for delivery today.", "Your order #9912 is out for delivery today.",
             "Eid sale 30% off"]
    g = build_groups(texts, 0.8)
    assert g[0] == g[1] != g[2]
    assert template_key("Tk 500 at hxxps://a[.]top") == template_key("Tk 99 at hxxps://b[.]xyz")


def test_folds_keep_groups_together_and_are_deterministic(tiny_data):
    d, cfg = tiny_data
    tr = load_train(cfg)
    meta1 = make_all_folds(tr, cfg, d / "folds")
    f1 = load_folds(tr, "folds", d / "folds")
    meta2 = make_all_folds(tr, cfg, d / "folds")
    assert meta1["main"]["fingerprint"] == meta2["main"]["fingerprint"]
    # the fixture repeats each text 3 times -> identical texts must share a fold
    for t, grp in tr.groupby("raw_text"):
        assert len(set(f1[grp.index])) == 1, t


def test_metrics_and_composite():
    yA = np.array([0, 1, 2, 3, 4, 5])
    pA = np.eye(6)
    yB = np.eye(6, dtype=int)
    m = evaluate(yA, yB, yA, pA, yB.astype(float), pA)
    assert m["F_A"] == 1.0 and m["F_B"] == 1.0 and m["F_C"] == 1.0 and m["Calibration"] == 1.0
    assert task_b_f1(np.zeros((3, 6)), np.zeros((3, 6))) == 0.0  # zero_division=0 assumption
    assert calibration_score([0.5, 0.5], [1, 0]) == 0.75
    c = composite_proxy({"F_A": 0.5, "F_B": 1.0})
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    assert "F_robust" in c["imputed_with_F_A"] and "F_B" not in c["imputed_with_F_A"]


def test_contract_validation_and_id_alignment(tmp_path):
    ids = np.array([10, 11, 12])
    P = np.full((3, 6), 1 / 6)
    meta = {"config": {}, "seed": 42, "dataset_fingerprint": "x", "fold_fingerprint": "y", "metrics": {}}
    save_predictions("m", tmp_path, {"A": P, "B": P, "C": P}, ids, meta)
    out = load_predictions(tmp_path / "m", "oof", expected_ids=np.array([12, 10, 11]))
    assert out["ids"].tolist() == [12, 10, 11]
    with pytest.raises(ContractError):
        load_predictions(tmp_path / "m", "oof", expected_ids=np.array([1, 2, 3]))
    with pytest.raises(ContractError):
        validate_probs("A", np.full((3, 6), 0.5))          # rows do not sum to 1
    with pytest.raises(ContractError):
        validate_probs("B", np.full((3, 5), 0.5))          # wrong width
    bad = P.copy(); bad[0, 0] = np.nan
    with pytest.raises(ContractError):
        validate_probs("C", bad)
    validate_probs("B", np.full((3, 6), 0.9))               # B is independent: rows may sum > 1
    with pytest.raises(ContractError):
        save_predictions("m2", tmp_path, {"A": P, "B": P, "C": P}, ids, {"seed": 42})
