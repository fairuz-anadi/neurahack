import numpy as np
import pandas as pd

from common.data_loader import load_sample_submission, load_test, load_train
from common.folds import load_folds, make_all_folds
from common.submission import build_submission, validate_submission, write_submission
from models.classical.tfidf_lr import TfidfLR, make_fit_predict


def _probs(n, k=6, seed=0):
    r = np.random.default_rng(seed).random((n, k))
    return r / r.sum(1, keepdims=True)


def test_submission_rules():
    ids = np.array([5, 6, 7])
    pA = np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02], [0.1, 0.5, 0.1, 0.1, 0.1, 0.1], _probs(1)[0]])
    pB = np.array([[0.9] * 6, [0.1] * 6, [0.6, 0.1, 0.7, 0.1, 0.1, 0.1]])
    sub = build_submission(ids, pA, pB, _probs(3))
    assert list(sub.columns) == ["id", "label", "techniques", "action", "confidence"]
    assert sub.techniques[0] == "NONE"            # label 0 -> NONE even with high technique probs
    assert sub.techniques[1] == "NONE"            # fraud, nothing above threshold, rule off
    assert validate_submission(sub) == []
    sub2 = build_submission(ids, pA, pB, _probs(3), fraud_requires_technique=True)
    assert sub2.techniques[1] != "NONE"
    assert sub.label.dtype.kind == "i" and sub.action.dtype.kind == "i"


def test_validator_catches_problems():
    good = pd.DataFrame({"id": [1, 2], "label": [0, 3], "techniques": ["NONE", "T1|T6"], "action": [0, 4],
                         "confidence": [0.5, 0.9]})
    assert validate_submission(good) == []
    cases = {
        "label": good.assign(label=[0.0, 2.0]),
        "outside": good.assign(action=[0, 6]),
        "malformed": good.assign(techniques=["NONE", "T7"]),
        "repeat": good.assign(techniques=["NONE", "T1|T1"]),
        "non-fraud": good.assign(techniques=["T1", "T1"]),
        "confidence": good.assign(confidence=[0.5, 1.5]),
        "duplicate": good.assign(id=[1, 1]),
    }
    for key, bad in cases.items():
        assert validate_submission(bad), key
    assert validate_submission(good[["id", "label"]])
    sample = good.assign(id=[1, 3])
    assert any("id set" in p for p in validate_submission(good, sample))


def test_end_to_end_on_fixture(tiny_data, tmp_path):
    d, cfg = tiny_data
    tr = load_train(cfg)
    make_all_folds(tr, cfg, d / "folds")
    fold = load_folds(tr, "folds", d / "folds")
    fp = make_fit_predict()
    k = 0
    pred = fp(tr[fold != k], tr[fold == k])
    assert pred["A"].shape == ((fold == k).sum(), 6)
    np.testing.assert_allclose(pred["A"].sum(1), 1, atol=1e-6)
    te = load_test(cfg)
    m = TfidfLR().fit(tr)
    p = m.predict(te)
    p2 = TfidfLR().fit(tr).predict(te)
    np.testing.assert_allclose(p["A"], p2["A"])  # deterministic
    sample = load_sample_submission(cfg)
    sub = build_submission(te.id.to_numpy()[::-1], p["A"][::-1], p["B"][::-1], p["C"][::-1])
    out = write_submission(sub, tmp_path / "submission.csv", sample)
    assert out.id.tolist() == sample.id.tolist()  # reordered to sample order
    assert validate_submission(pd.read_csv(tmp_path / "submission.csv", keep_default_na=False), sample) == []


def test_missing_technique_in_fold_is_handled(tiny_data):
    _, cfg = tiny_data
    tr = load_train(cfg)
    sub = tr[tr.label.isin([0, 1, 5])]  # T3, T4, T6 absent; classes 2-4 absent
    p = TfidfLR().fit(sub).predict(tr)
    assert np.isfinite(p["B"]).all() and (p["A"][:, [2, 3, 4]] == 0).all()
    np.testing.assert_allclose(p["A"].sum(1), 1, atol=1e-6)
