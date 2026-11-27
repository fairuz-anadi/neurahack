"""LLM/synthetic pipeline tests with a FAKE generator (no model download)."""
import json

import numpy as np
import pandas as pd
import pytest

from common.data_loader import load_train
from common.folds import load_folds, make_all_folds
from common.submission import build_submission, validate_submission
from models.finetune import llm


def test_parse_json_variants():
    assert llm.parse_json('```json\n{"label": 1}\n```') == {"label": 1}
    assert llm.parse_json('sure: [{"a": 1}, {"a": 2}] done') == [{"a": 1}, {"a": 2}]
    assert llm.parse_json("no json here") is None
    assert llm.parse_json(None) is None


def test_clean_record_rules():
    ok = llm.clean_record({"text": "Pay Tk 45 fee: hxxps://x[.]top", "label": 4, "techniques": "T3|T1", "action": 2})
    assert ok["techniques"] == "T1|T3"
    promo = llm.clean_record({"text": "Eid sale 30% off today", "label": 5, "techniques": "T5", "action": 3})
    assert promo["techniques"] == "NONE" and promo["action"] == 0           # official 0/5 convention
    assert llm.clean_record({"text": "Send OTP now please", "label": 1, "techniques": "NONE", "action": 1}) is None
    assert llm.clean_record({"text": "Send OTP now please", "label": 1, "techniques": "T9", "action": 1}) is None
    assert llm.clean_record({"text": "Send OTP now please", "label": 1, "techniques": "T2", "action": 1},
                            want_label=2) is None
    assert llm.clean_record({"text": "x", "label": 0, "techniques": "NONE", "action": 0}) is None


def _fake_synth_gen(conversations):
    outs = []
    for i, c in enumerate(conversations):
        lab = int(c[1]["content"].split("class ")[1].split(" ")[0])
        recs = [{"text": f"synthetic message {i} variant {k} for class {lab} " + "abcdefghij"[k] * (k + 3),
                 "label": lab, "techniques": "NONE" if lab in (0, 5) else "T1|T5",
                 "action": 0 if lab in (0, 5) else 5} for k in range(3)]
        recs.append({"text": recs[0]["text"], "label": lab, "techniques": "NONE", "action": 0})  # exact dup
        outs.append(json.dumps(recs) if i % 7 else "garbage")                                  # some unparseable
    return outs


def test_generate_synthetic_with_fake_llm(tmp_path):
    df, rep = llm.generate_synthetic(_fake_synth_gen, n_target=60, seed=1, per_prompt=3)
    assert rep["prompts"] == 20 and rep["unparseable_prompts"] == 3
    assert set(df["label"]) == set(range(6))
    assert df["text"].is_unique and df["id"].is_unique
    assert ((df.label.isin([0, 5])) <= (df.techniques == "NONE")).all()
    specs1, specs2 = llm.synth_specs(12, 7), llm.synth_specs(12, 7)
    assert specs1 == specs2                                                                     # reproducible
    path = tmp_path / "syn.csv"
    df.to_csv(path, index=False)
    from models.finetune.synthetic import load_synthetic
    syn = load_synthetic(str(path))
    assert len(syn) == len(df) and syn["is_synthetic"].all() and {"turns", "T1", "normalized_text"} <= set(syn.columns)


def test_synth_prompt_has_no_sample_rows(tiny_data):
    _, cfg = tiny_data
    tr = load_train(cfg)
    prompt = json.dumps([llm.synth_messages(s, 5) for s in llm.synth_specs(30, 0)], ensure_ascii=False)
    for t in tr["raw_text"]:
        assert t not in prompt


def test_llm_eval_never_leaks_validation_rows(tiny_data):
    d, cfg = tiny_data
    tr = load_train(cfg)
    make_all_folds(tr, cfg, d / "folds")
    fold = load_folds(tr, "folds", d / "folds")
    seen = []

    def fake(conversations):
        seen.extend(conversations)
        return ['{"label": 0, "techniques": "NONE", "action": 0}' for _ in conversations]

    m = llm.llm_eval(tr, fold, fake, per_class=1)
    assert len(seen) == len(tr) and m["unparseable"] == 0   # llm_eval asserts shots never include fold-k ids
    assert 0 <= m["F_A"] <= 1


def test_synthetic_rows_added_to_fit(tiny_data, tmp_path):
    from models.finetune.multitask import FTConfig, MultiTaskFT
    _, cfg = tiny_data
    tr = load_train(cfg)
    df, _ = llm.generate_synthetic(_fake_synth_gen, n_target=30, seed=2, per_prompt=3)
    p = tmp_path / "s.csv"; df.to_csv(p, index=False)
    m = MultiTaskFT(FTConfig(synthetic_path=str(p), synthetic_max=10, real_repeat=2), device="cpu")
    out = m._add_synthetic(tr)
    assert len(out) == 2 * len(tr) + min(10, len(df))


def test_submission_forces_action0_for_class5():
    pA = np.eye(6)[[5, 1]] * 0.9 + 0.1 / 6
    pC = np.eye(6)[[3, 1]] * 0.9 + 0.1 / 6
    sub = build_submission(np.array([1, 2]), pA, np.full((2, 6), 0.9), pC)
    assert sub.loc[0, "techniques"] == "NONE" and sub.loc[0, "action"] == 0
    assert sub.loc[1, "action"] == 1 and sub.loc[1, "techniques"] != "NONE"
    bad = sub.copy(); bad.loc[0, "action"] = 3
    assert any("action != 0" in p for p in validate_submission(bad))


def test_teacher_label_and_pseudo_join(tiny_data, tmp_path, monkeypatch):
    from common.data_loader import load_test
    from models.finetune import synthetic
    _, cfg = tiny_data
    tr, te = load_train(cfg), load_test(cfg)
    answers = iter(['{"label": 1, "techniques": "T2", "action": 1}', "junk", '{"label": 5, "techniques": "T4", "action": 3}'] * 10)

    def fake(convs):
        return [next(answers) for _ in convs]

    out = tmp_path / "teacher.csv"
    rep = llm.teacher_label(te, llm.pick_shots(tr, 1, 0), fake, n=6, seed=0, out_path=str(out), chunk=4)
    lab = pd.read_csv(out)
    assert rep["labeled"] == len(lab) == 4 and rep["unusable"] == 2
    assert set(lab["id"]) <= set(te["id"])
    assert (lab.loc[lab.label == 5, "techniques"] == "NONE").all() and (lab.loc[lab.label == 5, "action"] == 0).all()
    synthetic.load_pseudo.cache_clear()
    monkeypatch.setattr(synthetic, "_test_frame", lambda: te)
    ps = synthetic.load_pseudo(str(out))
    assert len(ps) == 4 and ps["is_pseudo"].all() and {"turns", "T2", "raw_text"} <= set(ps.columns)
    synthetic.load_pseudo.cache_clear()


def test_partial_label_loss_and_external_loader(tmp_path):
    import torch
    from models.finetune.multitask import _masked_loss
    from models.finetune.synthetic import load_external
    # masked loss equals plain CE for fully labeled rows
    la = torch.randn(4, 6); lb = torch.randn(4, 6); lc = torch.randn(4, 6)
    ta = torch.tensor([0, 1, 2, 3]); tb = torch.zeros(4, 6); tc = torch.tensor([0, 1, 2, 3])
    full = np.eye(6, dtype=np.float32)[[0, 1, 2, 3]]
    ones = np.ones(4, bool)
    got = _masked_loss(la, lb, lc, ta, tb, tc, full, ones, ones, None, None, torch.ones(6), (1.0, 0.5, 0.5), "cpu")
    ref = (torch.nn.functional.cross_entropy(la, ta) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(lb, tb)
           + 0.5 * torch.nn.functional.cross_entropy(lc, tc))
    assert torch.allclose(got, ref, atol=1e-5)
    # partial rows: allowed set {1,2,3,4}; B/C masked -> finite loss, gradients flow
    la.requires_grad_(True)
    part = np.zeros((4, 6), np.float32); part[:, 1:5] = 1
    ta2 = torch.full((4,), -1); zeros = np.zeros(4, bool)
    loss = _masked_loss(la, lb, lc, ta2, tb, tc, part, zeros, zeros, None, None, torch.ones(6), (1.0, 0.5, 0.5), "cpu")
    loss.backward(); assert torch.isfinite(loss) and la.grad is not None
    # loader
    p = tmp_path / "ext.csv"
    pd.DataFrame({"id": ["e1", "e2", "e3"], "text": ["Your OTP is 1234 do not share", "Win prize click now hxxps://a[.]b", "Eid sale 20% off"],
                  "label": [0, -1, 5], "allowed": ["", "1|2|3|4", ""], "techniques": ["NONE", "", "NONE"],
                  "action": [0, -1, 0], "source": ["s"] * 3}).to_csv(p, index=False)
    load_external.cache_clear()
    ex = load_external(str(p))
    assert ex["a_mask"].iloc[1].tolist() == [0, 1, 1, 1, 1, 0] and not ex["b_known"].iloc[1] and not ex["c_known"].iloc[1]
    assert ex["b_known"].iloc[0] and ex["c_known"].iloc[2] and ex["label"].tolist() == [0, -1, 5]
    load_external.cache_clear()


def test_fit_runs_with_external_partial_rows(tiny_data, tmp_path, monkeypatch):
    from tests.test_finetune import _tiny_backbone
    from models.finetune import multitask as mt
    from models.finetune.synthetic import load_external
    monkeypatch.setattr(mt, "load_backbone", _tiny_backbone)
    _, cfg = tiny_data
    tr = load_train(cfg)
    p = tmp_path / "ext.csv"
    pd.DataFrame({"id": [f"e{i}" for i in range(6)], "text": [f"external message number {i} pay fee" for i in range(6)],
                  "label": [0, -1, -1, 5, 0, -1], "allowed": ["", "1|2|3|4", "1|2|3|4|5", "", "", "1|2|3|4"],
                  "techniques": ["NONE", "", "", "NONE", "NONE", ""], "action": [0, -1, -1, 0, 0, -1],
                  "source": ["s"] * 6}).to_csv(p, index=False)
    load_external.cache_clear()
    m = mt.MultiTaskFT(mt.FTConfig(epochs=1, batch_size=4, max_len=32, external_path=str(p)), device="cpu").fit(tr)
    assert m.n_train_rows == len(tr) + 6
    load_external.cache_clear()
