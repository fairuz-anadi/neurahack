from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from common.data_loader import get_targets, load_test, load_train, split_turns
from common.labels import TECHNIQUES, LabelError, binary_to_techniques, parse_techniques, techniques_to_binary
from common.schema import SchemaError

ROOT = Path(__file__).resolve().parent.parent


def test_labels_yaml_matches_code():
    y = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    assert list(y["task_b"].keys()) == TECHNIQUES
    assert list(y["task_a"].keys()) == [0, 1, 2, 3, 4, 5]


def test_technique_roundtrip():
    assert parse_techniques("NONE") == []
    assert parse_techniques("T4|T2") == ["T4", "T2"]
    B = techniques_to_binary(["T4|T2", "NONE", "T1|T5|T6"])
    assert B.tolist() == [[0, 1, 0, 1, 0, 0], [0] * 6, [1, 0, 0, 0, 1, 1]]
    assert binary_to_techniques(B[0]) == "T2|T4"  # canonical order
    assert binary_to_techniques(B[1]) == "NONE"
    for bad in ("T7", "T1|T1", "", None):
        with pytest.raises(LabelError):
            parse_techniques(bad)


def test_split_turns():
    assert split_turns("S: hi ||| R: what? ||| S: pay") == [("S", "hi"), ("R", "what?"), ("S", "pay")]
    assert split_turns("bKash Helpline: send code") == [(None, "bKash Helpline: send code")]
    assert split_turns("S: a ||| 🎁 S: 40K") == [("S", "a"), ("S", "🎁 40K")]


def test_load_train_preserves_order_ids_and_raw(tiny_data):
    _, cfg = tiny_data
    tr = load_train(cfg)
    raw = pd.read_csv(Path(cfg["data"]["dir"]) / "sample.csv")
    assert tr.id.tolist() == raw.id.tolist()
    assert tr.raw_text.tolist() == raw.text.tolist()
    yA, yB, yC = get_targets(tr)
    assert yB.shape == (len(tr), 6) and yA.shape == yC.shape == (len(tr),)


def test_load_train_rejects_bad_schema(tiny_data):
    d, cfg = tiny_data
    df = pd.read_csv(d / "sample.csv")
    df.loc[0, "label"] = 9
    df.loc[1, "id"] = df.loc[2, "id"]
    df.to_csv(d / "sample.csv", index=False)
    with pytest.raises(SchemaError) as e:
        load_train(cfg)
    assert "label" in str(e.value) and "duplicate" in str(e.value)


def test_text_like_na_is_kept(tiny_data):
    d, cfg = tiny_data
    df = pd.read_csv(d / "sample.csv")
    df.loc[0, "text"] = "NA"
    df.to_csv(d / "sample.csv", index=False)
    assert load_train(cfg).raw_text.iloc[0] == "NA"


def test_load_test_and_missing_test(tiny_data):
    d, cfg = tiny_data
    te = load_test(cfg)
    assert te.id.tolist() == list(range(101, 107))
    assert "label" not in te.columns
    (d / "test.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_test(cfg)
