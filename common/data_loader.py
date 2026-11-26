"""One loader for every model family.

    from common.data_loader import load_config, load_train, load_test, get_targets
    cfg = load_config()
    train = load_train(cfg)          # canonical DataFrame, original order, original ids
    yA, yB, yC = get_targets(train)  # (n,), (n,6), (n,)

Canonical columns: row_idx, id, raw_text, normalized_text, turns, n_turns, speakers,
is_conversation, script, (train only) label, action, techniques, T1..T6.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import REPO_ROOT
from .labels import TECHNIQUES, binary_to_techniques, techniques_to_binary
from .preprocessing import DEFAULT, NormConfig, normalize, script_profile
from .schema import SchemaError, check_ids, check_train_labels, find_text_col

# Leading junk (e.g. an injected emoji) before the speaker tag is kept as content.
_SPEAKER_RE = re.compile(r"^([^\w]*?)\s*([A-Za-z]{1,12})\s*:\s*(.*)$", re.S)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "base.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _kaggle_dirs():
    root = Path("/kaggle/input")
    if root.exists():
        for p in sorted(root.rglob("sample_submission.csv")):
            yield p.parent


def resolve_data_dir(cfg: dict) -> Path:
    d = cfg["data"]
    tried = []
    cands = []
    if os.getenv("SCAMSHIELD_DATA_DIR"):
        cands.append(Path(os.environ["SCAMSHIELD_DATA_DIR"]))
    cands.append(REPO_ROOT / d["dir"])
    cands += [REPO_ROOT / x for x in d.get("fallback_dirs", [])]
    cands += list(_kaggle_dirs())
    for c in cands:
        tried.append(str(c))
        if (c / d["train_file"]).exists() or (c / d["sample_submission_file"]).exists():
            return c.resolve()
    raise FileNotFoundError(
        f"No data dir found containing {d['train_file']} or {d['sample_submission_file']}. "
        f"Tried: {tried}. Set SCAMSHIELD_DATA_DIR."
    )


def output_dirs(cfg: dict) -> tuple[Path, Path]:
    o = cfg["output"]
    art = Path(os.getenv("SCAMSHIELD_ARTIFACTS_DIR", REPO_ROOT / o["artifacts_dir"]))
    out = Path(os.getenv("SCAMSHIELD_OUTPUTS_DIR", REPO_ROOT / o["outputs_dir"]))
    art.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return art, out


def file_fingerprint(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def split_turns(text: str, sep: str = "|||") -> list[tuple[str | None, str]]:
    """'S: hi ||| R: what?' -> [('S','hi'), ('R','what?')]. Single messages -> [(None, text)]."""
    if text is None or (isinstance(text, float) and text != text):
        return [(None, "")]
    parts = [p.strip() for p in str(text).split(sep)] if sep in str(text) else [str(text).strip()]
    turns = []
    for p in parts:
        m = _SPEAKER_RE.match(p)
        # Only treat a prefix as a speaker when the message is a multi-turn conversation;
        # single messages like 'bKash Helpline: ...' or 'Boss: ...' keep their prefix as content.
        if m and len(parts) > 1 and len(m.group(2)) <= 2:
            content = (m.group(1).strip() + " " + m.group(3).strip()).strip()
            turns.append((m.group(2).upper(), content))
        else:
            turns.append((None, p))
    return turns


def _read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: a message like "NA" or "None" must stay text, never become NaN.
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"text": str}, encoding="utf-8")


def _canonicalize(df: pd.DataFrame, cfg: dict, text_col: str, norm: NormConfig) -> pd.DataFrame:
    sep = cfg["data"].get("turn_separator", "|||")
    raw = df[text_col].astype(object).where(df[text_col].notna(), "")
    out = pd.DataFrame({"row_idx": np.arange(len(df)), "id": df[cfg["data"]["id_col"]].to_numpy()})
    out["raw_text"] = raw.astype(str).to_numpy()
    out["normalized_text"] = [normalize(t, norm) for t in out["raw_text"]]
    out["turns"] = [split_turns(t, sep) for t in out["raw_text"]]
    out["n_turns"] = out["turns"].map(len)
    out["speakers"] = out["turns"].map(lambda ts: "|".join(s or "-" for s, _ in ts))
    out["is_conversation"] = out["n_turns"] > 1
    out["script"] = out["raw_text"].map(lambda t: script_profile(t)["script"])
    return out


def load_train(cfg: dict | None = None, norm: NormConfig = DEFAULT, strict: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_data_dir(cfg) / cfg["data"]["train_file"]
    df = _read_csv(path)
    id_col = cfg["data"]["id_col"]
    text_col = find_text_col(df, cfg["data"]["text_col_candidates"], id_col)
    problems = check_ids(df, id_col, "train") + check_train_labels(df)
    if problems and strict:
        raise SchemaError("train file failed validation:\n  " + "\n  ".join(problems))
    out = _canonicalize(df, cfg, text_col, norm)
    out["label"] = df["label"].astype(int).to_numpy()
    out["action"] = df["action"].astype(int).to_numpy()
    tb = techniques_to_binary(df["techniques"])
    out["techniques"] = [binary_to_techniques(r) for r in tb]  # canonical order T1..T6
    for j, t in enumerate(TECHNIQUES):
        out[t] = tb[:, j]
    out.attrs.update(source=str(path), text_col=text_col, fingerprint=file_fingerprint(path), problems=problems)
    return out


def load_test(cfg: dict | None = None, norm: NormConfig = DEFAULT) -> pd.DataFrame:
    """Loads test.csv. Raises FileNotFoundError when it is not present (expected locally)."""
    cfg = cfg or load_config()
    path = resolve_data_dir(cfg) / cfg["data"]["test_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. test.csv is only available in the Kaggle environment.")
    df = _read_csv(path)
    id_col = cfg["data"]["id_col"]
    text_col = find_text_col(df, cfg["data"]["text_col_candidates"], id_col)
    problems = check_ids(df, id_col, "test")
    if problems:
        raise SchemaError("test file failed validation:\n  " + "\n  ".join(problems))
    out = _canonicalize(df, cfg, text_col, norm)
    out.attrs.update(source=str(path), text_col=text_col, fingerprint=file_fingerprint(path),
                     extra_columns=[c for c in df.columns if c not in (id_col, text_col)])
    return out


def load_sample_submission(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_data_dir(cfg) / cfg["data"]["sample_submission_file"]
    df = pd.read_csv(path, keep_default_na=False)
    problems = check_ids(df, cfg["data"]["id_col"], "sample_submission")
    if problems:
        raise SchemaError("; ".join(problems))
    df.attrs["fingerprint"] = file_fingerprint(path)
    return df


def test_available(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    try:
        return (resolve_data_dir(cfg) / cfg["data"]["test_file"]).exists()
    except FileNotFoundError:
        return False


def get_targets(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (train["label"].to_numpy(int), train[TECHNIQUES].to_numpy(np.int8), train["action"].to_numpy(int))
