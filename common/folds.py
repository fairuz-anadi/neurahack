"""Shared grouped CV. Everyone reads folds/folds.csv (id,fold); nobody makes their own splits.

Groups: rows are merged into one group when they share a template key (digits/URLs/case masked)
or their char TF-IDF cosine >= near_dup_threshold. StratifiedGroupKFold (seed 42) then keeps each
group inside one fold, stratified on Task A.

Extra manifests for proxy scores (same id,fold format):
  folds_repeats.csv  id,repeat,fold  - 5 grouped re-splits, for mean +/- std
  folds_novel.csv    scenario clusters (complete-linkage, ~3 rows each) held out whole -> novel-scenario proxy
  folds_domain.csv   leave-brand-out (all rows mentioning the same brand in one fold) -> domain proxy
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from . import REPO_ROOT
from .labels import TASK_A_CLASSES, TASK_C_CLASSES, TECHNIQUES

FOLDS_DIR = Path(os.getenv("SCAMSHIELD_FOLDS_DIR", REPO_ROOT / "folds"))


def _fdir(fold_dir) -> Path:
    """Resolve at call time so FOLDS_DIR / SCAMSHIELD_FOLDS_DIR can be changed after import."""
    if fold_dir is not None:
        return Path(fold_dir)
    return Path(os.getenv("SCAMSHIELD_FOLDS_DIR", FOLDS_DIR))

# Brands/operators seen in Bangladesh SMS traffic. Domain knowledge, not derived from test data.
BRANDS = {
    "bkash": r"b\s*kash|bksah|বিকাশ", "nagad": r"nagad|নগদ", "rocket": r"\brocket\b|dbbl|dutch.?bangla",
    "upay": r"\bupay\b", "daraz": r"daraz|দারাজ", "grameenphone": r"grameenphone|\bgp\b|mygp",
    "robi": r"\brobi\b|রবি", "banglalink": r"banglalink", "airtel": r"airtel", "teletalk": r"teletalk",
    "pathao": r"pathao|পাঠাও", "sundarban": r"sundarban", "sa_paribahan": r"sa paribahan|\bsap\b",
    "redx": r"\bredx\b", "steadfast": r"steadfast", "dhl": r"\bdhl\b", "fedex": r"fedex",
    "city_bank": r"city bank", "brac": r"brac", "titas": r"titas", "breb": r"breb|desco|dpdc",
    "unilever": r"unilever", "aarong": r"aarong", "shwapno": r"shwapno", "bdjobs": r"bdjobs",
    "telegram": r"telegram|t\[?\.\]?me", "whatsapp": r"whats\s*app|হোয়াটসঅ্যাপ", "facebook": r"facebook|facccebook",
    "youtube": r"youtube", "pizza_hut": r"pizza hut", "foodpanda": r"foodpanda",
}
_BRAND_RES = {k: re.compile(v, re.I) for k, v in BRANDS.items()}


def template_key(text: str) -> str:
    s = text.lower()
    s = re.sub(r"(https?|hxxps?)://\S+|\S+\[\.\]\S+", " <url> ", s)
    s = re.sub(r"[0-9০-৯]+", "0", s)
    s = re.sub(r"[^\w<> ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def detect_brands(text: str) -> list[str]:
    return [k for k, r in _BRAND_RES.items() if r.search(text)]


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def similarity_matrix(texts) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    return S


def build_groups(texts, threshold: float, S: np.ndarray | None = None) -> np.ndarray:
    texts = list(texts)
    n = len(texts)
    uf = _UF(n)
    first = {}
    for i, t in enumerate(texts):
        k = template_key(t)
        if k in first:
            uf.union(i, first[k])
        else:
            first[k] = i
    S = similarity_matrix(texts) if S is None else S
    ii, jj = np.where(np.triu(S, 1) >= threshold)
    for a, b in zip(ii, jj):
        uf.union(int(a), int(b))
    roots = [uf.find(i) for i in range(n)]
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def scenario_clusters(S: np.ndarray, rows_per_cluster: int) -> np.ndarray:
    """Complete-linkage clusters on cosine distance. Groups related scenarios even when no pair is a
    near-duplicate (the case for the 86-row sample)."""
    n = S.shape[0]
    k = max(2, n // max(1, rows_per_cluster))
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="complete").fit_predict(D)


def choose_n_splits(y, groups, max_splits: int) -> int:
    y = np.asarray(y)
    min_class_groups = min(len(np.unique(groups[y == c])) for c in np.unique(y))
    return int(max(2, min(max_splits, min_class_groups, len(np.unique(groups)))))


def grouped_folds(y, groups, n_splits: int, seed: int) -> np.ndarray:
    fold = np.full(len(y), -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold[va] = k
    assert (fold >= 0).all()
    return fold


def manifest_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()[:16]


def coverage_report(train: pd.DataFrame, fold: np.ndarray) -> dict:
    rep = {}
    for k in sorted(np.unique(fold)):
        m = fold == k
        sub = train[m]
        rep[int(k)] = {
            "n": int(m.sum()),
            "task_a": {int(c): int((sub.label == c).sum()) for c in TASK_A_CLASSES},
            "task_c": {int(c): int((sub.action == c).sum()) for c in TASK_C_CLASSES},
            "techniques": {t: int(sub[t].sum()) for t in TECHNIQUES},
        }
    return rep


def leakage_report(train: pd.DataFrame, fold: np.ndarray, S: np.ndarray, warn_at: float = 0.6) -> list[dict]:
    """Cross-fold pairs that are still fairly similar (below the grouping threshold)."""
    ii, jj = np.where(np.triu(S, 1) >= warn_at)
    out = []
    for a, b in zip(ii, jj):
        if fold[a] != fold[b]:
            out.append({"id_a": train.id.iloc[a].item(), "id_b": train.id.iloc[b].item(),
                        "cosine": round(float(S[a, b]), 3)})
    return sorted(out, key=lambda d: -d["cosine"])


def make_all_folds(train: pd.DataFrame, cfg: dict, out_dir: Path | None = None) -> dict:
    fc, seed = cfg["folds"], int(cfg["seed"])
    out_dir = _fdir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = train["normalized_text"].tolist()
    y = train["label"].to_numpy()
    S = similarity_matrix(texts)

    groups = build_groups(texts, fc["near_dup_threshold"], S)
    k = choose_n_splits(y, groups, fc["max_splits"])
    fold = grouped_folds(y, groups, k, seed)
    main = pd.DataFrame({"id": train.id.to_numpy(), "fold": fold})
    main.to_csv(out_dir / "folds.csv", index=False, lineterminator="\n")

    reps = []
    for r in range(int(fc["n_repeats"])):
        reps.append(pd.DataFrame({"id": train.id.to_numpy(), "repeat": r,
                                  "fold": grouped_folds(y, groups, k, seed + r)}))
    reps = pd.concat(reps, ignore_index=True)
    reps.to_csv(out_dir / "folds_repeats.csv", index=False, lineterminator="\n")

    g_novel = scenario_clusters(S, int(fc["novel_rows_per_cluster"]))
    k_novel = choose_n_splits(y, g_novel, fc["max_splits"])
    novel = pd.DataFrame({"id": train.id.to_numpy(), "fold": grouped_folds(y, g_novel, k_novel, seed)})
    novel.to_csv(out_dir / "folds_novel.csv", index=False, lineterminator="\n")

    brands = [detect_brands(t) for t in train["raw_text"]]
    g_dom = np.array([("brand:" + b[0]) if b else f"grp:{g}" for b, g in zip(brands, groups)])
    _, g_dom_i = np.unique(g_dom, return_inverse=True)
    k_dom = choose_n_splits(y, g_dom_i, fc["max_splits"])
    domain = pd.DataFrame({"id": train.id.to_numpy(), "fold": grouped_folds(y, g_dom_i, k_dom, seed)})
    domain.to_csv(out_dir / "folds_domain.csv", index=False, lineterminator="\n")

    meta = {
        "seed": seed,
        "dataset_fingerprint": train.attrs.get("fingerprint"),
        "n_rows": int(len(train)),
        "main": {"n_splits": k, "n_groups": int(groups.max() + 1), "threshold": fc["near_dup_threshold"],
                 "fingerprint": manifest_fingerprint(main),
                 "multi_row_groups": int((np.bincount(groups) > 1).sum()),
                 "coverage": coverage_report(train, fold),
                 "leakage_risk_pairs": leakage_report(train, fold, S)[:25]},
        "repeats": {"n_repeats": int(fc["n_repeats"]), "fingerprint": manifest_fingerprint(reps)},
        "novel": {"n_splits": k_novel, "n_groups": int(g_novel.max() + 1), "rows_per_cluster": fc["novel_rows_per_cluster"],
                  "cluster_sizes": np.bincount(g_novel).tolist(),
                  "fingerprint": manifest_fingerprint(novel)},
        "domain": {"n_splits": k_dom, "n_brand_rows": int(sum(bool(b) for b in brands)),
                   "brand_counts": pd.Series([b[0] for b in brands if b]).value_counts().to_dict(),
                   "fingerprint": manifest_fingerprint(domain)},
    }
    with open(out_dir / "folds_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def load_folds(train: pd.DataFrame, name: str = "folds", fold_dir: Path | None = None) -> np.ndarray:
    """Return fold numbers aligned to train row order, verified by id."""
    m = pd.read_csv(_fdir(fold_dir) / f"{name}.csv")
    if "repeat" in m.columns:
        raise ValueError("use load_repeat_folds for folds_repeats.csv")
    return _align(train, m)


def load_repeat_folds(train: pd.DataFrame, fold_dir: Path | None = None) -> list[np.ndarray]:
    m = pd.read_csv(_fdir(fold_dir) / "folds_repeats.csv")
    return [_align(train, g[["id", "fold"]]) for _, g in m.groupby("repeat", sort=True)]


def _align(train: pd.DataFrame, m: pd.DataFrame) -> np.ndarray:
    if set(m.id) != set(train.id) or len(m) != len(train):
        raise ValueError("fold manifest ids do not match the training ids; regenerate with scripts/create_folds.py")
    return train[["id"]].merge(m, on="id", how="left", validate="one_to_one")["fold"].to_numpy(int)


def fold_fingerprint(name: str = "folds", fold_dir: Path | None = None) -> str:
    return manifest_fingerprint(pd.read_csv(_fdir(fold_dir) / f"{name}.csv"))
