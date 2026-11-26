"""Public external datasets -> one training CSV with FULL or PARTIAL (masked) labels.

    python -m models.finetune.external --out /kaggle/working/external_v1.csv [--cap 600]

Downloads each dataset from its original Mendeley Data source at run time (public API, no login), so the
submitted notebook rebuilds the file from scratch. Licences checked on the source pages (26 Nov 2026).

Label policy (never force binary spam labels into our 6 classes):
  * normal / ham                         -> class 0, techniques NONE, action 0   (full label; official 0/5 rule)
  * promotional                          -> class 5, techniques NONE, action 0   (full label)
  * smishing / financial scam            -> Task A in {1,2,3,4} (partial), Task B/C unknown (masked)
  * generic "spam" (ads + scams + gambling mixed) -> Task A in {1,2,3,4,5} (partial), Task B/C masked
Rows are cleaned (empty / <8 chars / >1500 chars dropped), exact-deduplicated within and across sources
(normalized text), capped per (source, label) with a seeded sample, and any row with char TF-IDF cosine >= 0.8 to a
sample.csv row is removed (so no held-out validation record leaks in). External rows are training-only.
Bangalabarta (CC BY-NC-SA 4.0) is excluded unless --include-nc is passed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

FULL0 = {"label": 0, "allowed": "", "techniques": "NONE", "action": 0}
FULL5 = {"label": 5, "allowed": "", "techniques": "NONE", "action": 0}
FRAUD = {"label": -1, "allowed": "1|2|3|4", "techniques": "", "action": -1}
SPAMMY = {"label": -1, "allowed": "1|2|3|4|5", "techniques": "", "action": -1}

SOURCES = [
    {"key": "bttc", "doi": "10.17632/5wrm959d6f.1", "id": "5wrm959d6f", "version": 1, "licence": "CC BY 4.0",
     "file": "BTTC.csv", "text": "Text_Clean", "label": "Label",
     "map": {"HAM": FULL0, "PROMO": FULL5, "SPAM": SPAMMY}},
    {"key": "bangla_spam_ham", "doi": "10.17632/hhd43jbv9p.1", "id": "hhd43jbv9p", "version": 1, "licence": "CC BY 4.0",
     "file": ".csv", "text": "sms", "label": "spamorham", "map": {"ham": FULL0, "spam": SPAMMY}},
    {"key": "bangalabarta", "doi": "10.17632/jfkfbw3gzh.3", "id": "jfkfbw3gzh", "version": 3, "licence": "CC BY-NC-SA 4.0",
     "nc": True, "file": ".csv", "text": "text", "label": "label", "map": {"normal": FULL0, "promo": FULL5, "smish": FRAUD}},
    {"key": "financial_scams", "doi": "10.17632/znsk27yk3h.2", "id": "znsk27yk3h", "version": 2, "licence": "CC BY 4.0",
     "file": ".csv", "text": "message", "label": "label", "map": {"ham": FULL0, "scam": FRAUD}},
    {"key": "sms_phishing_mishra", "doi": "10.17632/f45bkkt8pr.1", "id": "f45bkkt8pr", "version": 1, "licence": "CC BY 4.0",
     "file": ".csv", "text": "TEXT", "label": "LABEL", "map": {"ham": FULL0, "smishing": FRAUD, "spam": SPAMMY}},
]
URL = "https://data.mendeley.com/public-api/zip/{id}/download/{version}"


def _csvs_in_zip(data: bytes):
    """Yield (name, bytes) for every CSV in a zip, descending into nested zips. Never writes to disk."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/") or ".." in name:
                continue
            raw = zf.read(info)
            if name.lower().endswith(".zip"):
                yield from _csvs_in_zip(raw)
            elif name.lower().endswith(".csv"):
                yield name, raw


def fetch(src: dict, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / f"{src['id']}_v{src['version']}.zip"
    if not zpath.exists():
        with urllib.request.urlopen(URL.format(**src), timeout=120) as r:
            zpath.write_bytes(r.read())
    files = [(n, b) for n, b in _csvs_in_zip(zpath.read_bytes()) if n.endswith(src["file"])]
    if len(files) != 1:
        raise RuntimeError(f"{src['key']}: expected one CSV ending with {src['file']!r}, found {[n for n, _ in files]}")
    df = pd.read_csv(io.BytesIO(files[0][1]), encoding="utf-8")
    return df[[src["text"], src["label"]]].rename(columns={src["text"]: "text", src["label"]: "raw_label"})


def drop_near_sample(ext: pd.DataFrame, sample_texts, threshold: float = 0.8) -> tuple[pd.DataFrame, int]:
    """Remove external rows that duplicate / nearly duplicate any sample.csv row (char TF-IDF cosine)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    norm = lambda x: x.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    e, smp = norm(ext["text"]), norm(pd.Series(list(sample_texts)))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pd.concat([e, smp]))
    mx = (vec.transform(e) @ vec.transform(smp).T).max(axis=1).toarray().ravel()
    keep = (mx < threshold) & ~e.isin(set(smp))
    return ext[keep].reset_index(drop=True), int((~keep).sum())


def build(cache: Path, cap: int = 300, seed: int = 42, include_nc: bool = False,
          sample_texts=None) -> tuple[pd.DataFrame, dict]:
    parts, report = [], {}
    for src in SOURCES:
        if src.get("nc") and not include_nc:
            report[src["key"]] = {"doi": src["doi"], "licence": src["licence"], "excluded": "non-commercial licence"}
            continue
        df = fetch(src, cache)
        df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["raw_label"] = df["raw_label"].astype(str).str.strip().str.lower()
        mapping = {k.lower(): v for k, v in src["map"].items()}
        n0 = len(df)
        df = df[df["raw_label"].isin(mapping) & df["text"].str.len().between(8, 1500)]
        df = df.drop_duplicates("text")
        rows = []
        for (lab,), g in df.groupby(["raw_label"]):
            g = g.sample(min(cap, len(g)), random_state=seed) if cap else g
            rows.append(g.assign(**mapping[lab]))
        out = pd.concat(rows, ignore_index=True).assign(source=src["key"])
        report[src["key"]] = {"doi": src["doi"], "licence": src["licence"], "raw_rows": n0,
                              "kept_by_label": out["raw_label"].value_counts().to_dict()}
        parts.append(out)
    ext = pd.concat(parts, ignore_index=True)
    key = ext["text"].str.lower()
    before = len(ext)
    ext = ext[~key.duplicated()].reset_index(drop=True)
    report["cross_source_duplicates_removed"] = before - len(ext)
    if sample_texts is not None:
        ext, n_drop = drop_near_sample(ext, sample_texts)
        report["removed_near_sample_rows"] = n_drop
    ext.insert(0, "id", [f"ext_{i:06d}" for i in range(len(ext))])
    report["total"] = len(ext)
    report["full_label_counts"] = ext.loc[ext["label"] >= 0, "label"].value_counts().sort_index().to_dict()
    report["partial_counts"] = ext.loc[ext["label"] < 0, "allowed"].value_counts().to_dict()
    return ext[["id", "text", "label", "allowed", "techniques", "action", "source"]], report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="external_v1.csv")
    ap.add_argument("--cache", default=None, help="where the downloaded zips are kept")
    ap.add_argument("--cap", type=int, default=300, help="max rows per (source, label)")
    ap.add_argument("--include-nc", action="store_true", help="also use the CC BY-NC-SA dataset (Bangalabarta)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    cache = Path(args.cache) if args.cache else Path(args.out).resolve().parent / "external_cache"
    from common.data_loader import load_train
    ext, rep = build(cache, args.cap, args.seed, args.include_nc, load_train()["raw_text"].tolist())
    ext.to_csv(args.out, index=False)
    print(json.dumps(rep, ensure_ascii=False))
    print("wrote", args.out, len(ext), "rows")


if __name__ == "__main__":
    main()
