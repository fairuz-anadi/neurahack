# CELL 4: Anadi classical ML experiments (E002-E009) + artifact export
# Run AFTER Cell 1 (needs: train, test, TECH, clean_text, URL_RE, ZERO_WIDTH, EMOJI_RE).
# Validation-only by default. Set RUN_FULL = True to fit on all train rows and predict the real test (Kaggle only).
import os, re, json, math, hashlib, time
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

SEED = 42
RUN_FULL = False
REPEATS = 3                      # repeated grouped CV (seeds 42, 43, 44)
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
FOLDS = None                     # TODO for Rabib-compatible artifacts: set to the repo's shared fold manifest as a list of (train_idx, val_idx)

# ---------------- groups (near-duplicate templates stay in one fold) ----------------
_gv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
_G = _gv.fit_transform(train["text_clean"])
_S = (_G @ _G.T).tocsr(); _S.data = (_S.data > 0.8).astype(int); _S.eliminate_zeros()
_, groups = connected_components(_S, directed=False)
print("rows", len(train), "| groups", len(set(groups)))

# ---------------- controlled noise augmentation (E005) ----------------
LOOK_REV = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}
MASH = ["wkrjfejejrn", "zzqwpo", "mnbvcx", "lkjsdfoiu", "xcvbnm", "qwrtpsdf"]

def corrupt_msg(m, rng, p=0.12):
    out = []
    for ch in m:
        r = rng.rand()
        if ch.isalpha() and r < p * 0.25: continue                                   # deletion
        if ch.isalpha() and r < p * 0.50: out.append(ch * rng.randint(2, 4)); continue  # repetition
        if ch.lower() in LOOK_REV and r < p * 0.75: out.append(LOOK_REV[ch.lower()]); continue  # lookalike
        if ch == " " and r < p: continue                                              # drop a space
        if ch.isalpha() and r < p * 0.85: out.append("​" + ch); continue         # invisible char
        out.append(ch)
    s = "".join(out); words = s.split(" ")
    if rng.rand() < 0.2 and words:                                                    # keyboard-mash word
        words.insert(rng.randint(0, len(words) + 1), MASH[rng.randint(len(MASH))])
    return " ".join(words)

def corrupt_text(t, rng):
    return " ||| ".join(corrupt_msg(m.strip(), rng) for m in str(t).split("|||"))

def corrupted_df(df, seed):
    rng = np.random.RandomState(seed)
    d = df.copy()
    d["text_raw"] = d["text_raw"].map(lambda t: corrupt_text(t, rng))
    d["text_clean"] = d["text_raw"].map(clean_text)
    return d

# ---------------- structured features (E004) ----------------
def _entropy(s):
    if not s: return 0.0
    _, c = np.unique(list(s), return_counts=True); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def struct_feats(df):
    raw = df["text_raw"].astype(str); cl = df["text_clean"].astype(str)
    n = raw.str.len().clip(lower=1)
    f = pd.DataFrame(index=df.index)
    f["n_msgs"] = df["n_msgs"].values
    f["loglen"] = np.log1p(n.values)
    f["digit"] = raw.map(lambda s: sum(c.isdigit() for c in s)).values / n.values
    f["upper"] = raw.map(lambda s: sum(c.isupper() for c in s)).values / n.values
    f["nonascii"] = raw.map(lambda s: sum(ord(c) > 127 for c in s)).values / n.values
    f["url"] = raw.map(lambda s: int(bool(URL_RE.search(s)))).values
    f["bn"] = raw.map(lambda s: int(bool(re.search(r"[ঀ-৿]", s)))).values
    f["latin"] = raw.map(lambda s: int(bool(re.search(r"[A-Za-z]", s)))).values
    f["rep"] = raw.map(lambda s: len(re.findall(r"(.)\1{2,}", s))).values
    f["zw"] = raw.map(lambda s: len(ZERO_WIDTH.findall(s))).values
    f["ent"] = raw.map(_entropy).values
    f["kept"] = np.array([len(c) / max(len(r), 1) for c, r in zip(cl, raw)])
    f["emoji"] = raw.map(lambda s: len(EMOJI_RE.findall(s))).values
    f["ussd"] = raw.map(lambda s: int("*" in s and "#" in s)).values
    f["money"] = raw.str.contains(r"(?:\btk\b|taka|bdt|৳|টাকা)", case=False, regex=True).astype(int).values
    f["last_len"] = raw.map(lambda s: np.log1p(len(str(s).split("|||")[-1]))).values
    return f.values.astype(float)

# ---------------- model family ----------------
def _make_head(kind, y):
    cnt = np.bincount(y); present = cnt[cnt > 0]
    if kind == "svc" and present.min() >= 3 and len(present) >= 2:
        return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=SEED), cv=3, method="sigmoid")
    if kind == "cnb":
        return ComplementNB(alpha=0.3)
    return LogisticRegression(C=20, class_weight="balanced", max_iter=3000, random_state=SEED)

def _proba(m, X, n=6):
    p = np.zeros((X.shape[0], n)); p[:, m.classes_] = m.predict_proba(X); return p

class ClassicModel:
    """cfg: heads (list of 'lr'/'svc'/'cnb'), aug (noisy copies per train row), struct (bool)."""
    def __init__(self, cfg): self.cfg = cfg
    def _X(self, df, fit=False):
        txt = df["text_clean"]
        if fit:
            self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, dtype=np.float32)
            self.wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+", dtype=np.float32)
            tf = hstack([self.cv.fit_transform(txt), self.wv.fit_transform(txt)]).tocsr()
        else:
            tf = hstack([self.cv.transform(txt), self.wv.transform(txt)]).tocsr()
        if not self.cfg.get("struct"): return tf, tf
        S = struct_feats(df)
        if fit: self.sc = StandardScaler().fit(S)
        S = np.clip(self.sc.transform(S), -5, 5) * 0.3
        return tf, hstack([tf, csr_matrix(S)]).tocsr()
    def fit(self, tr, seed=SEED):
        parts = [tr] + [corrupted_df(tr, seed * 1000 + k) for k in range(self.cfg.get("aug", 0))]
        tr = pd.concat(parts, ignore_index=True)
        tf, full = self._X(tr, fit=True)
        self.m = {}
        for kind in self.cfg["heads"]:
            X = tf if kind == "cnb" else full
            h = {"label": _make_head(kind, tr["y"].values).fit(X, tr["y"].values),
                 "action": _make_head(kind, tr["act"].values).fit(X, tr["act"].values), "tech": []}
            for t in TECH:
                yt = tr[t].values
                if yt.min() == yt.max(): h["tech"].append(float(yt[0]))
                else: h["tech"].append(_make_head(kind if np.bincount(yt).min() >= 3 else "lr", yt).fit(X, yt))
            self.m[kind] = h
        return self
    def predict(self, df):
        tf, full = self._X(df)
        out = {"label": 0, "action": 0, "tech": 0}; k = len(self.m)
        for kind, h in self.m.items():
            X = tf if kind == "cnb" else full
            out["label"] = out["label"] + _proba(h["label"], X) / k
            out["action"] = out["action"] + _proba(h["action"], X) / k
            tp = np.zeros((X.shape[0], 6))
            for j, mt in enumerate(h["tech"]):
                tp[:, j] = mt if isinstance(mt, float) else mt.predict_proba(X)[:, list(mt.classes_).index(1)]
            out["tech"] = out["tech"] + tp / k
        return out

# ---------------- decoding + scoring ----------------
def decode(lp, ap, tp, constrain):
    lab = lp.argmax(1); act = ap.argmax(1); techs = []
    for i in range(len(lab)):
        if constrain and lab[i] in (0, 5): act[i] = 0
        if lab[i] == 0 or (constrain and lab[i] == 5): techs.append("NONE"); continue
        s = [TECH[j] for j in range(6) if tp[i, j] >= 0.5] or [TECH[int(tp[i].argmax())]]
        techs.append("|".join(s))
    return lab, act, techs

def f1s(df, o, constrain):
    lab, act, techs = decode(o["label"], o["action"], o["tech"], constrain)
    P = np.array([[int(t in s.split("|")) for t in TECH] for s in techs])
    return (f1_score(df["y"], lab, average="macro", labels=range(6), zero_division=0),
            f1_score(df[TECH].values, P, average="macro", zero_division=0),
            f1_score(df["act"], act, average="macro", labels=range(6), zero_division=0))

def calib_score(lp, y):
    mp = lp.max(1, keepdims=True); ok = (lp.argmax(1) == y).astype(int)
    p = cross_val_predict(LogisticRegression(C=10, random_state=SEED), mp, ok, cv=5, method="predict_proba")[:, 1]
    return 1 - float(np.mean((p - ok) ** 2))      # 1 - Brier (higher is better), a proxy only

def cv_splits(rep):
    if FOLDS is not None: return FOLDS
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep).split(train, train["y"], groups))

CONFIGS = {
    "E0 lr baseline":          {"heads": ["lr"], "aug": 0, "struct": False},
    "E1 lr +noise x1":         {"heads": ["lr"], "aug": 1, "struct": False},
    "E2 lr +noise x2":         {"heads": ["lr"], "aug": 2, "struct": False},
    "E3 svc +noise x1":        {"heads": ["svc"], "aug": 1, "struct": False},
    "E4 lr +struct +noise x1": {"heads": ["lr"], "aug": 1, "struct": True},
    "E5 lr+svc +noise x1":     {"heads": ["lr", "svc"], "aug": 1, "struct": False},
    "E6 lr+svc+cnb +noise x1": {"heads": ["lr", "svc", "cnb"], "aug": 1, "struct": False},
}

def run_config(name, cfg):
    res = {k: [] for k in ["A", "B0", "B1", "C0", "C1", "A_noisy", "calib"]}; first = None
    for rep in range(REPEATS):
        oof = {k: np.zeros((len(train), 6)) for k in ["label", "action", "tech"]}
        noisy = np.zeros((len(train), 6))
        for a, b in cv_splits(rep):
            m = ClassicModel(cfg).fit(train.iloc[a], seed=SEED + rep)
            r = m.predict(train.iloc[b])
            for k in oof: oof[k][b] = r[k]
            noisy[b] = m.predict(corrupted_df(train.iloc[b], 777 + rep))["label"]   # noise only on validation copies
        a0, b0, c0 = f1s(train, oof, False); a1, b1, c1 = f1s(train, oof, True)
        res["A"].append(a0); res["B0"].append(b0); res["B1"].append(b1); res["C0"].append(c0); res["C1"].append(c1)
        res["A_noisy"].append(f1_score(train["y"], noisy.argmax(1), average="macro", labels=range(6), zero_division=0))
        res["calib"].append(calib_score(oof["label"], train["y"].values))
        if rep == 0: first = oof
    return {k: float(np.mean(v)) for k, v in res.items()}, first

results, oofs = {}, {}
t0 = time.time()
for name, cfg in CONFIGS.items():
    results[name], oofs[name] = run_config(name, cfg)
    print(f"{name:26s} done {time.time() - t0:5.0f}s")
tbl = pd.DataFrame(results).T.round(3)
tbl.columns = ["A_F1", "B_F1(free)", "B_F1(NONE for 0,5)", "C_F1(free)", "C_F1(0 for 0,5)", "A_F1_noisy", "calib(1-Brier)"]
print("\nMean over %d repeated grouped 5-fold runs (real rows only; proxies, not Kaggle scores):" % REPEATS)
print(tbl.to_string())

# ---------------- pick winner: best A, tie -> best noisy A ----------------
BEST = (tbl["A_F1"] + 0.5 * tbl["A_F1_noisy"]).idxmax()
print("\nBEST config:", BEST)

# ---------------- artifact export (names follow TEAM_CONTRACT: oof_A/B/C_<name>.npy) ----------------
def save_artifacts(name, tag, oof, tst=None, test_ids=None):
    for task, key in (("A", "label"), ("B", "tech"), ("C", "action")):
        np.save(f"{OUT_DIR}/oof_{task}_{tag}.npy", oof[key].astype(np.float32))
        if tst is not None: np.save(f"{OUT_DIR}/test_{task}_{tag}.npy", tst[key].astype(np.float32))
    np.save(f"{OUT_DIR}/oof_ids.npy", train["id"].values)
    if test_ids is not None: np.save(f"{OUT_DIR}/test_ids.npy", test_ids)
    meta = {"model": tag, "config": CONFIGS[name], "seed": SEED, "oof_rows": int(len(train)),
            "columns": {"A": "classes 0-5", "B": "T1-T6 independent probabilities", "C": "actions 0-5"},
            "train_fingerprint": hashlib.md5((",".join(map(str, train["id"])) + ",".join(map(str, train["y"]))).encode()).hexdigest(),
            "folds": "shared manifest" if FOLDS is not None else "LOCAL groups (not the shared folds): validate before giving to Rabib"}
    json.dump(meta, open(f"{OUT_DIR}/metadata.json", "w"), indent=2)

TAG = "classic_v2"
if RUN_FULL:
    assert test is not None, "attach the competition data and rerun Cell 1"
    final = ClassicModel(CONFIGS[BEST]).fit(train)
    parts = [final.predict(test.iloc[i:i + 100000]) for i in range(0, len(test), 100000)]
    tst = {k: np.vstack([p[k] for p in parts]) for k in parts[0]}
    save_artifacts(BEST, TAG, oofs[BEST], tst, test["id"].values)
    lab, act, techs = decode(tst["label"], tst["action"], tst["tech"], True)
    mp = tst["label"].max(1, keepdims=True); ok = (oofs[BEST]["label"].argmax(1) == train["y"].values).astype(int)
    cal = LogisticRegression(C=10, random_state=SEED).fit(oofs[BEST]["label"].max(1, keepdims=True), ok)
    conf = np.clip(cal.predict_proba(mp)[:, 1], 0.05, 0.98)
    sub = pd.DataFrame({"id": test["id"], "label": lab.astype(int), "techniques": techs, "action": act.astype(int), "confidence": np.round(conf, 4)})
    sub.to_csv(f"{OUT_DIR}/submission_{TAG}.csv", index=False)
    print("saved submission_%s.csv" % TAG, sub.shape, "| label mix:", np.round(np.bincount(lab, minlength=6) / len(lab), 3))
else:
    save_artifacts(BEST, TAG, oofs[BEST])
    print("validation-only: saved oof artifacts for", BEST, "(set RUN_FULL = True for full test inference)")
