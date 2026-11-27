"""Duo notebook: v3 Part A with XLM-R large AND XLM-R base, blended with TF-IDF; no embedding/FT families (time).

    python scripts/make_duo_notebook.py notebooks/10_v3_large.ipynb notebooks/12_duo.ipynb

Changes from 10_v3_large: 1 final XLM-R-large seed (time); new cell A2 trains xlm-roberta-base on the same folds and
saves oof/test_xlmrb_*.npy (wrapped in try/except: on failure the notebook continues without it); Part B removed; the
final cell treats the embedding family as optional, loads xlmrb and adds a 'duo' candidate (TF-IDF + large + base
weights and threshold by CV). The candidate with the higher CV proxy is written to submission.csv.
"""
import json
import sys

A2_MD = """## A2. Second transformer: `xlm-roberta-base` (same folds, same synthetic data)
XLM-R base and XLM-R large make different errors, so the final cell can blend them with TF-IDF. Same multi-task head,
lr 3e-5, batch 32, 3 epochs; CV on the first repeat of the shared folds; one final fit for the test predictions.
If this step fails, its files are removed and the final cell runs without it."""

A2 = '''import traceback, glob as _glob
_keep = (MODEL_NAME, TOK, LR, BATCH)
try:
    t0 = time.time()
    MODEL_NAME, LR, BATCH = "xlm-roberta-base", 3e-5, 32
    TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
    OOF["xlmrb"] = run_cv(XlmrModel, "xlmrb", splits[:5 * XLMR_REPEATS])
    print("XLM-R base CV:", scores(train.y.values, train.act.values, TB_real, *OOF["xlmrb"][0], PL_noisy=OOF["xlmrb"][1]))
    xb = XlmrModel().fit(d_all, TB_all, seed=SEED); Pxb = xb.predict(uniq); xb.free()
    save_team("xlmrb", OOF["xlmrb"][0], Pxb)
    print(f"xlmrb done ({time.time() - t0:.0f}s)")
except Exception:
    traceback.print_exc()
    for f in _glob.glob(os.path.join(OUT_DIR, "*_xlmrb_*.npy")): os.remove(f)
    print("XLM-R base failed: files removed; the final cell runs without it")
finally:
    MODEL_NAME, TOK, LR, BATCH = _keep
'''

DUO = '''if 'xlmrb' in O and 'xlmr' in O:   # TF-IDF + XLM-R large + XLM-R base, weights and threshold by CV
    _fam = ['tfidf', 'xlmr', 'xlmrb']
    _mx = lambda w, src: [sum(w[i] * src[k][j] for i, k in enumerate(_fam)) for j in range(3)]
    _res = max(((proxy(*_mx(w, O), t)[0], w, t) for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3)
                if abs(sum(w) - 1) < 1e-6 for t in np.round(np.arange(0.3, 0.71, 0.05), 2)), key=lambda r: r[0])
    _wd, _td = np.round(_res[1], 1), _res[2]
    print('duo weights', dict(zip(_fam, _wd)), 'thr', _td)
    CANDS['duo'] = (_mx(_wd, O), lambda: _mx(_wd, TS), _td)
SCORES = {k: proxy(*v[0], v[2]) for k, v in CANDS.items()}'''


def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


def src(c):
    return "".join(c["source"])


def setsrc(c, s):
    c["source"] = s.splitlines(keepends=True)


def main():
    a, out = sys.argv[1:3]
    nb = json.load(open(a, encoding="utf-8"))
    C = nb["cells"]
    setsrc(C[3], rep(src(C[3]), 'os.environ.get("SS_FINAL_SEEDS", 2)', 'os.environ.get("SS_FINAL_SEEDS", 1)'))
    b0 = next(i for i, c in enumerate(C) if c["cell_type"] == "markdown" and src(c).startswith("# Part B"))
    c0 = next(i for i, c in enumerate(C) if c["cell_type"] == "markdown" and src(c).startswith("# Part C"))
    for c in C[b0:c0]:
        assert c["cell_type"] == "markdown" or "sentence_transformers" in src(c) or "folds_5x5" in src(c) or "emb" in src(c), src(c)[:200]
    fin = C[c0 + 1]
    s = src(fin)
    s = rep(s, "O, TS = {'emb': emb_family('oof')}, {'emb': emb_family('test')}",
            "O, TS = {}, {}\nif all(os.path.exists(f'{OUT}{s}_{k}_{t}.npy') for k in EMB for s in ['oof', 'test'] for t in 'ABC'):\n"
            "    O['emb'], TS['emb'] = emb_family('oof'), emb_family('test')")
    s = rep(s, "for name in ['tfidf', 'xlmr', 'ft']:", "for name in ['tfidf', 'xlmr', 'xlmrb', 'ft']:")
    s = rep(s, "SCORES = {k: proxy(*v[0], v[2]) for k, v in CANDS.items()}", DUO)
    compile(s, "final", "exec")
    setsrc(fin, s)
    compile(A2, "a2", "exec")
    lines = lambda t: t.splitlines(keepends=True)
    new = [{"cell_type": "markdown", "metadata": {}, "source": lines(A2_MD)},
           {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(A2)}]
    nb["cells"] = C[:b0] + new + C[c0:]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1); f.write("\n")
    print("wrote", out, len(nb["cells"]), "cells; removed Part B cells", b0, "to", c0 - 1)


if __name__ == "__main__":
    main()
