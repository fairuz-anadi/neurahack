"""Remove pseudo-labeling (Rulebook v1.3 §5.3 bans training on test text) and write two notebooks:
base (xlm-roberta-base) and large (xlm-roberta-large, lr 1.5e-5, batch 16).

    python scripts/make_v2_notebooks.py SRC.ipynb OUT_BASE.ipynb OUT_LARGE.ipynb
"""
import json
import re
import sys


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


def set_src(cell, s):
    lines = s.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def clear_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None


def base_edits(nb):
    C = nb["cells"]
    get = lambda i: "".join(C[i]["source"])

    s = get(0)
    s = rep(s, "* **Pseudo-labeling** (permitted by Rulebook Section 5): confident test predictions on which both models agree\n"
               "  are added to training for one more round. This is done fully by code and uses only model predictions.\n",
            "* **No test text is used for training** (Rulebook v1.3, Section 5, item 3): no pseudo-labeling, self-training or\n"
            "  domain adaptation. Every model is trained only on `sample.csv` and the synthetic rows; `test.csv` is used for inference only.\n")
    set_src(C[0], s)

    s = get(1)
    s = rep(s, " + pseudo-labeling (Anadi)", " (Anadi)")
    set_src(C[1], s)

    s = get(3)
    s = re.sub(r"USE_PSEUDO\s*=.*\n", "", s)
    s = re.sub(r"PSEUDO_PER_CLASS\s*=.*\n", "", s)
    s = rep(s, "LR           = 3e-5", 'LR           = float(os.environ.get("SS_LR", 3e-5))')
    s = rep(s, "BATCH        = 32", 'BATCH        = int(os.environ.get("SS_BATCH", 32))')
    set_src(C[3], s)

    s = get(7)
    s = rep(s, "(XLM-R, pseudo-label check)", "(XLM-R)")
    s = rep(s, " + optional pseudo rows.", ".")
    set_src(C[7], s)

    s = get(10)
    s = rep(s, ", plus an in-fold check that pseudo-labeling helps", "")
    set_src(C[10], s)

    s = get(11)
    i = s.index("# ---- does pseudo-labeling help?")
    set_src(C[11], s[:i].rstrip() + "\n")

    s = get(12)
    s = rep(s, ", pseudo-labeling round", "")
    set_src(C[12], s)

    s = get(13)
    a = s.index("P_test, Pt, Px = fit_predict_all(d_all, TB_all, \"round 1\")")
    b = s.index("lab, act, tb = decode(")
    s = s[:a] + 'P_test, Pt, Px = fit_predict_all(d_all, TB_all, "final")\n\n' + s[b:]
    s = rep(s, "# OOF = shared-split CV without pseudo-labels; test = final models (after the pseudo-label round if enabled).",
            "# OOF = shared-split CV; test = final models trained on synthetic + real rows only.")
    set_src(C[13], s)

    s = get(22)
    s = rep(s, "Pseudo-labeling and noise augmentation for the embedding family were tested and rejected by CV (see write-up).",
            "Noise augmentation of the real rows for the embedding family was tested and rejected by CV (see write-up). "
            "No family uses test text for training.")
    set_src(C[22], s)

    for i, c in enumerate(C):
        src = "".join(c["source"])
        assert not re.search(r"(?i)pseudo", src) or i == 0 and "no pseudo-labeling" in src, (i, src[:200])
    return nb


def large_edits(nb):
    C = nb["cells"]
    s = "".join(C[3]["source"])
    s = rep(s, 'os.environ.get("SS_MODEL", "xlm-roberta-base")', 'os.environ.get("SS_MODEL", "xlm-roberta-large")')
    s = rep(s, 'os.environ.get("SS_LR", 3e-5)', 'os.environ.get("SS_LR", 1.5e-5)')
    s = rep(s, 'os.environ.get("SS_BATCH", 32)', 'os.environ.get("SS_BATCH", 16)')
    set_src(C[3], s)
    s = "".join(C[0]["source"])
    s = rep(s, "| `xlm-roberta-base` | Hugging Face Hub (FacebookAI/xlm-roberta-base) | MIT |",
            "| `xlm-roberta-large` | Hugging Face Hub (FacebookAI/xlm-roberta-large) | MIT |")
    s = rep(s, "* Model 2: `xlm-roberta-base` fine-tuned", "* Model 2: `xlm-roberta-large` fine-tuned (lr 1.5e-5, batch 16)")
    s = s.replace("to download xlm-roberta-base", "to download xlm-roberta-large")
    s = s.replace("Runtime is about 3–3.5 h on a T4: v2 cells about 2–2.5 h", "Runtime is about 4–4.5 h on a T4: v2 cells about 3–3.5 h")
    set_src(C[0], s)
    return nb


if __name__ == "__main__":
    src, out_base, out_large = sys.argv[1:4]
    nb = json.load(open(src, encoding="utf-8"))
    clear_outputs(nb)
    nb = base_edits(nb)
    json.dump(nb, open(out_base, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nb = large_edits(nb)
    json.dump(nb, open(out_large, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out_base, out_large)
