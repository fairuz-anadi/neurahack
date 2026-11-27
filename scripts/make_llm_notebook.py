"""Add the LLM family (OpenAI, zero-shot from the official task definitions) to the clean notebook.

    python scripts/make_llm_notebook.py notebooks/14_clean_ft.ipynb notebooks/15_llm.ipynb

Inserts two cells at the end: (1) labels every unique normalized test text with the OpenAI model (key from .env /
environment, results cached in /kaggle/working), and checks it on sample.csv; (2) writes submission.csv from the LLM
labels, falling back to the ensemble's submission for any row the LLM did not return.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = """# Part D: LLM family (OpenAI `gpt-4.1-mini`, zero-shot)
Each message is labeled by an OpenAI chat model from the official task definitions, labeling-priority order and
illustrative examples in the problem statement (prompt below). No labeled or test rows are put in the prompt, and there
is no fine-tuning. Calls are made from this notebook; the key is read from `.env` (`OPENAI_API_KEY`). temperature 0,
seed 42, 40 messages per request, JSON output. On the 86 rows of `sample.csv` it scores Task A macro F1 0.966.
The final `submission.csv` uses the LLM label, techniques and action; rows the LLM does not return keep the Part C
ensemble prediction. Confidence: 0.97 when LLM and ensemble agree, 0.60 when they disagree."""

FINAL = '''import pandas as pd, numpy as np
L = pd.read_csv(LLM_OUT + f"llm_test_{MODEL}.csv", keep_default_na=False).set_index("id").loc[test_ids].reset_index()
F = pd.read_csv("/kaggle/working/submission.csv", keep_default_na=False).set_index("id").loc[test_ids].reset_index()
F.to_csv("/kaggle/working/sub_ensemble.csv", index=False)
ok = L["llm_label"].astype(int).between(0, 5).values
fraud = L["llm_label"].astype(int).isin([1, 2, 3, 4]).values
tech = np.where(fraud, L["llm_tech"].replace("", "NONE"), "NONE")
tech = np.where(fraud & (tech == "NONE"), "T5", tech)
act = np.where(fraud, L["llm_action"].astype(int), 0)
sub = pd.DataFrame({"id": test_ids,
                    "label": np.where(ok, L["llm_label"].astype(int), F["label"].astype(int)),
                    "techniques": np.where(ok, tech, F["techniques"]),
                    "action": np.where(ok, act, F["action"].astype(int))})
agree = sub["label"].values == F["label"].astype(int).values
sub["confidence"] = np.where(~ok, F["confidence"].astype(float), np.where(agree, 0.97, 0.60)).round(4)
sub = sub[list(pd.read_csv(D + "sample_submission.csv", nrows=1).columns)]
assert len(sub) == len(test_ids) and sub["id"].is_unique and not sub.isna().any().any()
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"submission.csv = LLM ({ok.mean():.4f} of rows) | agree with ensemble {agree.mean():.4f}")
'''


def main():
    src, out = sys.argv[1:3]
    cell = (ROOT / "notebooks" / "llm_cell.py").read_text(encoding="utf-8")
    cell = cell.split("\ndef main():")[0]
    cell += '''
# ---- run in the notebook ----
os.environ.setdefault("OPENAI_API_KEY", "")
D = next(p for p in ["/kaggle/input/competitions/nsucec-neurahack-2026-preliminary/",
                     "/kaggle/input/nsucec-neurahack-2026-preliminary/"] if os.path.exists(p + "test.csv"))
LLM_OUT = "/kaggle/working/"
from sklearn.metrics import f1_score
_s = pd.read_csv(D + "sample.csv"); _txt = _s["text"].map(norm).tolist()
_done = await run(_txt, LLM_OUT + f"llm_cache_sample_{MODEL}.jsonl")
_pa = [_done.get(t, (0, "NONE", 0))[0] for t in _txt]
print("LLM on sample.csv: Task A macro F1", round(f1_score(_s["label"], _pa, average="macro"), 4))
_t = pd.read_csv(D + "test.csv"); test_ids = _t["id"].values
_tx = _t["text"].fillna("").map(norm)
_done = await run(_tx.tolist(), LLM_OUT + f"llm_cache_test_{MODEL}.jsonl")
_p = [_done.get(x) for x in _tx]
pd.DataFrame({"id": test_ids, "llm_label": [p[0] if p else -1 for p in _p], "llm_tech": [p[1] if p else "" for p in _p],
              "llm_action": [p[2] if p else -1 for p in _p]}).to_csv(LLM_OUT + f"llm_test_{MODEL}.csv", index=False)
print("missing:", sum(p is None for p in _p))
'''
    cell = cell.replace('MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")', 'MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")')
    cell = "!pip install -q openai python-dotenv\n" + cell
    nb = json.loads(Path(src).read_text(encoding="utf-8"))
    lines = lambda s: s.splitlines(keepends=True)
    nb["cells"] += [{"cell_type": "markdown", "metadata": {}, "source": lines(MD)},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(cell)},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(FINAL)}]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1); f.write("\n")
    print("wrote", out, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
