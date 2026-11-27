"""Final notebook = the clean Kaggle notebook (clean.ipynb, produced the 0.89153 ensemble) + LLM labels from OpenAI
gpt-4.1-mini on the rows where the ensemble is unsure (max Task A probability < 0.6). Reproduces final_clean_llm.csv.

    python scripts/make_final_notebook.py D:/NSU/kaggle_push_clean/clean.ipynb notebooks/16_final.ipynb
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_llm_notebook as M  # noqa: E402

MD = """# Part D: LLM labels for uncertain rows (OpenAI `gpt-4.1-mini`, zero-shot)
Every unique test message is labeled by an OpenAI chat model from the official task definitions, labeling-priority
order and the illustrative examples in the problem statement (prompt below). No labeled or test row is in the prompt and
there is no fine-tuning. Calls are made from this notebook; the key is read from `.env` (`OPENAI_API_KEY`).
temperature 0, seed 42, 40 messages per request, JSON output; results are cached so an interrupted run resumes.
On the 86 `sample.csv` rows it scores Task A macro F1 0.966 (0.9645 on the 78 rows not used as prompt examples).

**Final rule.** Keep the Part C ensemble (TF-IDF + XLM-R blend) everywhere, except where its highest Task A
probability is below 0.6 (5.4% of test rows): there the LLM label is used. Techniques and action are then decoded from
the ensemble probabilities for that label (same post-rules as Part C). Confidence: the Part C confidence where the label
is unchanged, otherwise `clip(1 - confidence, 0.05, 0.6)`."""

FINAL = '''import numpy as np, pandas as pd
TC = [f"T{i}" for i in range(1, 7)]
P = [np.load(f"/kaggle/working/test_v2ens_{t}.npy").astype(np.float32) for t in "ABC"]
PA = P[0] / P[0].sum(1, keepdims=True)
base = pd.read_csv("/kaggle/working/submission.csv", keep_default_na=False)
base.to_csv("/kaggle/working/sub_ensemble.csv", index=False)
la = pd.read_csv(LLM_OUT + f"llm_test_{MODEL}.csv").set_index("id").loc[test_ids].llm_label.values
ma, mp = PA.argmax(1), PA.max(1)
a = np.where((mp < 0.6) & (la >= 0), la, ma)
fr = np.isin(a, [1, 2, 3, 4]); B = (P[1] >= 0.5).astype(int); B[~fr] = 0
idx = np.where(fr & (B.sum(1) == 0))[0]; B[idx, P[1][idx].argmax(1)] = 1
C = P[2].copy(); C[fr, 0] = -1; act = C.argmax(1); act[~fr] = 0
conf = np.where(a == ma, base["confidence"].values, np.clip(1 - base["confidence"].values, 0.05, 0.6))
sub = pd.DataFrame({"id": test_ids, "label": a, "techniques": ["|".join(TC[j] for j in np.flatnonzero(r)) or "NONE" for r in B],
                    "action": act, "confidence": conf.round(4)})[list(pd.read_csv(D + "sample_submission.csv", nrows=1).columns)]
assert len(sub) == len(test_ids) and sub["id"].is_unique and not sub.isna().any().any()
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"submission.csv: LLM label on {(mp < 0.6).mean():.4f} of rows, labels changed {(a != ma).mean():.4f}")
'''


def main():
    src, out = sys.argv[1:3]
    cell = (Path(M.ROOT) / "notebooks" / "llm_cell.py").read_text(encoding="utf-8").split("\ndef main():")[0]
    tail_start = M.main.__code__  # reuse the notebook-run block text from make_llm_notebook
    txt = Path(M.__file__).read_text(encoding="utf-8")
    block = txt.split("cell += '''", 1)[1].split("'''", 1)[0]
    cell = "!pip install -q openai python-dotenv\n" + cell + block
    cell = cell.replace('MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")', 'MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")')
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
