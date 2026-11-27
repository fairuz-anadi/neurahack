"""Add Samprity's FT family (notebooks/ft_integration_cell.py) to a final-team notebook, before Part C.

    python scripts/make_ft_notebook.py IN.ipynb OUT.ipynb [--repeats 1]

The cell is wrapped in try/except: if the FT family fails on Kaggle, its partial files are removed and the final
cell runs with the other families, so the notebook still writes submission.csv. The final cell loads `ft` by name.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MD = """## B4. FT family (Samprity): fine-tuned `multilingual-e5-small`
Multi-task fine-tune (label, action, 6 techniques) on `sample.csv` + embedded synthetic rows (written offline by
Qwen2.5-7B-Instruct from the official class definitions only) + embedded CC BY 4.0 external rows (4 Mendeley datasets,
see Declaration). Uses the team folds; `test.csv` is read only after training, for inference. No pseudo-labels.
Writes `oof_ft_*.npy` / `test_ft_*.npy`, which Part C picks up. If this family fails, the cell removes its partial
files and Part C runs without it."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--repeats", type=int, default=1, help="FT_REPEATS default (1 = 5 fits, like xlmr)")
    a = ap.parse_args()

    cell = (ROOT / "notebooks" / "ft_integration_cell.py").read_text(encoding="utf-8")
    old = "FT_REPEATS    = int(os.environ.get('FT_REPEATS', 5))"
    assert cell.count(old) == 1
    cell = cell.replace(old, f"FT_REPEATS    = int(os.environ.get('FT_REPEATS', {a.repeats}))")
    body = "\n".join(("    " + l) if l.strip() else l for l in cell.rstrip("\n").split("\n"))
    code = ("import os as _os, glob as _glob, traceback as _tb\n"
            "try:\n" + body + "\n"
            "except Exception:\n"
            "    _tb.print_exc()\n"
            "    for _f in _glob.glob('/kaggle/working/*_ft_*.npy'):\n"
            "        _os.remove(_f)\n"
            "    print('FT family failed: partial files removed; Part C runs without ft')\n"
            "finally:\n"
            "    import gc as _gc, torch as _torch\n"
            "    _gc.collect(); _torch.cuda.empty_cache()\n")
    compile(code, "ft_cell", "exec")

    nb = json.loads(Path(a.src).read_text(encoding="utf-8"))
    C = nb["cells"]
    i = next(k for k, c in enumerate(C) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("# Part C"))
    lines = lambda s: s.splitlines(keepends=True)
    C[i:i] = [{"cell_type": "markdown", "metadata": {}, "source": lines(MD)},
              {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(code)}]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1); f.write("\n")
    print(f"wrote {a.out}: FT cell at {i + 1} (FT_REPEATS default {a.repeats}), {len(C)} cells")


if __name__ == "__main__":
    main()
