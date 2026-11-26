"""Build the final package NeuraHack_Preli_<Team_Name>.zip (guidance email, 26 Nov) with EXACTLY
four items: submission.csv, notebook.ipynb, README.md and .env (empty if no API is used).

    python scripts/build_zip.py --team Team_Hampton --csv outputs/submission.csv \
        --notebook notebooks/07_final_team.ipynb --readme submission/README.md [--env path/to/.env]

Refuses to build if the submission is invalid, the notebook contains something that looks like an
API key, the README is missing or still has placeholders, or the .env is missing. Never commit the
resulting zip or a filled .env.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from common.data_loader import load_config, load_sample_submission, output_dirs
from common.submission import validate_submission

KEY_PATTERNS = [r"sk-[A-Za-z0-9_-]{20,}", r"AIza[0-9A-Za-z_-]{30,}", r"hf_[A-Za-z0-9]{30,}",
                r"ghp_[A-Za-z0-9]{30,}", r"sk-ant-[A-Za-z0-9_-]{20,}", r"\"key\"\s*:\s*\"[0-9a-f]{32}\""]
README_PLACEHOLDERS = r"<Team Name>|\[from run\]"
EXPECTED = [".env", "README.md", "notebook.ipynb", "submission.csv"]


def scan_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        for pat in KEY_PATTERNS:
            if re.search(pat, src):
                problems.append(f"cell {i}: looks like a hardcoded key ({pat})")
        if re.search(r"read_csv\([^)]*(label|pred)[^)]*\.csv", src) and "submission" not in src:
            problems.append(f"cell {i}: reads a label/prediction CSV - check it is not a handmade label file")
    if not nb["cells"] or nb["cells"][0]["cell_type"] != "markdown" or "Declaration" not in "".join(nb["cells"][0]["source"]):
        problems.append("first cell must be the markdown Declaration section")
    return problems


def scan_readme(path: Path) -> list[str]:
    if not path.is_file():
        return [f"README not found: {path}"]
    if re.search(README_PLACEHOLDERS, path.read_text(encoding="utf-8")):
        return ["README still has placeholders (<Team Name> or [from run])"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="registered team name, e.g. Team_Hampton")
    ap.add_argument("--csv", default="outputs/submission.csv")
    ap.add_argument("--notebook", default="notebooks/07_final_team.ipynb")
    ap.add_argument("--readme", default="submission/README.md")
    ap.add_argument("--env", default=None, help="path to the .env to include; default: an empty .env")
    args = ap.parse_args()

    cfg = load_config()
    _, out_dir = output_dirs(cfg)
    csv, nb, readme = Path(args.csv), Path(args.notebook), Path(args.readme)
    problems = validate_submission(pd.read_csv(csv, keep_default_na=False), load_sample_submission(cfg),
                                   cfg["submission"]["nonfraud_classes"])
    problems += scan_notebook(nb)
    problems += scan_readme(readme)
    if problems:
        print("REFUSING TO BUILD:\n - " + "\n - ".join(problems))
        sys.exit(1)

    env_bytes = Path(args.env).read_bytes() if args.env else b""
    team = re.sub(r"[^A-Za-z0-9_]+", "_", args.team.strip()).strip("_")
    zpath = out_dir / f"NeuraHack_Preli_{team}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, "submission.csv")
        z.write(nb, "notebook.ipynb")
        z.write(readme, "README.md")
        z.writestr(".env", env_bytes)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    assert sorted(names) == EXPECTED, names
    print(f"built {zpath} with {names} (.env {'empty' if not env_bytes else f'{len(env_bytes)} bytes'})")


if __name__ == "__main__":
    main()
