"""Build scamshield-code.zip for upload as a PRIVATE Kaggle Dataset, plus a copy of 04_finetune.ipynb.

    python -m models.finetune.build_code_zip [--out E:\\neurahack\\kaggle_upload]

Contents: source modules, configs, shared fold manifests, requirements, README/docs. Never: competition
data, .env / credentials, checkpoints, prediction arrays, outputs, notebooks, caches, git-ignored files.
The build fails if a forbidden file slips in or a file looks like it contains a key.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INCLUDE_DIRS = ["common", "models", "augmentation", "configs", "folds", "scripts", "docs"]
INCLUDE_FILES = ["requirements.txt", "README.md"]
FORBIDDEN = ["*.npy", "*.npz", "*.pt", "*.bin", "*.ckpt", "*.safetensors", "*.onnx", "*.zip", "*.parquet",
             ".env", ".env.*", "*kaggle.json", "*.pem", "*token*.json", "*.pyc", "*.ipynb", "*.log",
             "sample.csv", "test.csv", "sample_submission.csv", "train.csv", "submission.csv"]
KEY_PATTERNS = [r"sk-[A-Za-z0-9_-]{20,}", r"AIza[0-9A-Za-z_-]{30,}", r"hf_[A-Za-z0-9]{30,}",
                r"ghp_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}", r"sk-ant-[A-Za-z0-9_-]{20,}"]
REQUIRED = ["common/data_loader.py", "common/folds.py", "common/prediction_io.py", "configs/base.yaml",
            "folds/folds.csv", "folds/folds_novel.csv", "folds/folds_domain.csv", "models/finetune/train.py",
            "models/finetune/multitask.py", "models/finetune/benchmark.py", "models/finetune/token_stats.py",
            "models/finetune/calibration.py", "augmentation/corrupt.py"]


def git_ignored(paths: list[str]) -> set[str]:
    if not paths:
        return set()
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(paths), capture_output=True,
                       text=True)
    return {line.strip().replace("\\", "/") for line in r.stdout.splitlines() if line.strip()}


def collect() -> list[str]:
    files = [f for f in INCLUDE_FILES if (ROOT / f).is_file()]
    for d in INCLUDE_DIRS:
        files += [p.relative_to(ROOT).as_posix() for p in (ROOT / d).rglob("*") if p.is_file()]
    files = [f for f in files if "__pycache__" not in f]
    ignored = git_ignored(files)
    return sorted(f for f in files if f not in ignored)


def check(files: list[str]) -> list[str]:
    problems = []
    for f in files:
        name = Path(f).name
        if any(fnmatch.fnmatch(name, pat) for pat in FORBIDDEN):
            problems.append(f"forbidden file type: {f}")
        if f.endswith(".csv") and not f.startswith("folds/"):
            problems.append(f"csv outside folds/: {f}")
        try:
            text = (ROOT / f).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"binary file: {f}")
            continue
        for pat in KEY_PATTERNS:
            if re.search(pat, text):
                problems.append(f"possible key in {f} ({pat})")
        if f.startswith("folds/") and f.endswith(".csv"):
            header = text.splitlines()[0]
            if header not in ("id,fold", "id,repeat,fold"):
                problems.append(f"unexpected fold manifest columns in {f}: {header}")
    problems += [f"missing required file: {r}" for r in REQUIRED if r not in files]
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "kaggle_upload"))
    args = ap.parse_args(argv)
    files = collect()
    problems = check(files)
    if problems:
        print("REFUSING TO BUILD:\n - " + "\n - ".join(problems))
        sys.exit(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "scamshield-code.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)               # files at the zip root -> /kaggle/input/<dataset>/common/...
    nb_src = ROOT / "notebooks" / "04_finetune.ipynb"
    shutil.copy(nb_src, out / nb_src.name)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    assert not check(names), "post-build check failed"
    print(f"built {zpath} ({zpath.stat().st_size // 1024} KB, {len(names)} files)")
    print(f"copied {out / nb_src.name}")
    top = sorted({n.split('/')[0] for n in names})
    print("top level:", top)


if __name__ == "__main__":
    main()
