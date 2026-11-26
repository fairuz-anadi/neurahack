"""Build the notebooks from the repo code.

    python scripts/build_notebook.py          -> notebooks/05_final_pipeline.ipynb (+ 01-04 dev notebooks)

05_final_pipeline.ipynb is SELF-CONTAINED: every project source file listed in EMBED is written into
the notebook as a %%writefile cell, so the organizers can run it in a clean Kaggle/Colab session with no
GitHub access and no precomputed prediction files. Folds are regenerated in the notebook and checked
against the committed fingerprint.

To add a model family to the final notebook: add its module to EMBED and a stage to FINAL_STAGES.
"""
import _bootstrap  # noqa: F401

import json
from pathlib import Path

import nbformat as nbf

from common import REPO_ROOT
from common.folds import fold_fingerprint

NB_DIR = REPO_ROOT / "notebooks"

EMBED = [
    "configs/base.yaml", "configs/labels.yaml",
    "common/__init__.py", "common/seed.py", "common/labels.py", "common/schema.py", "common/preprocessing.py",
    "common/data_loader.py", "common/folds.py", "common/metrics.py", "common/stress.py",
    "common/prediction_io.py", "common/submission.py",
    "augmentation/__init__.py", "augmentation/corrupt.py",
    "models/__init__.py", "models/classical/__init__.py", "models/classical/tfidf_lr.py",
]

DECLARATION = """# Declaration — ScamShield BD (NSUCEC NeuraHack 2026)

**Team:** <TEAM NAME> (<TEAM CODE>) · Anadi, Rabib, Samprity

## 1. Libraries and versions
No installation is required on Kaggle: the notebook uses the preinstalled scientific stack and prints the
exact versions it ran with in the next cell. Minimum versions: python>=3.10, numpy>=1.24, pandas>=2.0,
scikit-learn>=1.2, scipy>=1.10, PyYAML>=6.0. Developed and tested locally with python 3.14.6, numpy 2.5.1,
pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, PyYAML 6.0.3.

## 2. External resources
| Resource | Type | Source | License / provider |
|---|---|---|---|
| scikit-learn, numpy, pandas, scipy, PyYAML | libraries | PyPI | BSD-3 / BSD-3 / BSD-3 / BSD-3 / MIT |
| (add every pretrained model, dataset and public code here before submitting) | | | |

- **Hosted APIs:** none. The `.env` in the ZIP is empty.
- **External labelled datasets:** none.
- **Pseudo-labeling / use of unlabeled test text:** none in this version. (If added: done by code below,
  using only model predictions.)
- **Synthetic data:** none in this version.

## 3. Method and validation
- Text is kept raw and also normalized conservatively (Unicode NFC, invisible-character removal outside Bangla
  conjuncts, Cyrillic/Greek homoglyph folding inside Latin tokens, Latin lowercasing). URLs, numbers, emojis
  and Bangla marks are preserved.
- Model: character (2–5, word-boundary) + word (1–2) TF-IDF → logistic regression (balanced class weights),
  one multiclass model for Task A, one for Task C, six binary models for Task B. Task D confidence = max Task A
  probability. Rows predicted Legitimate (label 0) get techniques NONE.
- Validation: shared grouped stratified 5-fold CV (seed 42). Groups merge rows with the same template
  (digits/URLs masked) or char TF-IDF cosine ≥ 0.80, so near-duplicates never cross folds. Vectorizers are
  fit inside each training fold.
- Local proxies for the hidden subsets (not Kaggle scores): robustness = validation text corrupted with
  homoglyphs / zero-width chars / junk tokens / repeats; novel scenarios = whole scenario clusters held out;
  domain shift = all rows of a brand held out together.

## 4. Seeds and reproduction
- `SEED = 42` for python, numpy, scikit-learn (and torch if used).
- Run all cells top to bottom with the competition data attached. Runtime on Kaggle CPU: a few minutes.
- Output: `/kaggle/working/submission.csv`.
"""


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def embed_cells():
    cells = [md("## Project source (embedded so this notebook runs without any external code)")]
    cells.append(code("import os, sys\nSRC = os.path.abspath('scamshield_src')\nos.makedirs(SRC, exist_ok=True)\n"
                      "for d in ['configs','common','augmentation','models/classical']:\n"
                      "    os.makedirs(os.path.join(SRC, d), exist_ok=True)\nprint(SRC)"))
    for rel in EMBED:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8") or "# package marker\n"
        cells.append(code(f"%%writefile scamshield_src/{rel}\n{text}"))
    return cells


SETUP = r'''
import os, sys, json, platform
SRC = os.path.abspath('scamshield_src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('.')
os.environ['SCAMSHIELD_ARTIFACTS_DIR'] = os.path.join(WORK, 'artifacts')
os.environ['SCAMSHIELD_OUTPUTS_DIR'] = WORK
# Data dir: auto-detected under /kaggle/input (folder containing sample_submission.csv),
# or set SCAMSHIELD_DATA_DIR yourself when running elsewhere.
import numpy, pandas, sklearn, scipy, yaml
print('python', platform.python_version(), '| numpy', numpy.__version__, '| pandas', pandas.__version__,
      '| scikit-learn', sklearn.__version__, '| scipy', scipy.__version__, '| PyYAML', yaml.__version__)
from common.seed import set_seed, device_info
SEED = set_seed(42)
print('seed', SEED, device_info())
'''

DATA = r'''
from common.data_loader import load_config, load_train, load_test, load_sample_submission, resolve_data_dir
cfg = load_config(os.path.join(SRC, 'configs', 'base.yaml'))
print('data dir:', resolve_data_dir(cfg))
train = load_train(cfg)
test = load_test(cfg)
sample = load_sample_submission(cfg)
print('train', train.shape, 'fingerprint', train.attrs['fingerprint'])
print('test', test.shape, '| text column:', test.attrs['text_col'], '| extra columns:', test.attrs['extra_columns'])
assert set(test['id']) == set(sample['id']), 'test ids differ from sample_submission'
# Aggregate statistics only — no individual test rows are displayed.
print('train Task A counts:', train['label'].value_counts().sort_index().to_dict())
print('test script share:', test['script'].value_counts(normalize=True).round(4).to_dict())
print('test conversation share:', round(float(test['is_conversation'].mean()), 4))
'''

FOLDS = r'''
from pathlib import Path
from common import folds as F
os.environ['SCAMSHIELD_FOLDS_DIR'] = os.path.join(WORK, 'folds')
meta = F.make_all_folds(train, cfg)
EXPECTED_FOLD_FINGERPRINT = '__FOLD_FP__'
print('fold fingerprint', meta['main']['fingerprint'], '| expected', EXPECTED_FOLD_FINGERPRINT)
assert meta['main']['fingerprint'] == EXPECTED_FOLD_FINGERPRINT, 'fold manifest differs from the one used in development'
'''

BASELINE = r'''
from common.stress import run_stress
from common.metrics import composite_proxy
from models.classical.tfidf_lr import TfidfLR, TfidfLRConfig, make_fit_predict
mcfg = TfidfLRConfig(seed=SEED)
report, oof = run_stress(make_fit_predict(mcfg), train, seed=SEED, repeats=True)
m = report['main']
comp = composite_proxy({'F_A': m['F_A'], 'F_B': m['F_B'], 'F_C': m['F_C'], 'Calibration': m['Calibration'],
                        'F_robust': report['F_robust_proxy'], 'F_novel': report['F_novel_proxy'],
                        'F_domain': report['F_domain_proxy']})
print('OOF (grouped CV):', {k: m[k] for k in ('F_A', 'F_B', 'F_C', 'Brier', 'Calibration')})
print('Local proxies (NOT Kaggle scores):', {k: report[k] for k in ('F_robust_proxy', 'F_novel_proxy', 'F_domain_proxy')})
print('Repeated grouped CV F_A:', report['repeated_F_A'])
print('LOCAL composite proxy:', comp['composite_proxy'])
'''

PREDICT = r'''
from common.submission import build_submission, write_submission, validate_submission
model = TfidfLR(mcfg).fit(train)          # final model: all labeled rows
pred = model.predict(test)
sc = cfg['submission']
sub = build_submission(test['id'].to_numpy(), pred['A'], pred['B'], pred['C'],
                       technique_threshold=sc['technique_threshold'], nonfraud_classes=sc['nonfraud_classes'],
                       fraud_requires_technique=sc['fraud_requires_technique'])
sub = write_submission(sub, os.path.join(WORK, 'submission.csv'), sample)
print('submission.csv written:', sub.shape)
print('predicted label share:', sub['label'].value_counts(normalize=True).sort_index().round(4).to_dict())
print('validation problems:', validate_submission(sub, sample, sc['nonfraud_classes']))
'''

ANALYSIS = r'''
import pandas as pd
from common.labels import TASK_A_NAMES
print('Per-class Task A (OOF):')
print(pd.DataFrame(m['per_class_A']).T)
print('Confusion matrix Task A (rows = true, cols = predicted):')
print(pd.DataFrame(m['confusion_A'], index=list(TASK_A_NAMES.values()), columns=range(6)))
print('Per-technique Task B (OOF):')
print(pd.DataFrame(m['per_technique_B']).T)
print('Slices (script / conversation):')
print(pd.DataFrame(report['slices_A']).T)
'''

GENERALIZATION_MD = """## Generalization analysis
- **Languages/scripts:** see the slice table. The 86 labeled rows contain only 6 Bangla-script rows; Latin-script
  rows mix English and Banglish (script detection cannot separate them).
- **Unseen brands:** `F_domain_proxy` holds out every row of a brand together.
- **Noise:** `F_robust_proxy` corrupts validation text (homoglyphs, zero-width characters, junk tokens, repeats).
- **Class confusions:** see the confusion matrix (most errors are scams predicted as Legitimate).
- These are local proxies; the hidden evaluation subsets are not available to us.
"""


def final_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [md(DECLARATION)] + embed_cells() + [
        code(SETUP),
        md("## Data (aggregate checks only)"), code(DATA),
        md("## Shared grouped folds (regenerated, fingerprint-checked)"), code(FOLDS.replace("__FOLD_FP__", fold_fingerprint("folds"))),
        md("## Model: TF-IDF + logistic regression — grouped CV and local stress tests"), code(BASELINE),
        md("## Final fit and test prediction"), code(PREDICT),
        md("## Error analysis"), code(ANALYSIS), md(GENERALIZATION_MD),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


DEV_HEADER = r'''
# Dev notebook: imports the shared code from the repo.
# Local: run from the repo (notebooks/ folder). Kaggle: attach your private code dataset and set CODE_DIR.
import os, sys
CODE_DIR = os.environ.get('SCAMSHIELD_CODE_DIR') or ('/kaggle/input/scamshield-code' if os.path.isdir('/kaggle/input/scamshield-code') else os.path.abspath('..'))
sys.path.insert(0, CODE_DIR)
from common.data_loader import load_config, load_train, load_test, get_targets, test_available, output_dirs
from common.folds import load_folds, fold_fingerprint
from common.seed import set_seed
cfg = load_config(os.path.join(CODE_DIR, 'configs', 'base.yaml'))
SEED = set_seed(cfg['seed'])
train = load_train(cfg)
fold = load_folds(train)            # shared manifest: folds/folds.csv
print(train.shape, 'folds', sorted(set(fold)), 'test available:', test_available(cfg))
'''


def dev_notebooks():
    out = {}
    nb = nbf.v4.new_notebook()
    nb.cells = [md("# 01 — Data analysis\nRuns `scripts/inspect_data.py` logic. Test data: aggregate statistics only."),
                code(DEV_HEADER),
                code("import subprocess, sys\nprint(subprocess.run([sys.executable, os.path.join(CODE_DIR, 'scripts', 'inspect_data.py')], capture_output=True, text=True).stdout)\n"
                     "print(open(os.path.join(CODE_DIR, 'outputs', 'eda_report.md'), encoding='utf-8').read())")]
    out["01_data_analysis.ipynb"] = nb

    nb = nbf.v4.new_notebook()
    nb.cells = [md("# 02 — Classical baseline (Anadi)"), code(DEV_HEADER),
                code(BASELINE.replace("seed=SEED, repeats=True", "seed=SEED, repeats=True")),
                code(ANALYSIS)]
    out["02_classical_baseline.ipynb"] = nb

    emb = r'''
# Rabib: implement fit_predict with frozen multilingual embeddings + a light classifier.
# Cache embeddings ONCE per model for train and test (test is 500k rows: batch on GPU, fp16, save .npy).
import numpy as np
from common.labels import N_A, N_B, N_C
from common.stress import run_stress
from common.prediction_io import save_predictions

def fit_predict(train_df, valid_df):
    # X_tr = EMB[train_df.row_idx.values]; X_va = EMB[valid_df.row_idx.values]  (precomputed, row-aligned)
    # fit heads for A (multiclass), B (6 binary / one-vs-rest sigmoid), C (multiclass)
    raise NotImplementedError

# report, oof = run_stress(fit_predict, train, seed=SEED)
# save_predictions('emb_e5_lr', output_dirs(cfg)[0], oof, train['id'].to_numpy(), meta, test_pred, test_ids)
'''
    nb = nbf.v4.new_notebook()
    nb.cells = [md("# 03 — Embeddings (Rabib)\nContract: see docs/TEAM_CONTRACT.md §5."), code(DEV_HEADER), code(emb)]
    out["03_embeddings.ipynb"] = nb

    ft = r'''
# Samprity: transformer fine-tuning with the shared folds. Per fold: train on fold != k ONLY
# (synthetic / augmented rows generated from that training fold only), predict fold == k.
import numpy as np
from common.stress import run_stress
from common.prediction_io import save_predictions
from augmentation.corrupt import corrupt   # adversarial augmentation for TRAINING rows

def fit_predict(train_df, valid_df):
    # 1) optional: augment train_df rows (corrupt(text, seed)), never valid_df
    # 2) fine-tune xlm-roberta / banglabert with heads for A (softmax), B (6 sigmoids), C (softmax)
    # 3) return {"A": (n,6) softmax, "B": (n,6) sigmoid, "C": (n,6) softmax} for valid_df
    raise NotImplementedError

# report, oof = run_stress(fit_predict, train, seed=SEED, repeats=False)   # repeats are costly on GPU
'''
    nb = nbf.v4.new_notebook()
    nb.cells = [md("# 04 — Fine-tuning (Samprity)\nContract: see docs/TEAM_CONTRACT.md §5."), code(DEV_HEADER), code(ft)]
    out["04_finetune.ipynb"] = nb
    for nb in out.values():
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return out


def main():
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(final_notebook(), NB_DIR / "05_final_pipeline.ipynb")
    for name, nb in dev_notebooks().items():
        nbf.write(nb, NB_DIR / name)
    print("wrote", sorted(p.name for p in NB_DIR.glob("*.ipynb")))


if __name__ == "__main__":
    main()
