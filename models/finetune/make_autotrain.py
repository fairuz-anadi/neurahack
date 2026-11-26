"""Builds notebooks/ScamShield_AutoTrain.ipynb: ONE self-contained notebook for Kaggle "Save & Run All".

    python -m models.finetune.make_autotrain

The whole project code (same file set as build_code_zip: common, models, augmentation, configs, folds, scripts,
docs) is embedded as a base64 zip, so the only Kaggle input needed is the official competition dataset.
External data and models are downloaded from their declared public sources at run time (Internet ON).
The notebook also embeds a copy of itself (without that copy) so final_package.zip can include a notebook.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat as nbf

from models.finetune.build_code_zip import check, collect

NB_NAME = "ScamShield_AutoTrain.ipynb"

DECLARATION = """# ScamShield BD — AutoTrain (NSUCEC NeuraHack 2026) · Declaration

**How to run:** attach the official competition dataset · Accelerator **GPU T4 x2** · Internet **On** ·
**Save Version → Save & Run All (Commit)**. Everything else runs unattended (planned ≤ ~3.5 h); results appear in
the Output tab: `submission.csv`, `prediction_artifacts.zip`, `experiment_report.json`, `validation_report.html`,
`final_package.zip`, `progress.log`.

## 1. Libraries
Kaggle's preinstalled stack (versions printed in the setup cell): python 3, torch, transformers, tokenizers,
accelerate, numpy, pandas, scikit-learn, scipy, PyYAML. No extra installs. Optional vLLM is not used by default.

## 2. External resources (all downloaded by code from the original source)
| Resource | Use | Source | Licence |
|---|---|---|---|
| intfloat/multilingual-e5-small | student encoder (fine-tuned) | huggingface.co/intfloat/multilingual-e5-small | MIT |
| Qwen/Qwen2.5-7B-Instruct | local LLM: few-shot check, synthetic data, test pseudo-labels | huggingface.co/Qwen/Qwen2.5-7B-Instruct | Apache-2.0 |
| BTTC (Tahmid et al.) | external data, full/partial labels | data.mendeley.com/datasets/5wrm959d6f/1 | CC BY 4.0 |
| Labeled Bangla SMS Spam and Ham (Islam et al.) | external data | data.mendeley.com/datasets/hhd43jbv9p/1 | CC BY 4.0 |
| Financial scams detection dataset (Ahmed & Islam) | external data | data.mendeley.com/datasets/znsk27yk3h/2 | CC BY 4.0 |
| SMS Phishing Dataset (Mishra & Soni) | external data | data.mendeley.com/datasets/f45bkkt8pr/1 | CC BY 4.0 |
| Bangalabarta | **not used** (CC BY-NC-SA 4.0; excluded by default) | data.mendeley.com/datasets/jfkfbw3gzh/3 | CC BY-NC-SA 4.0 |

Hosted APIs: **none** (empty `.env`). Public code: none copied.
**Semi-supervised use of test text (declared):** if the LLM is clearly stronger than the student, Qwen labels a seeded
random sample of test rows by code (no human involvement, only model predictions); the e5 student trains on labels
with label-token confidence ≥ 0.8. **Synthetic data (declared):** Qwen generates labeled examples from the official
class definitions only (no sample.csv rows in prompts), greedy decoding, seeded spec grid, programmatic checks.

## 3. Method and validation
Multilingual e5-small with three heads (Task A softmax, Task B six sigmoids, Task C softmax); Task D = cross-fitted
temperature-scaled max probability. Candidates: A FT1 fallback (noise augmentation) · B + external data with
partial labels (ham→0, promo→5 full; scam→{1..4}, spam→{1..5} partial; Task B/C masked) · C + synthetic · D + both ·
E balanced (more weight on official rows) · F teacher→student (optional). All scored on the shared 5-fold manifest
of the 86 official rows; stress tests (noisy text, novel scenarios, unseen brands) on the top candidates; the best is
retrained on all data and predicts every test row. Official rules applied: priority 1→2→4→3→5→0 in the LLM guide;
classes 0 and 5 get techniques NONE and action 0. Scores in this notebook are local, not leaderboard scores.

## 4. Seeds and reproduction
Seed 42 (second seed 43 for the stability check). Run all cells top to bottom with the competition data attached,
GPU and Internet on. Generation uses greedy decoding; GPU kernels can add tiny run-to-run noise.
"""

CONFIG = r'''
# 1) CONFIGURATION — sensible defaults; nothing needs to change for a normal run.
AUTOTRAIN = dict(
    hours=3.5,                # planned total runtime; optional stages are skipped automatically when short on time
    final_reserve_min=30,     # always kept for final training, 500k-row inference and packaging
    seed=42, second_seed=43,
    test_max_len=256,         # 0% of test rows exceed 256 tokens (measured)
    run_external=True, include_nc=False, external_cap=300,
    run_synth=True, synth_n=900, synth_max_min=45,
    verify_llm=True,          # re-measures the Qwen few-shot score on the shared folds
    run_teacher=True, teacher_n=6000, teacher_max_min=30, pseudo_min_conf=0.8,   # runs only if Qwen is clearly stronger
    stress_top_k=2,
)
'''

SOURCES = r'''
# 2) PROJECT SOURCE — embedded so no private repository or extra dataset is needed.
import base64, gzip, io, os, zipfile
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('autotrain_work')
os.makedirs(WORK, exist_ok=True)
CODE = os.path.join(WORK, 'scamshield_code')
SRC_ZIP_B64 = "__SRC__"
NB_COPY_B64 = "__NBCOPY__"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_ZIP_B64))) as z:
    z.extractall(CODE)
if NB_COPY_B64:
    with open(os.path.join(WORK, '__NBNAME__'), 'wb') as f:
        f.write(gzip.decompress(base64.b64decode(NB_COPY_B64)))
print('code files:', sum(len(f) for _, _, f in os.walk(CODE)), '->', CODE)
'''

SETUP = r'''
# 3) ENVIRONMENT — finds the competition data, GPUs and library versions. Prints aggregates only.
import glob, sys, json, subprocess, platform
KAGGLE_INPUT = os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input')
def find_data_dir():
    for p in sorted(glob.glob(os.path.join(KAGGLE_INPUT, '**', 'sample_submission.csv'), recursive=True)):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, 'sample.csv')):
            return d
    raise FileNotFoundError('Attach the official competition dataset (sample.csv + sample_submission.csv not found).')
DATA = os.environ.get('SCAMSHIELD_DATA_DIR') or find_data_dir()
os.environ.update(SCAMSHIELD_DATA_DIR=DATA, SCAMSHIELD_ARTIFACTS_DIR=os.path.join(WORK, 'artifacts'),
                  SCAMSHIELD_OUTPUTS_DIR=WORK, PYTHONPATH=CODE, TOKENIZERS_PARALLELISM='false')
import torch, transformers, sklearn, pandas, numpy
print('python', platform.python_version(), '| torch', torch.__version__, '| transformers', transformers.__version__,
      '| sklearn', sklearn.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__)
print('GPUs:', [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1))
                for i in range(torch.cuda.device_count())] or 'NONE - enable Accelerator GPU T4 x2')
print('data dir:', DATA, '| test.csv present:', os.path.exists(os.path.join(DATA, 'test.csv')))
'''

RUN = r'''
# 4) RUN EVERYTHING — data, candidates, validation, selection, final training, 500k inference, submission, reports.
#    Runs in a child process (GPU memory is always released); progress is streamed and saved to progress.log.
cfg = dict(AUTOTRAIN, work=WORK, notebook_copy=os.path.join(WORK, '__NBNAME__'))
with open(os.path.join(WORK, 'autotrain_config.json'), 'w') as f:
    json.dump(cfg, f)
p = subprocess.Popen([sys.executable, '-u', '-m', 'models.finetune.autotrain', '--config-json',
                      os.path.join(WORK, 'autotrain_config.json')], cwd=CODE, env=os.environ.copy(),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    if 'Loading weights' not in line:
        print(line, end='')
RETURN_CODE = p.wait()
print('autotrain exit code', RETURN_CODE)
'''

SUMMARY = r'''
# 5) SUMMARY
rep_path = os.path.join(WORK, 'experiment_report.json')
rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
sel = rep.get('selected', {})
done = [s['stage'] for s in rep.get('stages', []) if s['status'] in ('ok', 'reused')]
skipped = [(s['stage'], s['status']) for s in rep.get('stages', []) if s['status'] not in ('ok', 'reused')]
fs, ft1 = sel.get('oof_scores') or {}, sel.get('ft1_fallback_scores') or {}
print('Best selected model :', sel.get('best_candidate'), '| members:', sel.get('members'))
print('Local OOF (86 rows) :', {k: fs.get(k) for k in ('F_A', 'F_B', 'F_C', 'Calibration_crossfit', 'screen_composite')})
print('FT1 fallback        :', {k: ft1.get(k) for k in ('F_A', 'F_B', 'F_C', 'Calibration_crossfit', 'screen_composite')})
print('LLM few-shot check  :', rep.get('llm_eval'))
print('Total minutes       :', rep.get('total_minutes'))
print('Completed stages    :', done)
print('Skipped / failed    :', skipped)
for f in ['submission.csv', 'prediction_artifacts.zip', 'experiment_report.json', 'validation_report.html',
          'final_package.zip', 'progress.log']:
    print(('OK   ' if os.path.exists(os.path.join(WORK, f)) else 'MISSING ') + os.path.join(WORK, f))
sc = fs.get('screen_composite')
print('Local target 0.80 reached (local proxy, NOT a Kaggle score):', None if sc is None else sc >= 0.80)
print('Official leaderboard evaluation is still required. Submit submission.csv manually after team agreement.')
if os.path.exists(os.path.join(WORK, 'AUTOTRAIN_FAILED.txt')):
    print('FAILURE DETAILS:', open(os.path.join(WORK, 'AUTOTRAIN_FAILED.txt')).read()[-2000:])
'''


def src_zip_b64() -> str:
    files = collect()
    problems = check(files)
    if problems:
        raise SystemExit("refusing to embed:\n - " + "\n - ".join(problems))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)
    return base64.b64encode(buf.getvalue()).decode()


def build(nb_copy_b64: str, src_b64: str):
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(DECLARATION.strip())]
    for c in (CONFIG, SOURCES, SETUP, RUN, SUMMARY):
        src = c.strip().replace("__SRC__", src_b64).replace("__NBCOPY__", nb_copy_b64).replace("__NBNAME__", NB_NAME)
        cells.append(nbf.v4.new_code_cell(src))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main():
    src = src_zip_b64()
    core = build("", src)
    core_bytes = nbf.writes(core).encode("utf-8")
    full = build(base64.b64encode(gzip.compress(core_bytes)).decode(), src)
    out = ROOT / "notebooks" / NB_NAME
    nbf.write(full, out)
    up = ROOT.parent / "kaggle_upload" / NB_NAME
    up.parent.mkdir(exist_ok=True)
    nbf.write(full, up)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) and {up}")


if __name__ == "__main__":
    main()
