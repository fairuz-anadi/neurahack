"""Writes notebooks/04_finetune.ipynb (Kaggle GPU runner for the fine-tuning family).

    python -m models.finetune.make_notebook

scripts/build_notebook.py skips 04_finetune.ipynb (OWNED_ELSEWHERE), so this file is the only writer.
Every step is its own cell; the CONFIG cell decides what runs. Full test inference is OFF by default.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

INTRO = """# 04 — Multi-task transformer fine-tuning (Samprity) — Kaggle GPU runner

**Attach before running:** (1) the competition dataset, (2) the private dataset made from `scamshield-code.zip`.
**Settings:** Accelerator = GPU T4 (or P100) · Internet = On (first download of the model from Hugging Face).

Cells: 1 setup (finds code + data) · 2 CONFIG · 3 GPU benchmark · 4 test token-length stats (aggregate) ·
5 FT0 validation · 6 FT1 validation · 7 full test inference (**opt-in**) · 8 contract check + download zip ·
9 submission.csv · 10 LLM few-shot eval · 11 synthetic data · 12 FT2/FT3 screening · 13 LLM teacher ·
14 student screening · 15 external data + FT3/FT4.
Each cell after 2 can be run on its own (cell 7 needs cell 5 or 6 of the same `TEST_RUN_NAME` first, in the
same session, or the artifacts it saved).

- Backbone `intfloat/multilingual-e5-small` (MIT). Shared encoder + heads A/B/C; Task D = cross-fitted temperature scaling.
- Scores printed here are LOCAL OOF / proxy numbers on 86 labeled rows, not Kaggle scores.
- Test rows are only read by code; only aggregate statistics are printed.
"""

SETUP = r'''
# 1) SETUP — locate the code dataset and the competition data under /kaggle/input.
import os, sys, glob, shutil, subprocess, json, time
KAGGLE_INPUT = os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input')
WORK = os.environ.get('KAGGLE_WORK_DIR', '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('.'))

def find_code_dir():
    hits = sorted(glob.glob(os.path.join(KAGGLE_INPUT, '**', 'common', 'data_loader.py'), recursive=True))
    hits = [os.path.dirname(os.path.dirname(h)) for h in hits
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(h)), 'models', 'finetune', 'train.py'))]
    if not hits:  # Kaggle sometimes keeps an uploaded zip as a single file: unpack it into WORK
        zips = sorted(glob.glob(os.path.join(KAGGLE_INPUT, '**', 'scamshield-code*.zip'), recursive=True))
        if zips:
            import zipfile
            dst = os.path.join(WORK, 'scamshield_code_unzipped')
            shutil.rmtree(dst, ignore_errors=True)
            zipfile.ZipFile(zips[0]).extractall(dst)
            print('unzipped', zips[0])
            return dst
    if not hits:
        raise FileNotFoundError(f'Code dataset not found under {KAGGLE_INPUT}. Add the private dataset made from '
                                'scamshield-code.zip (Add Input -> Your Datasets).')
    return hits[0]

def find_data_dir(exclude):
    cands = []
    for p in sorted(glob.glob(os.path.join(KAGGLE_INPUT, '**', 'sample_submission.csv'), recursive=True)):
        d = os.path.dirname(p)
        if not os.path.abspath(d).startswith(os.path.abspath(exclude)) and os.path.exists(os.path.join(d, 'sample.csv')):
            cands.append(d)
    if not cands:
        raise FileNotFoundError(f'Competition data (sample.csv + sample_submission.csv) not found under {KAGGLE_INPUT}. '
                                'Add the competition as an input.')
    return cands[0]

SRC = find_code_dir()
CODE = os.path.join(WORK, 'scamshield_code')            # writable copy (Kaggle inputs are read-only)
shutil.rmtree(CODE, ignore_errors=True)
shutil.copytree(SRC, CODE, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
DATA = find_data_dir(SRC)
os.environ.update(SCAMSHIELD_DATA_DIR=DATA, SCAMSHIELD_ARTIFACTS_DIR=os.path.join(WORK, 'artifacts'),
                  SCAMSHIELD_OUTPUTS_DIR=WORK, PYTHONPATH=CODE, TOKENIZERS_PARALLELISM='false')
if CODE not in sys.path:
    sys.path.insert(0, CODE)

def run(*args):
    """Run a repo module in a fresh process (frees GPU memory when it ends) and stream its output."""
    p = subprocess.Popen([sys.executable, '-u', '-m', *map(str, args)], cwd=CODE, env=os.environ.copy(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if 'Loading weights' not in line:
            print(line, end='')
    if p.wait() != 0:
        raise RuntimeError(f'{args[0]} failed with exit code {p.returncode}')

import torch, transformers
from common.data_loader import load_config, test_available
from common.folds import fold_fingerprint
print('code dataset :', SRC)
print('data dir     :', DATA, '| test.csv present:', test_available(load_config()))
print('fold fingerprint:', fold_fingerprint(), '(expected c357ab8ee361835a)')
print('torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(no GPU: enable Accelerator -> GPU)')
'''

CONFIG = r'''
# 2) CONFIG — edit here, then run the cells you need.
MODEL        = 'intfloat/multilingual-e5-small'
MAX_LEN      = 128      # raise to 256 if cell 4 shows many test rows above 128 tokens
TRAIN_BATCH  = 16
EVAL_BATCH   = 256      # inference batch; halves itself automatically on CUDA out-of-memory
EPOCHS       = 15
FP16         = True     # mixed precision on GPU

RUN_FT0      = True     # cell 5: baseline validation (shared folds + proxies), no test inference
RUN_FT1      = True     # cell 6: + in-fold adversarial copies, no test inference
PREDICT_TEST = False    # cell 7: OPT-IN full inference on every test row (~4 min on T4)
TEST_RUN_NAME = 'ft_e5s_corrupt'   # which validated run to use for test inference / submission

# Phase 2-4 (LLM + synthetic data). Qwen2.5-7B-Instruct (Apache-2.0) is ~15 GB: first download takes minutes.
RUN_LLM_EVAL = False    # cell 10: few-shot LLM on the shared folds (86 rows) -> compare with FT0/FT1
RUN_SYNTH    = False    # cell 11: generate synthetic labeled data from the official definitions only
SYNTH_N      = 900      # target records before quality filtering
SYNTH_PATH   = os.path.join(WORK, 'synth_v1.csv')
REAL_REPEAT  = 3        # real rows repeated per fit so synthetic data does not swamp them
RUN_FT2      = False    # cell 12: real + synthetic (main folds only)
RUN_FT3      = False    # cell 12: real + synthetic + noisy copies (main folds only)
# Teacher -> student (LLM pseudo-labels a seeded sample of test rows by code; e5 learns from them).
INSTALL_VLLM = False    # cell 13: pip install vllm first (much faster teacher); falls back to HF if absent
RUN_TEACHER  = False    # cell 13
TEACHER_BENCH_N = 256   # cell 13 first labels this many rows to measure throughput, then TEACHER_N
TEACHER_N    = 20000    # test rows to pseudo-label (set from the benchmark: rows/s x minutes you can afford)
PSEUDO_MIN_CONF = 0.8   # quality filter: keep teacher labels whose label-token probability >= this
TEACHER_MINUTES = 60    # hard time limit; whatever is finished is kept
TEACHER_PATH = os.path.join(WORK, 'teacher_labels.csv')
RUN_STUDENT  = False    # cell 14: e5 trained on real (x3) + pseudo-labels, main folds only
STUDENT_ARGS = ['--pseudo', TEACHER_PATH, '--pseudo-min-conf', PSEUDO_MIN_CONF, '--real-repeat', REAL_REPEAT,
                '--epochs', 3, '--batch-size', 32]
# Public external data (5 Mendeley datasets, licences in docs/RESOURCE_DECLARATION.md), full + partial labels.
RUN_EXTERNAL = False    # cell 15: download + build external_v1.csv, then FT3 (real+external), FT4 (real+synthetic+external)
EXTERNAL_PATH = os.path.join(WORK, 'external_v1.csv')
EXTERNAL_CAP = 300      # max rows per (source, label); CC BY 4.0 sources only (Bangalabarta excluded)
EXTERNAL_MAX = 1200     # external rows sampled per fit, so partial labels never outnumber the labeled data
EXT_TRAIN = ['--epochs', 4, '--batch-size', 32]      # bigger data -> fewer epochs, same order of steps
EXTRA = {'ft_e5s_base': [], 'ft_e5s_corrupt': ['--augment', 'corrupt'], 'ft_e5s_student': STUDENT_ARGS,
         'ft_e5s_ext': ['--external', EXTERNAL_PATH, '--external-max', EXTERNAL_MAX, '--real-repeat', REAL_REPEAT, *EXT_TRAIN],
         'ft_e5s_syn_ext': ['--external', EXTERNAL_PATH, '--external-max', EXTERNAL_MAX, '--synthetic', SYNTH_PATH,
                            '--real-repeat', REAL_REPEAT, *EXT_TRAIN],
         'ft_e5s_syn': ['--synthetic', SYNTH_PATH, '--real-repeat', REAL_REPEAT],
         'ft_e5s_syn_corrupt': ['--synthetic', SYNTH_PATH, '--real-repeat', REAL_REPEAT, '--augment', 'corrupt']}

COMMON = ['--model', MODEL, '--max-len', MAX_LEN, '--batch-size', TRAIN_BATCH, '--eval-batch', EVAL_BATCH,
          '--epochs', EPOCHS] + ([] if FP16 else ['--no-fp16'])
'''

BENCH = r'''
# 3) GPU BENCHMARK — stand-in rows built from TRAINING texts (10% stitched into 3-12 turn conversations).
#    Prints rows/sec and the estimated minutes for 500,000 rows. No test data is used.
run('models.finetune.benchmark', '--rows', 4000, '--batch', 128, 256, 512, '--max-len', 128, 256,
    '--conv-share', 0.1, '--model', MODEL)
'''

TOKENS = r'''
# 4) TEST TOKEN-LENGTH / TRUNCATION STATS — aggregates only (quantiles, shares over 64..512 tokens,
#    by number of turns and by script). Falls back to training rows if test.csv is not attached.
run('models.finetune.token_stats', '--model', MODEL)
'''

FT0 = r'''
# 5) FT0 VALIDATION — baseline on the shared folds + robust/novel/domain proxies. No test inference.
if RUN_FT0:
    run('models.finetune.train', '--name', 'ft_e5s_base', *COMMON)
else:
    print('RUN_FT0 is False - skipped')
'''

FT1 = r'''
# 6) FT1 VALIDATION — same as FT0 + 2 corrupted copies of every TRAINING-fold row. No test inference.
if RUN_FT1:
    run('models.finetune.train', '--name', 'ft_e5s_corrupt', '--augment', 'corrupt', *COMMON)
else:
    print('RUN_FT1 is False - skipped')
'''

PREDICT = r'''
# 7) FULL TEST INFERENCE (OPT-IN) — fits TEST_RUN_NAME's config on all 86 labeled rows, predicts every test
#    row, re-saves artifacts/<name>/ with test_{A,B,C}, test_ids, test_conf. Reuses that run's OOF + temperature.
if PREDICT_TEST:
    run('models.finetune.train', '--name', TEST_RUN_NAME, '--no-cv', '--predict-test', *COMMON, *EXTRA[TEST_RUN_NAME])
else:
    print('PREDICT_TEST is False - skipped (set it to True in cell 2 when a run is worth submitting)')
'''

CHECK = r'''
# 8) CONTRACT CHECK + SUMMARY + DOWNLOAD — verifies every saved run against the shared prediction contract.
import numpy as np, pandas as pd
from pathlib import Path
from common.data_loader import load_config, load_train, load_sample_submission
from common.prediction_io import load_predictions, check_metadata_compatible
cfg = load_config(); train = load_train(cfg); sample = load_sample_submission(cfg)
art = Path(os.environ['SCAMSHIELD_ARTIFACTS_DIR'])
metas = []
for d in sorted(p for p in art.glob('*') if (p / 'metadata.json').exists()):
    meta = json.loads((d / 'metadata.json').read_text()); metas.append(meta)
    load_predictions(d, 'oof', expected_ids=train['id'].to_numpy())          # shapes, ranges, ids
    line = f"{d.name}: OOF ok | F_A {meta['metrics'].get('F_A')} | calib {meta['metrics'].get('calibrated_Calibration')}"
    if meta.get('has_test'):
        t = load_predictions(d, 'test', expected_ids=sample['id'].to_numpy())  # every official id exactly once
        line += f" | TEST ok ({len(t['ids'])} rows, A counts {np.bincount(t['A'].argmax(1), minlength=6).tolist()})"
    print(line)
if metas:
    check_metadata_compatible(metas)
    print('dataset/fold fingerprints compatible across runs')
for f in sorted(Path(WORK).glob('eval_ft_*.json')):
    r = json.loads(f.read_text())
    if 'main' in r:
        print(f.stem, {k: r['main'][k] for k in ('F_A', 'F_B', 'F_C')}, r['task_d'],
              {k: r.get(k) for k in ('F_robust_proxy', 'F_novel_proxy', 'F_domain_proxy')},
              'composite', r['composite']['composite_proxy'])
if art.exists():
    for f in Path(WORK).glob('eval_ft_*.json'):
        shutil.copy(f, art)
zip_path = shutil.make_archive(os.path.join(WORK, 'ft_artifacts'), 'zip', WORK,
                               base_dir='artifacts') if art.exists() else None
print('download from the Output panel:', zip_path)
'''


SUBMIT = r'''
# 9) SUBMISSION — builds /kaggle/working/submission.csv from TEST_RUN_NAME's test arrays (official 0/5
#    convention, calibrated confidence) and validates all 500,000 ids / 5 columns. Aggregates only.
run('models.finetune.make_submission', '--name', TEST_RUN_NAME, '--out', os.path.join(WORK, 'submission.csv'))
'''

LLM_EVAL = r'''
# 10) LLM FEW-SHOT EVAL — Qwen2.5-7B-Instruct labels each held-out fold using 2 examples per class drawn
#     ONLY from the other folds. Compare F_A/F_B/F_C with FT0/FT1 before investing in an LLM teacher.
if RUN_LLM_EVAL:
    run('models.finetune.llm', 'eval', '--shots', 2, '--batch', 8)
else:
    print('RUN_LLM_EVAL is False - skipped')
'''

SYNTH = r'''
# 11) SYNTHETIC DATA — seeded spec grid (class x language x format x scenario), greedy decoding, official
#     definitions only (no sample.csv rows in prompts), schema/label/consistency checks, exact + near-dup removal.
if RUN_SYNTH:
    run('models.finetune.llm', 'synth', '--n', SYNTH_N, '--per-prompt', 5, '--batch', 16, '--out', SYNTH_PATH)
else:
    print('RUN_SYNTH is False - skipped')
'''

SCREEN = r'''
# 12) FT2 / FT3 SCREENING — main folds only (5 fits each). Run full proxies only for the winner (cell 5/6 style).
if RUN_FT2:
    run('models.finetune.train', '--name', 'ft_e5s_syn', '--main-only', *COMMON, *EXTRA['ft_e5s_syn'])
if RUN_FT3:
    run('models.finetune.train', '--name', 'ft_e5s_syn_corrupt', '--main-only', *COMMON, *EXTRA['ft_e5s_syn_corrupt'])
if not (RUN_FT2 or RUN_FT3):
    print('RUN_FT2 / RUN_FT3 are False - skipped')
'''


TEACHER = r'''
# 13) LLM TEACHER — Qwen2.5-7B-Instruct pseudo-labels a seeded random sample of TEST rows (code only, declared
#     semi-supervised use of test text). Fixed few-shot set (2 per class from sample.csv, ids in teacher_report.json).
#     Writes id,label,techniques,action incrementally; prints counts only.
if INSTALL_VLLM:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm'], capture_output=True, text=True)
    print('vllm install exit code', r.returncode, r.stderr[-1500:])
if RUN_TEACHER:
    run('models.finetune.llm', 'teacher', '--n', TEACHER_BENCH_N, '--shots', 2, '--backend', 'auto', '--batch', 32,
        '--out', os.path.join(WORK, 'teacher_bench.csv'))           # throughput benchmark (aggregate only)
    run('models.finetune.llm', 'teacher', '--n', TEACHER_N, '--shots', 2, '--backend', 'auto', '--batch', 32,
        '--max-minutes', TEACHER_MINUTES, '--out', TEACHER_PATH)
else:
    print('RUN_TEACHER is False - skipped')
'''

STUDENT = r'''
# 14) STUDENT SCREENING — e5-small on real rows (x REAL_REPEAT) + teacher pseudo-labels; main folds only.
#     Validation rows are real sample.csv rows only. Caveat: the 12 teacher shot rows can sit in a validation fold.
if RUN_STUDENT:
    run('models.finetune.train', '--name', 'ft_e5s_student', '--main-only', *COMMON, *STUDENT_ARGS)
else:
    print('RUN_STUDENT is False - skipped')
'''


EXTERNAL = r'''
# 15) EXTERNAL DATA — downloads 5 public datasets from Mendeley Data (original sources), maps ham->0 and promo->5
#     (full labels), smishing/scam -> {1,2,3,4} and generic spam -> {1..5} (partial labels; Task B/C masked),
#     dedups and caps per (source, label). Then FT3 = real + external, FT4 = real + synthetic + external (main folds).
if RUN_EXTERNAL:
    run('models.finetune.external', '--out', EXTERNAL_PATH, '--cap', EXTERNAL_CAP)
    run('models.finetune.train', '--name', 'ft_e5s_ext', '--main-only', *COMMON, *EXTRA['ft_e5s_ext'])
    if os.path.exists(SYNTH_PATH):
        run('models.finetune.train', '--name', 'ft_e5s_syn_ext', '--main-only', *COMMON, *EXTRA['ft_e5s_syn_ext'])
    else:
        print('no synthetic file yet - FT4 skipped')
else:
    print('RUN_EXTERNAL is False - skipped')
'''


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO.strip())] + [
        nbf.v4.new_code_cell(c.strip()) for c in (SETUP, CONFIG, BENCH, TOKENS, FT0, FT1, PREDICT, CHECK, SUBMIT, LLM_EVAL, SYNTH, SCREEN,
                                           TEACHER, STUDENT, EXTERNAL)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / "04_finetune.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
