"""Writes notebooks/04_finetune.ipynb (Kaggle GPU runner for the fine-tuning family).

    python -m models.finetune.make_notebook

NOTE for Anadi: scripts/build_notebook.py also writes a 04 template; run this after it (or drop 04 from
build_notebook.py) so this version is not overwritten.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

INTRO = """# 04 — Multi-task transformer fine-tuning (Samprity)

Kaggle GPU runner for `models/finetune`. Attach: the competition data and the private `scamshield-code`
dataset (zip of the repo code, no data). Accelerator: GPU T4 / P100. Internet: ON (first model download).

- Backbone: `intfloat/multilingual-e5-small` (MIT) — shared encoder + heads for Tasks A, B, C; Task D via
  cross-fitted temperature scaling.
- Validation: shared `folds/folds.csv`; proxies from `common.stress` (local estimates, not Kaggle scores).
- Test data is only read by code in this notebook; nothing is printed row-by-row.
"""

SETUP = r'''
import os, sys, subprocess, json
CODE_DIR = os.environ.get('SCAMSHIELD_CODE_DIR') or ('/kaggle/input/scamshield-code' if os.path.isdir('/kaggle/input/scamshield-code') else os.path.abspath('..'))
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('.')
ENV = dict(os.environ, SCAMSHIELD_ARTIFACTS_DIR=os.path.join(WORK, 'artifacts'), SCAMSHIELD_OUTPUTS_DIR=WORK,
           PYTHONPATH=CODE_DIR)
def run(*args):
    r = subprocess.run([sys.executable, '-m', *args], cwd=CODE_DIR, env=ENV, capture_output=True, text=True)
    print(r.stdout[-6000:]); print(r.stderr[-3000:] if r.returncode else '')
    r.check_returncode()
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
'''

BENCH = r'''
# 1) Throughput on stand-in rows (training texts stitched into conversations) - sizes the 500k run.
run('models.finetune.benchmark', '--rows', '4000', '--batch', '128', '256', '--max-len', '128', '256', '--conv-share', '0.1')
'''

TRAIN = r'''
# 2) FT0 baseline on the shared folds (+ robust/novel/domain proxies). If test.csv is attached, also
#    fits on all 86 rows and saves test_{A,B,C}/test_conf arrays in artifacts/ft_e5s_base/.
run('models.finetune.train', '--name', 'ft_e5s_base')
'''

ABL = r'''
# 3) FT1 ablation: + in-fold adversarial copies. Compare with FT0 in EXPERIMENT_LOG.
run('models.finetune.train', '--name', 'ft_e5s_corrupt', '--augment', 'corrupt')
'''

SHOW = r'''
for name in ['ft_e5s_base', 'ft_e5s_corrupt']:
    p = os.path.join(WORK, f'eval_{name}.json')
    if os.path.exists(p):
        r = json.load(open(p))
        print(name, {k: r['main'][k] for k in ('F_A', 'F_B', 'F_C', 'Calibration')}, r['task_d'],
              {k: r.get(k) for k in ('F_robust_proxy', 'F_novel_proxy', 'F_domain_proxy')}, r['composite'])
'''


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP.strip()),
                nbf.v4.new_code_cell(BENCH.strip()), nbf.v4.new_code_cell(TRAIN.strip()),
                nbf.v4.new_code_cell(ABL.strip()), nbf.v4.new_code_cell(SHOW.strip())]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / "04_finetune.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
