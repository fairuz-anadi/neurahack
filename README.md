# ScamShield BD — NeuraHack 2026 (Team_Hampton)

Code for the ScamShield BD preliminary round (NSUCEC NeuraHack 2026): classify Bangladeshi SMS and chat messages
for Task A (class 0–5), Task B (techniques T1–T6), Task C (victim action 0–5) and Task D (confidence).
Team: Fairuz Anadi, Rabib, Samprity. Read `docs/TEAM_CONTRACT.md` before writing code.

**Never commit competition data (`sample.csv`, `test.csv`, `sample_submission.csv`), `.env`, Kaggle or Hugging Face
tokens, or model checkpoints. This repository is public.**

## Final notebooks (run on Kaggle, GPU T4)

| notebook | what it is |
|---|---|
| `notebooks/09_v3.ipynb` | final team pipeline: synthetic data + TF-IDF, XLM-R base, frozen embeddings (e5-large, bge-m3, LaBSE), ensemble |
| `notebooks/10_v3_large.ipynb` | same, with XLM-R large |
| `notebooks/11_v3_large_ft.ipynb` | `10_v3_large` + the FT family (fine-tuned multilingual-e5-small, see below) |

Rulebook v1.3: no training on test text. Pseudo-labelling is removed; `test.csv` is used only for final inference.
Submission package notes: `submission/README.md`. Rule-compliance audit: `AUDIT.md`.

## FT family (Samprity)

- `notebooks/ft_integration_cell.py`: one self-contained Kaggle cell. It trains and validates the selected configuration
  (E_balanced) on the team folds (`folds_5x5`), then predicts test. It writes `oof_ft_{A,B,C}.npy` (86×6) and
  `test_ft_{A,B,C}.npy` (500000×6). It is built by `models/finetune/make_integration_cell.py`;
  `scripts/make_ft_notebook.py` inserts it into a team notebook.
- `docs/FT_INTEGRATION.md`: outputs, fold handling, compliance, resource declaration, runtime and GPU memory.
- `notebooks/ScamShield_AutoTrain.ipynb`: the unattended Kaggle run that screened the FT candidates and selected E_balanced.
- `snapshots/external_v2.csv.gz`: four CC BY 4.0 external datasets (training only). Attribution: `snapshots/README.md`.

## Local setup (Windows / PowerShell)

Put `sample.csv` and `sample_submission.csv` in the folder above this repo (or in `data/`, or set
`$env:SCAMSHIELD_DATA_DIR`). `test.csv` is only used on Kaggle.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q                       # 41 tests
python scripts/inspect_data.py            # outputs/eda_report.md
python scripts/create_folds.py            # folds/*.csv (already committed; only Anadi re-runs)
python scripts/train_baseline.py          # artifacts/tfidf_lr/, outputs/eval_tfidf_lr.json
```

Checks before submitting:

```powershell
python scripts/check_submission.py submission.csv --test test.csv --sample sample_submission.csv   # on Kaggle
python scripts/check_zip.py NeuraHack_Preli_Team_Hampton.zip --team Team_Hampton
python scripts/build_zip.py --help
```

## Layout

```
configs/      base.yaml (paths, seed, folds, submission rules), labels.yaml
common/       data_loader, schema, preprocessing, labels, folds, metrics, stress, prediction_io, submission, seed
models/       classical/ (Anadi), embeddings/ + ensemble/ (Rabib), finetune/ (Samprity)
augmentation/ corrupt.py (adversarial noise, shared)
folds/        shared fold manifests + fingerprints (committed)
scripts/      data checks, folds, baseline, notebook builders (make_v2/v3/ft), submission and ZIP checks
notebooks/    01–05 early dev notebooks, 06–08 earlier team versions, 09–11 final, FT cell, AutoTrain
snapshots/    CC BY 4.0 external data snapshot (FT family)
submission/   README for the submission package
docs/         TEAM_CONTRACT, EXPERIMENT_LOG, RESOURCE_DECLARATION, FT_INTEGRATION
```

Scores in this repo are local out-of-fold or proxy numbers, never Kaggle scores.
