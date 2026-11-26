# ScamShield BD — NeuraHack 2026 (private)

Shared ML infrastructure for the ScamShield BD preliminary round: one data loader, one set of grouped folds,
one metrics module, one prediction format, and a CPU baseline. Read `docs/TEAM_CONTRACT.md` before writing code.

**Never commit competition data, `.env`, Kaggle tokens or model checkpoints. Keep this repo private.**

## Local setup (Windows / PowerShell)

Put `sample.csv` and `sample_submission.csv` in the folder above this repo (or in `data/`, or set
`$env:SCAMSHIELD_DATA_DIR`). `test.csv` is only used on Kaggle.

```powershell
python -m pip install -r requirements.txt
python -m pytest -q                       # 23 tests
python scripts/inspect_data.py            # outputs/eda_report.md
python scripts/create_folds.py            # folds/*.csv (already committed; only Anadi re-runs)
python scripts/train_baseline.py          # artifacts/tfidf_lr/, outputs/eval_tfidf_lr.json
python scripts/build_notebook.py          # notebooks/*.ipynb (05 = self-contained final notebook)
```

On Kaggle (with `test.csv`), `train_baseline.py` and the final notebook also write a validated `submission.csv`.

```powershell
python scripts/validate_submission.py outputs/submission.csv
python scripts/build_zip.py --team "TeamName" --code NH000 --csv outputs/submission.csv --notebook notebooks/05_final_pipeline.ipynb
```

## Layout

```
configs/      base.yaml (paths, seed, folds, submission rules), labels.yaml
common/       data_loader, schema, preprocessing, labels, folds, metrics, stress, prediction_io, submission, seed
models/       classical/ (Anadi), embeddings/ + ensemble/ (Rabib), finetune/ (Samprity)
augmentation/ corrupt.py (adversarial noise, shared)
folds/        shared fold manifests + fingerprints (committed)
scripts/      inspect_data, create_folds, train_baseline, validate_submission, build_notebook, build_zip
notebooks/    01–04 dev notebooks, 05 final self-contained notebook (generated)
docs/         TEAM_CONTRACT, EXPERIMENT_LOG, RESOURCE_DECLARATION
```

Scores in this repo are local OOF / proxy numbers, never Kaggle scores.
