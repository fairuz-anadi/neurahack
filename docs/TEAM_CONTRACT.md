# Team contract — ScamShield BD

Everyone uses the same loader, the same folds, the same metrics and the same file format.
If you need something different, change `common/` on `main` (Anadi reviews), never copy it into your folder.

## 1. Data loader

```python
from common.data_loader import load_config, load_train, load_test, get_targets, test_available
cfg = load_config()
train = load_train(cfg)          # 86 labeled rows, original order, original ids
yA, yB, yC = get_targets(train)  # (n,), (n,6) for T1..T6, (n,)
if test_available(cfg):          # True only on Kaggle — test.csv is never on our laptops
    test = load_test(cfg)        # 500,000 rows
```

Data folder search order: `SCAMSHIELD_DATA_DIR` → `data/` → `..` (repo's parent, e.g. `E:\neurahack`) →
any `/kaggle/input/*` folder with `sample_submission.csv`.

### Canonical columns

| column | meaning |
|---|---|
| `row_idx` | 0..n-1, original file order |
| `id` | original id, unchanged |
| `raw_text` | exact original text (never modified) |
| `normalized_text` | conservative normalized view (see §3) |
| `turns` | list of `(speaker, text)`; speaker is `S`/`R`/`A`/`B` in conversations, `None` for single messages |
| `n_turns`, `speakers`, `is_conversation` | conversation shape (turns are split on `|||`) |
| `script` | `bangla` / `latin` / `mixed` / `none`. **Latin = English or Banglish; script cannot tell them apart.** |
| `label`, `action` | Task A / Task C ints (train only) |
| `techniques`, `T1`..`T6` | canonical string + six 0/1 columns (train only) |

Official file schemas (checked 26 Nov): `sample.csv` = `id,text,label,techniques,action`;
`sample_submission.csv` = `id,label,techniques,action,confidence` (500,000 rows, ids 20001–520000).
`test.csv`: id + one text column (auto-detected; loader errors clearly if it can't find it).

## 2. Labels (column order of every matrix)

- **Task A** 0 Legitimate · 1 MFS/OTP scam · 2 Prize/lottery scam · 3 Fake job/investment · 4 Delivery/parcel phishing · 5 Promotional spam
- **Task B** T1 impersonation · T2 credential harvesting · T3 advance fee · T4 prize bait · T5 urgency/social engineering · T6 task/job/investment. Empty = `NONE`.
- **Task C** 0 none · 1 disclose info · 2 pay · 3 visit link · 4 install/join/task · 5 contact/call/reply

Source of truth: `common/labels.py` (`configs/labels.yaml` mirrors it; a test checks they match).

## 3. Text views

`normalized_text` = NFC, invisible characters removed (ZWJ/ZWNJ kept inside Bangla words), Cyrillic/Greek
lookalikes folded inside Latin words (`sаlаry` → `salary`), Latin lowercased, whitespace tidied.
Punctuation, emojis, numbers, URLs and Bangla marks are **kept**. Other views:

```python
from common.preprocessing import normalize, NormConfig, AGGRESSIVE
normalize(text, NormConfig(squeeze_repeats=True))   # 'PIIIN' -> 'piin'
normalize(text, AGGRESSIVE)                          # + NFKC + refang 'hxxps://x[.]top' -> 'https://x.top'
```
Transformers may prefer `raw_text` (their tokenizers handle case). Pick a view and write it in your metadata.

## 4. Folds (mandatory)

```python
from common.folds import load_folds, fold_fingerprint
fold = load_folds(train)          # aligned to train rows by id; values 0..4
```
- `folds/folds.csv` (`id,fold`) is the only split anyone trains/evaluates on. Never make your own split.
- Generated once by `python scripts/create_folds.py` (seed 42). Fingerprint: `c357ab8ee361835a`.
- Also available: `folds_repeats.csv` (5 re-splits), `folds_novel.csv`, `folds_domain.csv` — used by `common/stress.py`.
- **Synthetic / augmented / pseudo-labeled data:** generate it inside each training fold from that fold's
  training rows only. Never from validation rows. Evaluate on real rows only.

## 5. Plugging a model in (the only thing you implement)

```python
def fit_predict(train_df, valid_df):
    ...                                          # train on train_df only
    return {"A": pA, "B": pB, "C": pC}            # each (len(valid_df), 6)

from common.stress import run_stress
report, oof = run_stress(fit_predict, train, seed=42)   # main OOF + robust/novel/domain proxies
```
- `A`, `C`: softmax rows (sum to 1). `B`: six independent sigmoids (**no softmax across techniques**).
- A class missing from a training fold: give it probability 0 and renormalize (see `models/classical/tfidf_lr.py`).
- Expensive models: `run_stress(..., repeats=False)` skips the 5× repeated CV.

### Saving (the contract)

```python
from common.prediction_io import save_predictions
save_predictions("xlmr_mt", art_dir, oof, train["id"].to_numpy(),
                 {"config": {...}, "seed": 42, "dataset_fingerprint": train.attrs["fingerprint"],
                  "fold_fingerprint": fold_fingerprint(), "metrics": {...}},
                 test=test_pred, test_ids=test["id"].to_numpy())   # test only on Kaggle
```
Writes `artifacts/<model>/oof_{A,B,C}_<model>.npy`, `test_{A,B,C}_<model>.npy`, `oof_ids.npy`,
`test_ids.npy`, `metadata.json`. Shapes, NaN, ranges and row sums are checked on save.

### ID alignment

Never combine arrays by position. Load with ids:
```python
from common.prediction_io import load_predictions, check_metadata_compatible
p = load_predictions(Path("artifacts/xlmr_mt"), "oof", expected_ids=train["id"].to_numpy())
```
This reorders by id and raises if the id sets differ. `check_metadata_compatible` refuses models trained on
a different dataset or fold fingerprint.

## 6. Who does what

| Person | Branch | Owns |
|---|---|---|
| Anadi | `anadi/classical`, `main` | `common/`, `scripts/`, `models/classical/`, final notebook, Kaggle submissions, ZIP |
| Samprity | `samprity/finetune` | `models/finetune/`, `augmentation/` (except `corrupt.py` API), `notebooks/04_finetune.ipynb` |
| Rabib | `rabib/embeddings`, `rabib/ensemble` | `models/embeddings/`, `models/ensemble/`, calibration, `notebooks/03_embeddings.ipynb` |

Don't edit files you don't own; open a PR to `main` instead. No force-push.

### Samprity — transformer integration
1. Put code in `models/finetune/<name>.py` with a `fit_predict(train_df, valid_df)` and a
   `fit(train_df)` / `predict(df)` pair for the final test run.
2. Multi-task head: softmax(6) for A, sigmoid(6) for B, softmax(6) for C.
3. Augment only `train_df` inside `fit_predict` (`augmentation.corrupt.corrupt(text, seed)` for adversarial
   copies; synthetic generation from `train_df` rows only). Seed everything (`common.seed.set_seed`).
4. Run `run_stress(fit_predict, train, repeats=False)` on Kaggle GPU, `save_predictions(...)`.
5. Test set is 500k rows: predict in batches, fp16, `max_length` ≈ 128–192, and time it on 10k rows first.
6. Declare every model (name, Hugging Face URL, license) in `docs/RESOURCE_DECLARATION.md`.

### Rabib — embedding integration
1. Encode train + test once per embedding model on Kaggle GPU, save `emb_<model>_{train,test}.npy`
   (row order = loader order; save the ids next to them).
2. `fit_predict` slices the cached matrix with `train_df.row_idx` / `valid_df.row_idx` and fits light heads
   (LogReg / small MLP) for A, B (one binary head per technique), C.
3. Ensemble: load every model with `load_predictions(..., expected_ids=...)`, pick weights on OOF only,
   tune technique thresholds and confidence calibration on OOF, report with `common.metrics.evaluate`.
4. Confidence (Task D) is scored by Brier: calibrate `max(pA)` on OOF (temperature or isotonic) and check
   `Calibration` goes up.

## 7. Kaggle

- **Final notebook (what organizers re-run):** `python scripts/build_notebook.py` → upload
  `notebooks/05_final_pipeline.ipynb` (Kaggle → New Notebook → File → Import Notebook), attach the
  competition data, Run All. It contains all project code as `%%writefile` cells — no GitHub, no token,
  no precomputed predictions. It regenerates the folds and checks the fingerprint.
- **Dev work on Kaggle:** zip the code (no data) and upload it as a **private** Kaggle Dataset named
  `scamshield-code`; the dev notebooks look for `/kaggle/input/scamshield-code` automatically.
  ```powershell
  Compress-Archive -Path common,models,augmentation,configs,folds,scripts -DestinationPath $env:TEMP\scamshield-code.zip -Force
  ```
  (Optional: clone the private repo inside a dev notebook using a token stored in Kaggle **Secrets**.
  Never print it, never put it in the final notebook.)
- GPU: only switch it on for embedding / fine-tuning runs.
