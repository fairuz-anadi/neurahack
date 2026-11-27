# FT family handoff: `ft_integration_cell.py` (Samprity, config E_balanced)

One self-contained Kaggle cell. Paste it into the team notebook **before** the ensemble cell and run it.
It needs GPU T4 and internet ON (to download `intfloat/multilingual-e5-small` from Hugging Face).

## Outputs (`/kaggle/working/`)

| file | shape | rows | columns |
|---|---|---|---|
| `oof_ft_A.npy`, `oof_ft_C.npy` | (86, 6) float32 | `sample.csv` order | softmax over classes / actions 0..5 |
| `oof_ft_B.npy` | (86, 6) float32 | `sample.csv` order | 6 independent sigmoids, T1..T6 |
| `test_ft_A.npy`, `test_ft_C.npy` | (500000, 6) float32 | `test.csv` order | softmax |
| `test_ft_B.npy` | (500000, 6) float32 | `test.csv` order | sigmoids |
| `ft_manifest.json` | | | config, data hashes, fold fingerprint, timings, OOF scores, versions |

The probabilities are raw (no temperature); the ensemble cell does calibration. The ensemble already loads `ft` by name.

## Folds and OOF aggregation

- Split = the team's `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` on `label`, i.e. `folds_5x5.npy`.
  If `/kaggle/working/folds_5x5.npy` exists, the cell asserts it equals the rebuild. Otherwise it rebuilds it.
- Each repeat predicts every row once, from the model that did not see it. The saved OOF is the **mean over repeats**,
  the same as the embedding family (`oo.mean(0)`).
- `FT_REPEATS=5` (default) runs 25 fits. `FT_REPEATS=1` runs only repeat 0 (5 fits), like the xlmr family.

## Rulebook v1.3 compliance

- Training data: `sample.csv` (86 rows), plus synthetic rows, plus CC BY 4.0 external rows. Both extra files are
  embedded in the cell, checked by SHA-256, and are training-only.
- **No pseudo-labelling.** E_balanced never had a `pseudo_path`; the AutoTrain report's `kw` for E_balanced lists only
  synthetic and external paths. The teacher candidate was not part of the selection.
- No test text is used in any fit, near-duplicate filter, augmentation, calibration or threshold. `test.csv` is opened only
  after the final model is trained, and it is used only for inference. Identical test texts share one forward pass
  (this is an inference shortcut only).
- The synthetic rows were written by Qwen2.5-7B-Instruct from the official class definitions only (no sample or test rows
  in prompts). Rows near `sample.csv` were removed.

## Resource declaration (add to the team Declaration cell)

| Resource | Type | Source | Licence | Use |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | pretrained encoder | https://huggingface.co/intfloat/multilingual-e5-small | MIT | fine-tuned (FT family) |
| `Qwen/Qwen2.5-7B-Instruct` | open-weight LLM, run locally on Kaggle, no API | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Apache-2.0 | wrote the embedded synthetic training rows (offline, before this cell) |
| BTTC: Bangla Tri-class Text Corpus | dataset | https://data.mendeley.com/datasets/5wrm959d6f/1 (DOI 10.17632/5wrm959d6f.1) | CC BY 4.0 | external training rows |
| Labeled Bangla SMS Spam and Ham | dataset | https://data.mendeley.com/datasets/hhd43jbv9p/1 (DOI 10.17632/hhd43jbv9p.1) | CC BY 4.0 | external training rows |
| Financial scams detection dataset | dataset | https://data.mendeley.com/datasets/znsk27yk3h/2 (DOI 10.17632/znsk27yk3h.2) | CC BY 4.0 | external training rows |
| SMS Phishing Dataset (Mishra & Soni) | dataset | https://data.mendeley.com/datasets/f45bkkt8pr/1 (DOI 10.17632/f45bkkt8pr.1) | CC BY 4.0 | external training rows |
| torch, transformers, tokenizers, numpy, pandas, scikit-learn | libraries | Kaggle image (versions saved in `ft_manifest.json`) | BSD-3 / Apache-2.0 / Apache-2.0 / BSD-3 / BSD-3 / BSD-3 | |

Not used: Bangalabarta (CC BY-NC-SA 4.0), hosted APIs, pseudo-labels, private files.
The team text "No external datasets" must change once this cell is included.
Attribution for the external snapshot is in `snapshots/README.md`.

## Resources and runtime (one T4; measured on Kaggle in AutoTrain v2)

| step | estimate |
|---|---|
| model download | < 1 min |
| 25 CV fits (measured ~48 s each, max_len 128) | ~21 min |
| final fit on all 86 rows (max_len 256) | ~1.5 min |
| inference, 500k rows (measured ~2,460 rows/s at max_len 256) | ≤ 3.5 min |
| **total, FT_REPEATS=5** | **~27 min** |
| **total, FT_REPEATS=1** | **~10 min** |

GPU memory: measured peak 1.6 GB during CV. Expect ≤ 3 GB for the max_len-256 final fit. One GPU is used.

Measured OOF on the AutoTrain main folds (not the team folds): F_A 0.855, F_B 0.713, F_C 0.638, accuracy 0.837.
The cell prints its own OOF scores on the team folds.

## Rebuild the cell

```
python -m models.finetune.make_integration_cell --synth E:/neurahack/synth_v2/synth_filtered.csv
```

Local check (CPU, tiny random encoder, made-up data; no real training):
`FT_SMOKE=1 FT_REPEATS=2 FT_DATA_DIR=... FT_OUT_DIR=... python ft_integration_cell.py`.
The inlined trainer gives bit-identical predictions to `models/finetune/multitask.py` on the same inputs.
