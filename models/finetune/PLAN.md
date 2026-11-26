# Phase 2 — Fine-tuning, data expansion, robustness (Samprity)

Branch: `samprity/finetune`. Owner files: `models/finetune/`, `augmentation/` (new modules only),
`notebooks/04_finetune.ipynb`. Shared code (`common/`, `folds/`, `scripts/`) is used, not changed.

## Integration requirements (from TEAM_CONTRACT)
- Train/evaluate only on `folds/folds.csv` (fingerprint `c357ab8ee361835a`) through `common.stress.run_cv / run_stress`.
- `fit_predict(train_df, valid_df) -> {"A","B","C"}`: A/C softmax, B six independent sigmoids.
- Save with `common.prediction_io.save_predictions` (oof/test A/B/C + ids + metadata with dataset/fold fingerprints).
- Augmented / synthetic / pseudo-labeled rows are created **inside each training fold only**, never from
  validation rows; evaluation is always on real labeled rows.
- `test.csv` exists only on Kaggle. Real test inference runs only there; locally we use stand-in fixtures.
- Every model / dataset goes into `docs/RESOURCE_DECLARATION.md` + the notebook Declaration.

## Backbone choice (initial)
**`intfloat/multilingual-e5-small`** — 118M params, 12 layers × 384 hidden, XLM-R SentencePiece vocab (250k, covers
Bangla script and Latin Banglish), contrastively pre-trained sentence encoder, **MIT license** (checked on the official
model card; `bn` listed). Reasons:
1. 86 labels: a sentence-embedding-pretrained encoder starts from useful sentence geometry; raw MLM encoders
   (XLM-R base/large) are known to be unstable with <100 examples.
2. 500k test rows: ~3–4× cheaper than base-size models at inference; fits Kaggle T4 time comfortably.
3. Trainable on our laptop CPU for development (no GPU locally).
Rabib uses *frozen* e5-large / bge-m3 / LaBSE; this model is *fine-tuned* and smaller → different errors for the ensemble.

Later candidates for ablation (GPU, Kaggle): `xlm-roberta-base` (MIT), `microsoft/mdeberta-v3-base` (MIT),
`intfloat/multilingual-e5-base` (MIT), `csebuetnlp/banglabert` (Bangla-only; check license). Verify each license before use.

## Architecture
Shared encoder → mean pooling → dropout → three linear heads:
A (6, softmax, class-weighted CE) · B (6, sigmoid, BCE with capped pos_weight) · C (6, softmax, class-weighted CE).
Loss = 1.0·L_A + 0.5·L_B + 0.5·L_C (A carries 40% of the score directly and feeds domain/robust/novel).
**Task D**: temperature scaling of Task A probabilities, temperature chosen to minimize the Brier score of
max-probability vs. correctness (exactly what Calibration measures), **cross-fitted** across folds so the
reported calibration is not fit on the rows it is scored on. Argmax is unchanged.

## Prioritized experiment plan (one change at a time, all on the shared folds)
| # | Experiment | Measures | Status |
|---|---|---|---|
| FT0 | e5-small multi-task, real 86 rows only | transformer baseline vs. TF-IDF (0.566) | **done: F_A 0.637, robust proxy 0.449 (worse than TF-IDF)** |
| FT1 | + in-fold adversarial copies (`augmentation.corrupt`, 2 per row) | robustness proxy, F_A | next (Kaggle GPU; ~3x CPU cost locally) |
| FT2 | + rule-free script/transliteration views (Bangla ↔ Banglish via local model, in-fold) | Bangla slice | later |
| FT3 | + external public data, mapped conservatively (see below), as auxiliary/weak labels | F_A, domain proxy | needs license check |
| FT4 | + synthetic data from a local open LLM (Qwen2.5-3B/7B-Instruct, fixed seed, greedy), generated per fold from that fold's training rows + class definitions | novel proxy | later, Kaggle GPU |
| FT5 | + pseudo-labels on unlabeled test (confident, capped per class), Kaggle only, declared | all | last |
| FT6 | backbone swap (e5-base / xlm-r-base / mdeberta) with the winning data recipe | all | GPU |

Ablation rule: every data source is reported **with and without** it, with the same seed and folds; keep it only
if Task A OOF F1 improves by more than the fold noise (baseline repeated-CV std ≈ 0.018) or a proxy improves
without hurting F_A.

## External data candidates (licenses NOT yet verified — do not use until checked on the dataset page)
| Dataset | Content | Possible use |
|---|---|---|
| Bangalabarta: Spam/Smishing SMS Dataset Bangla (Mendeley, DOI 10.17632/jfkfbw3gzh) | 2,772 Bangla SMS: smishing / promotional / normal | Bangla script coverage; normal→0, promotional→5, smishing→fraud (aux "is_fraud" head, not a 6-way label) |
| BTTC: Bangla Tri-class Text Corpus (Mendeley, DOI 10.17632/5wrm959d6f) | 10,283 Bangla spam / ham / promotional | same as above, larger |
| Labeled Bangla SMS Spam and Ham (Mendeley, DOI 10.17632/hhd43jbv9p) | Bangla spam / ham | aux binary |
| Financial scams detection dataset (Mendeley, DOI 10.17632/znsk27yk3h) | English + Bangla financial scam / ham | aux binary, MFS-scam flavour |
| SMS Phishing Dataset (Mishra & Soni, Mendeley, DOI 10.17632/f45bkkt8pr) | 5,971 English ham / spam / smishing (overlaps UCI SMS Spam) | English aux binary |

None of these carry our Task B/C labels, and their "spam" ≠ our 6 classes. Plan: a 4th auxiliary head
(`fraud / promo / legit`) trained on external rows, while heads A/B/C are trained on real (+ in-fold synthetic)
rows only. External rows are never placed in validation.

## Compute (measured 26 Nov, laptop CPU 16 threads, no GPU)
- Training: ~95 s per fold (69 rows, 15 epochs, frozen embeddings; 240 s with embeddings trainable).
  FT0 with all proxies = 19 fits ≈ 30 min. FT1 (3× rows) ≈ 3× that.
- Inference on stand-in rows: 68 rows/s at max_len 128, 47 rows/s at 256 → 500k ≈ 2.0 h / 3.0 h on this CPU.
  Kaggle CPU sessions have fewer cores, so **test inference must run on a Kaggle GPU**.
- GPU throughput is NOT measured yet: run cell 2 of `notebooks/04_finetune.ipynb` on Kaggle before the 500k run.
- Truncation: training texts are ≤ 89 tokens; 12-turn test conversations can exceed 128 tokens → compare
  max_len 128 vs 256 cost on GPU and check how many test rows get truncated (aggregate count only).
