# ScamShield BD: Multilingual Scam Message and Conversation Classification

**NSUCEC NeuraHack 2026, Preliminary Round**

| | |
|---|---|
| **Team** | Team_Hampton |
| **Members** | Fairuz Anadi, Rabib, Samprity |
| **Competition** | `nsucec-neurahack-2026-preliminary` (Kaggle Community Competition) |
| **Notebook** | `notebook.ipynb` (one self-contained notebook that produces `submission.csv`) |
| **Random seed** | `42` everywhere |
| **External APIs** | None (the `.env` file is intentionally empty) |

---

## 1. Summary

ScamShield BD classifies Bangladeshi SMS and chat messages (English, Bangla script, Banglish and mixed text,
single messages or conversations of up to 12 turns joined by `|||`) for all four tasks:

| Task | Output | How we predict it |
|---|---|---|
| A: primary class (0–5) | `label` | Ensemble of three model families, macro-F1-oriented decoding |
| B: attack techniques (T1–T6) | `techniques` | Six sigmoid outputs with one threshold tuned out-of-fold; `NONE` for classes 0 and 5 |
| C: intended victim action (0–5) | `action` | Separate softmax head; action 0 for classes 0 and 5 |
| D: confidence (0–1) | `confidence` | Task A probability after temperature scaling, fitted by Brier score |

Only 86 labeled rows are provided, so the approach has four parts:
1. **Programmatic synthetic training data** that covers every class, language, conversation format and hard legitimate case.
2. **Robust text normalization** against adversarial noise.
3. **Three complementary model families**: lexical, fine-tuned transformer and frozen multilingual embeddings.
4. **Validation that measures transfer to real messages**, not memorization of templates.

---

## 2. Package contents

| File | Description |
|---|---|
| `submission.csv` | Final predictions for all 500,000 test rows (`id,label,techniques,action,confidence`). It matches one of our selected Kaggle submissions. |
| `notebook.ipynb` | Reproducible notebook. It starts with the declaration section and generates every prediction from code. |
| `README.md` | This document. |
| `.env` | Empty, because the notebook uses no API keys. |

---

## 3. Method

### 3.1 Text normalization
The same cleaning is applied to training, validation and test text:
- Unicode NFKC normalization; zero-width and soft-hyphen characters removed.
- Cyrillic and Greek lookalike characters mapped to Latin; Bangla digits mapped to ASCII.
- Repeated characters collapsed; in-word leetspeak reversed (for example `p1n` → `pin`).
- Defanged links (`example[.]com`, `hxxp`) unified into one link form; text lowercased.

### 3.2 Synthetic training data (generated in the notebook)
- About **12,000 labeled rows** are generated **by code inside the notebook** from **223 hand-written templates** with random slots (brands, operators, amounts, fees, names, links, dates). Fixed seed: the same rows every run.
- Coverage:
  - all six classes in English, Banglish and Bangla script;
  - multi-turn conversations with speaker markers and group chats;
  - **hard legitimate cases**: genuine OTP messages that warn "do not share", transaction confirmations, real delivery updates, legitimate cashback promotions, and people quoting a scam to warn a friend (class 0).
- **Noise augmentation:** 35% of synthetic rows are corrupted: typos, character insertions and deletions, lookalike characters, zero-width characters, leetspeak, emojis and short meaningless strings. The 86 real rows are kept clean and upweighted ×6. An ablation chose this setting (Section 5).
- **Disclosure:** no test row text and no test label was copied into the training data. Some templates were written with the general style of the unlabeled test messages in mind (types of brands, warning conversations, message wrappers). No row-level content was taken.

### 3.3 Model families

| Family | Model | Description |
|---|---|---|
| `tfidf` | TF-IDF + logistic regression | Character 2–5-grams + word 1–2-grams; logistic-regression heads for label, action and the six techniques (one-vs-rest). Robust to spelling variation and obfuscation. |
| `xlmr` | `xlm-roberta-base`, fine-tuned | Multi-task head: label softmax, action softmax and 6 technique sigmoids. 3 epochs, learning rate 3e-5, batch size 32, max length 192, fp16. |
| `emb` | `multilingual-e5-large`, `bge-m3`, `LaBSE` (frozen) | Sentence embeddings of the cleaned text; logistic-regression heads per encoder for A, B and C, blended with weights chosen by CV. |

### 3.4 Semi-supervised pseudo-labeling (declared, Rulebook §5)
After the first full training run, the notebook selects up to **4,000 test rows per class** on which TF-IDF and XLM-R **agree with high confidence**. It adds them to the training data and trains one more round.
- This is done entirely by code and uses **only model predictions**; no test labels are looked up or edited by hand.
- An in-fold check measures the gain on the real labeled rows before the round is applied. It found **+0.02 to +0.06 Task A F1** for TF-IDF. If the check shows a loss larger than 0.01, the notebook turns pseudo-labeling off automatically.
- With pseudo-labels, the final models train on up to about 36,000 rows (12k synthetic + up to 24k pseudo-labeled + upweighted real rows).
- This applies to the `tfidf` and `xlmr` families only. For the embedding family, pseudo-labeling and noise augmentation were tested and rejected by CV, so that family is trained on the 86 real rows only. Pseudo-labeling scored proxy 0.725 vs 0.759 without it. Noise augmentation scored 0.727 vs 0.756 on 0.8 × clean + 0.2 × noisy text.

### 3.5 Ensemble, decoding and calibration
- **Ensemble:** the family probabilities for A, B and C are averaged with weights. The weights and the technique threshold are chosen by grid search on out-of-fold predictions, then **shrunk 50% toward equal weights** to avoid overfitting 86 rows.
- **Consistency rules from the labeling guide:**
  - Classes 0 and 5 get `NONE` and action 0.
  - Fraud classes (1–4) always get at least one technique and a non-zero action.
- **Confidence:** a temperature for the Task A probabilities is fitted on out-of-fold predictions by minimizing the Brier score, as Task D asks.

---

## 4. Validation strategy

- **Folds:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` over the 86 real labeled rows. All team members used these folds, so out-of-fold predictions from all families can be ensembled. Out-of-fold probabilities are averaged over repeats. XLM-R uses the first repeat (5 fits) to fit the time budget.
- **Real rows only in validation:** synthetic rows are only ever in the training part. Every score therefore measures transfer to real, human-written messages.
- **Template blocking (leakage control):** in every split, synthetic templates whose text resembles a validation row (character TF-IDF cosine > 0.35) are removed from training. Without this blocking, CV was inflated to about 0.99.
- **Robustness:** each fold also scores a corrupted copy of its validation rows, to estimate the adversarial-noise score.
- **Segment analysis:** the notebook reports accuracy by language, single message vs conversation, hard legitimate cases, and clean vs noisy text (Part B3), plus confusion matrices and the worst errors (Section 7 of the notebook).
- **Final submissions** are chosen by these CV results, not by the public leaderboard, which covers only 30% of the test set.

---

## 5. Results (out-of-fold, real labeled rows)

| Model | Task A macro-F1 | Task B macro-F1 | Task C macro-F1 | Task A, noisy text |
|---|---|---|---|---|
| Baseline (TF-IDF on the 86 rows only) | 0.566 | – | – | – |
| Embedding family (e5 + bge-m3 + LaBSE) | 0.841 | 0.737 | 0.576 | – |
| TF-IDF + synthetic data (v2.1) | 0.877 | 0.752 | 0.710 | 0.842 |
| XLM-R fine-tuned + synthetic data | `[from run]` | `[from run]` | `[from run]` | – |
| **Final ensemble** | `[from run]` | `[from run]` | `[from run]` | – |

**Noise-augmentation ablation** (TF-IDF, 25 splits, Task A clean / noisy):

| Setting | Clean | Noisy |
|---|---|---|
| No noise | 0.852 | 0.804 |
| Noise on real-row copies only | 0.839 | 0.793 |
| Noise on synthetic + real rows | 0.865 | 0.842 |
| **Noise on synthetic rows only (used)** | **0.877** | **0.842** |

---

## 6. External resources and licenses

| Resource | Source | License | Use |
|---|---|---|---|
| `xlm-roberta-base` | Hugging Face Hub (`FacebookAI/xlm-roberta-base`) | MIT | Pretrained encoder, fine-tuned |
| `multilingual-e5-large` | Hugging Face Hub (`intfloat/multilingual-e5-large`) | MIT | Frozen sentence embeddings |
| `bge-m3` | Hugging Face Hub (`BAAI/bge-m3`) | MIT | Frozen sentence embeddings |
| `LaBSE` | Hugging Face Hub (`sentence-transformers/LaBSE`) | Apache-2.0 | Frozen sentence embeddings |
| `sentence-transformers` | PyPI | Apache-2.0 | Encoding with the three models above |
| Competition data | Kaggle competition | Organizer | `sample.csv` for training and validation; `test.csv` for inference and pseudo-labeling |

- **No external datasets.**
- **No hosted LLM or inference API calls.** API steps: none. API cost: 0. The `.env` file is empty.
- **No public code** beyond library documentation.
- AI coding assistants were used while writing the code, which the rules permit.

---

## 7. Reproduction

**Environment:** Kaggle Notebook, latest container image, **GPU T4**, **Internet ON** (needed to download the four pretrained models from Hugging Face).

**Libraries:** the versions preinstalled in the Kaggle image. Reference run: python `[from run]`, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, torch 2.11.0, transformers `[from run]`, sentence-transformers 5.7.0. The notebook prints all of them (setup cell and cell B1). If the image lacks `sentence-transformers`, it is installed as the pinned version `5.7.0`. No other installs are needed.

**Steps:**
1. Create a Kaggle notebook and import `notebook.ipynb`.
2. Add the competition data `nsucec-neurahack-2026-preliminary` as an input. The notebook finds it under `/kaggle/input/`.
3. Settings: Accelerator **GPU T4**, Internet **ON**.
4. Click **Save Version → Save & Run All (Commit)**.
5. The notebook writes `/kaggle/working/submission.csv`. The last cell overwrites the intermediate submission with the final ensemble.

**Runtime:** about 3–3.5 hours on a T4.

| Stage | Time |
|---|---|
| Synthetic data and TF-IDF CV | ~10 min |
| XLM-R CV and final training | ~30 min |
| Test inference and pseudo-label round | ~45–60 min |
| Embeddings | ~47 min |
| Heads, analysis and ensemble | ~20 min |

**Seeds:** `SEED = 42` for Python `random`, NumPy, PyTorch (CPU and CUDA) and `PYTHONHASHSEED`. cuDNN runs in deterministic mode. The cross-validation folds also use `random_state = 42`. Some GPU kernels have small run-to-run variation; we expect reruns to be within the 0.01 reproduction tolerance.

**Optional environment variables** (defaults reproduce the submission): `SS_MODEL`, `SS_N_SYNTH`, `SS_EPOCHS`, `SS_XLMR_REPEATS`, `SS_RUN_CV`, `SS_USE_XLMR`.

---

## 8. Rule compliance

- [x] Every prediction is produced by code in `notebook.ipynb`; no hand labeling, editing or hardcoded predictions.
- [x] Synthetic training data is generated by code in the notebook; no test example or label is copied.
- [x] Pseudo-labeling on `test.csv` is done by code, uses only model predictions, and is declared (Rulebook §5).
- [x] All pretrained models and libraries are declared with source and license; no external datasets or APIs.
- [x] No API keys anywhere; `.env` is included and empty.
- [x] The submission format matches `sample_submission.csv`: every test `id` exactly once, valid codes, no blank fields.

---

## 9. Limitations

- Only 86 real labeled rows exist, so validation scores have high variance. We reduce it with 25 repeated splits but can't remove it.
- The synthetic templates reflect our own understanding of Bangladeshi scam patterns. Scam scenarios that are completely new may still be misclassified, especially when a fraud class is close to legitimate promotions (class 5 vs class 2).
- Task B and Task C depend on the Task A decision, because of the consistency rules. A wrong class can cascade into wrong techniques and action.
- The composite score in the notebook is a proxy: the hidden domain-shift and novel-scenario subsets are approximated by Task A F1.
