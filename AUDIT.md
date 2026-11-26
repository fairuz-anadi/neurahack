# Rule-compliance audit: `notebooks/07_final_team.ipynb`

Audited on 27 Nov 2026 (branch `audit-fixes`), by static analysis only. Nothing was run, and no test row was opened or printed.
Cell numbers are 1-based positions in the notebook (24 cells). The clean copy keeps the same numbering.
The local copy has **no saved outputs**. Check 12 must therefore be re-run on the notebook downloaded from the Kaggle version that goes into the ZIP: `scripts/check_zip.py` does this.
`neurahack.ipynb` (scratch) is not imported by 07, so it was not audited again. Its git/token cells never reach the final notebook.

## Results

| # | Check | Result | Evidence (cell: quoted line) | Fix |
|---|---|---|---|---|
| 1 | Secrets (`token`, `GH_TOKEN`, `sk-`, `hf_`, `api_key`, `secret`, `password`, `UserSecretsClient`) | **PASS** (false positives only) | 4: `TOKEN = re.compile(r"[A-Za-z0-9@$]+")` (regex for leetspeak); 8: `token_pattern=r"\S+"`; 10: `AutoTokenizer.from_pretrained(MODEL_NAME)`; 6: `"Confirm your password and OTP here: {link}"` (synthetic scam template text). No `GH_TOKEN`, `sk-`, `hf_`, `api_key`, `secret` or `UserSecretsClient` anywhere. | none |
| 2 | Network / side effects | **WARN → fixed** | 18: `subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sentence-transformers'])` is unpinned (it only runs if the package is missing). There is no `git`, `clone`, `push`, `requests`, `curl`, `wget` or upload. The only downloads are Hugging Face models via `from_pretrained` / `SentenceTransformer(`. | Pinned to `sentence-transformers==5.7.0` |
| 3 | Declaration first, with all 5 parts | **FAIL → fixed** | Cell 1 was markdown but had no part 5 (API steps/costs) and no versions ("the cell below prints them"; the versions are actually printed in cell 4). | Cell 1 replaced with a 5-part declaration that matches the code. Two versions are still `<version>`: see "Missing versions" |
| 4 | Declared vs used models | **PASS** (1 WARN) | Loaded: 10 `AutoTokenizer/AutoModel.from_pretrained(MODEL_NAME)` → `xlm-roberta-base`; 18 and 22 `SentenceTransformer(name, …)` → `intfloat/multilingual-e5-large`, `BAAI/bge-m3`, `sentence-transformers/LaBSE`. All 4 are declared with licenses. No `pipeline(` and no LLM. WARN: 4 `MODEL_NAME = os.environ.get("SS_MODEL", "xlm-roberta-base")` would load an undeclared model if `SS_MODEL` were set (it is not on Kaggle). | none (defaults are declared) |
| 5 | Data reads | **PASS** | `read_csv`: only `sample.csv`, `test.csv`, `sample_submission.csv` (4, 18, 20, 22, 24) plus 14 `submission.csv` (written by this cell). `np.load`: `emb_*_{train,test}.npy` (written in 18), `folds_5x5.npy` (written in 20), `oof_/test_*_{A,B,C}.npy` (written in 14 and 20). No `open(`. No hand-made label files. | none |
| 6 | Test-row exposure | **WARN → fixed** | 14: `print(chk.head())` printed 5 per-row test predictions (ids + labels, no text). Test text is never printed. Test samples that are not printed: 12 `test["n"]…sample(60000, …)` (in-fold pseudo-label check), 18 `probe = [te_txt[i] for i in rng.choice(…, 5000 …)]` (speed timing only). Printed rows are train/synthetic only: 6 `synth.sample(8)`, 16 `err.sort_values…head(10)`, 18 `print(tr_txt[:3])`, 22 `print('clean:', txt[i] …)`. Saved outputs could not be checked locally (none saved). | `print(chk.head())` → aggregate print (confidence mean, NONE share) |
| 7 | Seeds | **FAIL → fixed** | 4 sets `SEED = 42` and seeds `random`/`np`. Cells 6 (`make_synth(…, seed=SEED)`), 8 (`random_state=SEED`, `random.Random(SEED + 7)`), 10 (`set_seed(seed)`), 12 (`.sample(…, random_state=SEED)`) and 14 (`XlmrModel().fit(…, seed=SEED)`) used SEED without setting it. 18, 20, 22, 24 set it. Folds use the literal `random_state=42` (8, 20). OK as is: 20/22 `LogisticRegression(...)` has no `random_state`, but lbfgs is deterministic. | `SEED = 42` added at the top of cells 6, 8, 10, 12, 14 (same value, no behavior change) |
| 8 | Leakage (OOF) | **PASS** (2 WARN) | 8 `run_cv`: each split fits on `build_train(tr_i, block=blocked_templates(va_i))`, so OOF comes only from train-fold models, and synthetic rows are never in validation. 10: XLM-R uses the same `run_cv` (repeat 0). 20: emb heads `lr().fit(X[tr], …)` → `X[va]`. No OOF comes from a model fit on all 86 rows. No synthetic data is generated from validation rows; the noisy copies in 8/22 are scored only, never trained on. WARN a: 8 notes "Several templates were written after studying sample.csv". Templates with char cosine > 0.35 to a validation row are blocked, but weaker paraphrases are not, so tfidf/xlmr CV may be optimistic. WARN b: 24 picks the weights and threshold on the same OOF it reports, so the printed ensemble CV is in-sample (the 50% shrink limits this). | none (model logic): see D3/D4 |
| 9 | Run order / outside state | **PASS** (1 FAIL-risk) | Every file is written before it is read in the same run (14 → 24, 18 → 20/22, 20 → 22/24). Part B/C cells reload the CSVs themselves, so redefining `train`, `OOF` and `splits` is harmless. No `/kaggle/input/<other-notebook>`. **Risk:** 24 `for name in ['tfidf', 'xlmr', 'ft']`: no cell creates `oof_ft_*` / `test_ft_*`. On a clean Save & Run All, `ft` is silently absent. If its files are copied into `/kaggle/working` from outside, those test predictions are not produced by notebook code (rule 1). | needs decision D2 |
| 10 | Ensemble cell | **PASS → strengthened** | 24: the family list covers every family the notebook saves (`emb` + `tfidf`, `xlmr`; `v2ens` is a tfidf+xlmr blend and is correctly left out to avoid double counting). Shapes are asserted: `assert O[k][j].shape == (n, 6) and TS[k][j].shape == (NT, 6)`. There are finite and range asserts. The validator checks columns, unique ids in test order, label/action 0–5, `fullmatch(r'NONE|T[1-6](\|T[1-6])*')` and no NaN. | Added asserts: int dtype, confidence in [0,1], ids == sample_submission ids, labels 0/5 → NONE + action 0, labels 1–4 → technique + action ≠ 0 |
| 11 | Runtime | **PASS** (estimate) | Part A: TF-IDF 25-split CV + pseudo-label check about 20–40 min; XLM-R 5 CV fits about 15–30 min; final fit, pseudo round 2 (about 36k rows) and 2 × inference about 45–60 min. Then embeddings about 47 min, and heads, B3 and ensemble about 15–20 min. **Total about 2.5–3.5 h** on a T4, well under 6 h (cell 1 says 3–3.5 h). | Confirm with the v1 run's "Run time" on Kaggle |
| 12 | Saved outputs | **N/A locally → cleared** | 07 has no saved outputs in git. The clean copy has outputs cleared. After the Kaggle run, check the downloaded notebook with `check_zip.py`, which scans the outputs for tokens, git, size and test text. | `check_zip.py` |

## Needs decision (model logic: not changed; would need a full rerun and team approval)

- **D1 Pseudo-labeling on test is ON in Part A.** 4: `USE_PSEUDO = True`. It only switches off if the in-fold gain is below −0.01 (12: `if PSEUDO_GAIN < -0.01`). So `test_tfidf_*` and `test_xlmr_*` come from round-2 models trained on pseudo-labels, while `oof_tfidf_*` and `oof_xlmr_*` come from models without them (14 comment: "OOF = shared-split CV without pseudo-labels"). The ensemble weights are tuned on one distribution and applied to another. The team's "pseudo-labeling rejected by CV" result was for the emb family only. Options:
  - (a) Keep it. It is declared and permitted (§5). The declaration now says so.
  - (b) Set `USE_PSEUDO = False` for the v2 run.
- **D2 `ft` family.** No code in this notebook produces `ft`. To include it, Samprity's training and inference code must go into the notebook as a cell that writes `oof_ft_{A,B,C}.npy` (86×6, shared folds) and `test_ft_{A,B,C}.npy` (500000×6) to `/kaggle/working/`. The model must also be added to the declaration table. Otherwise drop `ft`. Copying precomputed `test_ft` files in breaks rule 1.
- **D3 xlmr keep or drop** (the 8:15 call). Its OOF is repeat 0 only (5 fits), while emb and tfidf average 5 repeats. Averaged OOF is smoother, so the grid in 24 compares unlike things. Look at the "each family alone" lines in cell 24's output.
- **D4 The ensemble CV printed by cell 24 is in-sample.** It is optimistic and fine for ranking, but do not quote it as an unbiased score.
- **D5 Fixed emb weights.** 22/24 `W = EMB_W = [0.367, 0.417, 0.217]` and 22 `THR = 0.45` are hardcoded from an earlier CV run, not recomputed in this notebook. That is allowed (they are hyperparameters, not predictions) and is now declared. Recomputing them in-notebook would be cleaner.
- **D6 README disclosure wording.** `submission/README.md` §3.2 says some templates were written "with the general style of the unlabeled test messages in mind". Rule 2 forbids reasoning about individual test rows. The text says no row-level content was used. The team should confirm this sentence is accurate and keep or soften it deliberately.

## Missing versions (placeholders left)

- `python` and `transformers`: cell 4 prints them on Kaggle (python was added to that print in the clean copy). Copy them from the v2 run output into cell 1 and `submission/README.md`, replacing `<version>` and `[from run]`. `check_zip.py` and `build_zip.py` refuse placeholders.
- These were taken from the brief or the README's reference run, not from an output I could see: numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, torch 2.11.0, sentence-transformers 5.7.0. Verify them against the v2 run output.
- `submission/README.md` §5 still has `[from run]` for the XLM-R and ensemble CV rows.

## Diff summary: `07_final_team.ipynb` → `07_final_team_clean.ipynb`

| Cell | Change |
|---|---|
| 1 | Declaration rewritten with the 5 required parts (libraries table, resources + licenses, method + validation, seeds + reproduction, API steps/costs = none). It matches the code: synthetic data, Part A pseudo-labeling, fixed emb weights. The brief's draft was not used verbatim because it said pseudo-labeling and noise augmentation were rejected, which is true for emb only. |
| 4 | `import sys`; the version print now starts with `python <version>` |
| 6, 8, 10, 12, 14 | `SEED = 42` line added at the top |
| 14 | `print(chk.head())` → aggregate print (confidence mean, NONE share) |
| 18 | `pip install … 'sentence-transformers==5.7.0'` (pinned) |
| 24 | 6 extra validator asserts (dtype, confidence range, ids vs sample_submission, both post-rules) |
| all | Outputs and execution counts cleared (same as `nbconvert --clear-output`; done in Python because nbformat is not installed here) |

No model, hyperparameter, weight, threshold or post-rule was changed. No git/token cells existed in 07, so none were deleted. All code cells parse (`ast.parse`). The new asserts always hold given `post()`: rows labeled 0/5 have their technique bits zeroed and action 0; rows labeled 1–4 get at least one technique bit and `C[:, 0] = -1`.

## Packaging helpers

- `scripts/check_submission.py submission.csv --test test.csv --sample sample_submission.csv`: all format rules plus label/action/technique/confidence distributions. It prints only ids and counts, never text.
- `scripts/check_zip.py NeuraHack_Preli_Team_Hampton.zip --team Team_Hampton --test test.csv --sample sample_submission.csv`: checks the name, exactly 4 top-level entries, an empty or comment-only `.env`, the csv, the declaration (5 parts, no placeholders), secrets/git/curl in sources and outputs, test text in outputs (counts only), and whether the csv label share matches the notebook's printed share (same Kaggle version).
- Both use only the standard library and were tested on fake data with planted errors: every planted problem was caught.
- `submission/README.md`: added the library versions, "API steps: none, cost 0", "no public code", and the note that the emb family has no pseudo-labels or noise augmentation.
