# Experiment log

All scores are local OOF on `folds/folds.csv` unless marked. Proxy columns are LOCAL estimates of the hidden
subsets, not Kaggle scores. Kaggle public scores go in the last column only.

| Exp ID | Family | Model / config | Dataset fp | Fold fp | F_A | F_B | F_C | Robust proxy | Domain proxy | Novel proxy | Calibration | Composite proxy | Runtime | Git commit | Kaggle public |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | classical | tfidf_lr: char_wb 2–5 + word 1–2, LR C=4 balanced, normalized_text | 026bb459116907ee | c357ab8ee361835a | 0.5662 | 0.4063 | 0.3683 | 0.5183 | 0.5131 | 0.4620 | 0.7435 | 0.5108 | 15 s (incl. stress) | uncommitted | — |

Notes for E001: repeated grouped CV F_A 0.557 ± 0.018. Bangla-script rows (n=6): 0/6 correct. Most errors are
scams predicted as Legitimate, and Promotional spam predicted as Legitimate (8/12).
