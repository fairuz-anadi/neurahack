# Experiment log

All scores are local OOF on `folds/folds.csv` unless marked. Proxy columns are LOCAL estimates of the hidden
subsets, not Kaggle scores. Kaggle public scores go in the last column only.

| Exp ID | Family | Model / config | Dataset fp | Fold fp | F_A | F_B | F_C | Robust proxy | Domain proxy | Novel proxy | Calibration | Composite proxy | Runtime | Git commit | Kaggle public |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | classical | tfidf_lr: char_wb 2–5 + word 1–2, LR C=4 balanced, normalized_text | 026bb459116907ee | c357ab8ee361835a | 0.5662 | 0.4063 | 0.3683 | 0.5183 | 0.5131 | 0.4620 | 0.7435 | 0.5108 | 15 s (incl. stress) | uncommitted | — |

| FT0 | finetune | ft_e5s_base: multilingual-e5-small, 3 heads, frozen embeddings, 15 ep, lr 5e-5/1e-3, max_len 128, raw_text, no augmentation | 026bb459116907ee | c357ab8ee361835a | 0.6368 | 0.4404 | 0.3372 | 0.4491 | 0.5946 | 0.4857 | 0.7980 (cross-fit temp.; raw 0.7517) | 0.5473 | 30 min CPU (19 fits × ~95 s) | uncommitted (samprity/finetune) | — |

Notes for FT0: single seed (fine-tuning seed variance not yet measured). Bangla-script rows 3/6 correct (TF-IDF 0/6).
Beats TF-IDF on F_A (+0.071), F_B, domain and novel proxies and calibration; **worse on Task C (−0.031) and the
robustness proxy (−0.069)** → FT1 (in-fold adversarial copies) is the next ablation. CPU inference 68 rows/s
(max_len 128) ≈ 2 h for 500k on a 16-core laptop → GPU required; GPU throughput still to be measured on Kaggle.

Notes for E001: repeated grouped CV F_A 0.557 ± 0.018. Bangla-script rows (n=6): 0/6 correct. Most errors are
scams predicted as Legitimate, and Promotional spam predicted as Legitimate (8/12).
