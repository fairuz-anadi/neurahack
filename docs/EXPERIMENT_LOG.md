# Experiment log

All scores are local OOF on `folds/folds.csv` unless marked. Proxy columns are LOCAL estimates of the hidden
subsets, not Kaggle scores. Kaggle public scores go in the last column only.

| Exp ID | Family | Model / config | Dataset fp | Fold fp | F_A | F_B | F_C | Robust proxy | Domain proxy | Novel proxy | Calibration | Composite proxy | Runtime | Git commit | Kaggle public |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E001 | classical | tfidf_lr: char_wb 2–5 + word 1–2, LR C=4 balanced, normalized_text | 026bb459116907ee | c357ab8ee361835a | 0.5662 | 0.4063 | 0.3683 | 0.5183 | 0.5131 | 0.4620 | 0.7435 | 0.5108 | 15 s (incl. stress) | uncommitted | — |

| FT0 | finetune | ft_e5s_base: multilingual-e5-small, 3 heads, frozen embeddings, 15 ep, lr 5e-5/1e-3, max_len 128, raw_text, no augmentation | 026bb459116907ee | c357ab8ee361835a | 0.6368 | 0.4404 | 0.3372 | 0.4491 | 0.5946 | 0.4857 | 0.7980 (cross-fit temp.; raw 0.7517) | 0.5473 | 30 min CPU (19 fits × ~95 s) | uncommitted (samprity/finetune) | — |

| FT0-gpu | finetune | same as FT0, Kaggle T4 (fp16) | 026bb459116907ee | c357ab8ee361835a | 0.6474 | 0.4253 | 0.2800 | 0.4741 | 0.6105 | 0.4593 | 0.7737 | 0.5438 | 19 fits × ~7 s | uncommitted (samprity/finetune) | — |
| FT1 | finetune | ft_e5s_corrupt: FT0 + 2 in-fold corrupted copies per training row, Kaggle T4 | 026bb459116907ee | c357ab8ee361835a | 0.6063 | 0.6064 | 0.4646 | 0.5632 | 0.6383 | 0.5445 | 0.8161 | 0.5953 | 19 fits × ~12.5 s | uncommitted (samprity/finetune) | — |

Notes for FT1 (26 Nov, Kaggle): vs FT0-gpu, Task A −0.041 but Task B +0.181, Task C +0.185, all three proxies up,
composite +0.052. Caveat: the 3× rows also mean 3× optimizer steps, so "augmentation" and "longer training" are
confounded; single seed. Bangla rows 2/6 (FT0-gpu 4/6). Chosen for test inference: FT1, max_len 256 →
500,000 rows in 203 s (2,459 rows/s, peak GPU 1.3 GB). Test Task A prediction counts:
[214886, 37348, 32928, 54371, 78175, 82292], mean confidence 0.730. Contract check passed (every official id once).
Test aggregate stats (cell 4): 78.6% single messages; script 89.0% Latin / 8.7% mixed / 2.2% Bangla
(training has 0 mixed rows); 1.07% of rows > 128 tokens, 0 > 256.

Notes for FT0: single seed (fine-tuning seed variance not yet measured). Bangla-script rows 3/6 correct (TF-IDF 0/6).
Beats TF-IDF on F_A (+0.071), F_B, domain and novel proxies and calibration; **worse on Task C (−0.031) and the
robustness proxy (−0.069)** → FT1 (in-fold adversarial copies) is the next ablation. CPU inference 68 rows/s
(max_len 128) ≈ 2 h for 500k on a 16-core laptop → GPU required; GPU throughput still to be measured on Kaggle.

Notes for E001: repeated grouped CV F_A 0.557 ± 0.018. Bangla-script rows (n=6): 0/6 correct. Most errors are
scams predicted as Legitimate, and Promotional spam predicted as Legitimate (8/12).
