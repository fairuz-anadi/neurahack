# Resource declaration (copy into the notebook's Declaration cell before submitting)

Add a row the moment you start using something. Undeclared resources are prohibited (rulebook §5.6).

| Resource | Type | Used by | Source | License / provider |
|---|---|---|---|---|
| numpy, pandas, scipy, scikit-learn, PyYAML | library | all | PyPI | BSD-3 / BSD-3 / BSD-3 / BSD-3 / MIT |
| nbformat | library | notebook builder (not in final run) | PyPI | BSD-3 |
| torch, transformers, tokenizers | library | fine-tuning (Samprity) | PyPI | BSD-3 / Apache-2.0 / Apache-2.0 |
| intfloat/multilingual-e5-small | pretrained model | fine-tuning (Samprity) | https://huggingface.co/intfloat/multilingual-e5-small | MIT (checked on model card, 26 Nov) |
| Qwen/Qwen2.5-7B-Instruct | pretrained LLM (local, no API) | few-shot eval, synthetic data, test pseudo-labels (Samprity) | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | Apache-2.0 |
| vllm (optional) | library | fast local LLM inference for the teacher | PyPI | Apache-2.0 |
| BTTC: Bangla Tri-class Text Corpus (Tahmid et al.) | external dataset, 10,283 Bangla SMS/Telegram, HAM/PROMO/SPAM | external training data (FT3/FT4) | https://data.mendeley.com/datasets/5wrm959d6f/1 (DOI 10.17632/5wrm959d6f.1) | CC BY 4.0 |
| Labeled Bangla SMS Spam and Ham (Islam et al.) | external dataset, 3,999 Bangla SMS, spam/ham | external training data | https://data.mendeley.com/datasets/hhd43jbv9p/1 (DOI 10.17632/hhd43jbv9p.1) | CC BY 4.0 |
| Bangalabarta (Shahriyar & Tanbhir) | external dataset, 2,772 Bangla SMS (1,208 unique), smish/promo/normal | NOT USED by default (excluded until non-commercial licence compatibility with the prize competition is confirmed; opt-in flag --include-nc) | https://data.mendeley.com/datasets/jfkfbw3gzh/3 (DOI 10.17632/jfkfbw3gzh.3) | CC BY-NC-SA 4.0 |
| Financial scams detection dataset (Ahmed & Islam) | external dataset, 523 English+Bangla messages, scam/ham | external training data | https://data.mendeley.com/datasets/znsk27yk3h/2 (DOI 10.17632/znsk27yk3h.2) | CC BY 4.0 |
| SMS Phishing Dataset (Mishra & Soni) | external dataset, 5,971 English SMS, ham/spam/smishing (overlaps UCI SMS Spam Collection) | external training data | https://data.mendeley.com/datasets/f45bkkt8pr/1 (DOI 10.17632/f45bkkt8pr.1) | CC BY 4.0 |

- Hosted APIs: **none** (team decision; `.env` is empty).
- External labelled datasets: the four CC BY 4.0 Mendeley datasets above, downloaded by code from the original source.
  Label policy: ham/normal -> class 0 and promo -> class 5 (full labels, official 0/5 rule); smishing/scam -> Task A
  in {1,2,3,4} and generic spam -> {1..5} as PARTIAL labels (Task B/C masked). Deduplicated, capped at 300 rows per
  (source, label), at most 1,200 sampled per training fit, rows with cosine >= 0.8 to any sample.csv row removed.
  Training-only; never used for validation.
- External data snapshot: snapshots/external_v2.csv.gz (same build code, CC BY 4.0 sources only) is embedded
  in the AutoTrain notebook and used only if the Mendeley download is refused (HTTP 403 seen from Kaggle).
- Public code: none copied. (List repo/notebook URL + license if any is used.)
- Pseudo-labeling / unlabeled test text: optional teacher step (models/finetune/llm.py teacher) — Qwen2.5-7B labels a
  seeded random sample of test rows by code; e5 student trains on them. Only model predictions; declared.
- Synthetic data: models/finetune/llm.py synth — Qwen2.5-7B, greedy decoding, seeded spec grid, official class
  definitions only (no sample.csv rows in prompts).
