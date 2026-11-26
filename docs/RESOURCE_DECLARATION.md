# Resource declaration (copy into the notebook's Declaration cell before submitting)

Add a row the moment you start using something. Undeclared resources are prohibited (rulebook §5.6).

| Resource | Type | Used by | Source | License / provider |
|---|---|---|---|---|
| numpy, pandas, scipy, scikit-learn, PyYAML | library | all | PyPI | BSD-3 / BSD-3 / BSD-3 / BSD-3 / MIT |
| nbformat | library | notebook builder (not in final run) | PyPI | BSD-3 |
| torch, transformers, tokenizers | library | fine-tuning (Samprity) | PyPI | BSD-3 / Apache-2.0 / Apache-2.0 |
| intfloat/multilingual-e5-small | pretrained model | fine-tuning (Samprity) | https://huggingface.co/intfloat/multilingual-e5-small | MIT (checked on model card, 26 Nov) |

- Hosted APIs: **none** (team decision; `.env` is empty).
- External labelled datasets: **none**.
- Public code: none copied. (List repo/notebook URL + license if any is used.)
- Pseudo-labeling / unlabeled test text: **none yet**. If added, it must run in the notebook, use only model
  predictions, and be described here and in the Declaration.
- Synthetic data: **none yet**. If generated with a local model, declare the model and seed here.
