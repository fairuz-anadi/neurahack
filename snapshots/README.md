# External data snapshot

`external_v2.csv.gz` = output of `python -m models.finetune.external --cap 300` (CC BY 4.0 sources only, Bangalabarta
excluded, deduplicated, rows near sample.csv removed), built on 26 Nov 2026 from the original Mendeley Data files:

| source column | dataset | DOI | licence |
|---|---|---|---|
| bttc | BTTC: Bangla Tri-class Text Corpus (Tahmid, Abedin, Ahmed, Chowdhury, Sultana) | 10.17632/5wrm959d6f.1 | CC BY 4.0 |
| bangla_spam_ham | Labeled Bangla SMS Spam and Ham Dataset (Islam et al.) | 10.17632/hhd43jbv9p.1 | CC BY 4.0 |
| financial_scams | Financial scams detection dataset (Ahmed, Islam) | 10.17632/znsk27yk3h.2 | CC BY 4.0 |
| sms_phishing_mishra | SMS Phishing Dataset for ML and Pattern Recognition (Mishra, Soni) | 10.17632/f45bkkt8pr.1 | CC BY 4.0 |

Why a snapshot: Mendeley's download endpoint answered HTTP 403 from Kaggle. The AutoTrain notebook first tries the
original download; if it is refused, it uses this embedded snapshot (same code, same sources, redistributed under
CC BY 4.0 with attribution). Training-only data; never used for validation.
