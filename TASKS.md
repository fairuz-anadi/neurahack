# Final-day tasks (Fri 27 Nov, BST)

Deadlines: **Kaggle closes 18:00** (select 2 finals before then). **ZIP due 22:00** through the Google Form (Anadi uploads).

## Running on Kaggle

| Run | Notebook | Expected finish |
|---|---|---|
| `neurahack-duo` | `notebooks/12_duo.ipynb` (TF-IDF + XLM-R large + XLM-R base) | ~16:35 |
| `neurahack-final-team-v3-large-ft` | `notebooks/11_v3_large_ft.ipynb` (v3-large + FT family) | ~16:55 |

Current best public score: 0.91567 (large run, `notebooks/08_large.ipynb`).

## Rules to keep in mind (Rulebook v1.3)
- No training on test text in any form: no pseudo-labels, self-training or domain adaptation.
- No public code outside Kaggle: **do not make this repo public again**, and don't share code anywhere else.
- Declarations must be true. Every model, dataset and LLM used must be listed with source and licence.
- Don't commit data, `.env`, tokens or checkpoints.

## Rabib
- [ ] **By 17:15:** when a run finishes, read its log and send Anadi the "families / weights / CV proxy / chosen" lines from the last cell.
- [ ] **By 19:00:** fill `submission/README.md` from the chosen run:
  - section 5 results table (TF-IDF, XLM-R, final);
  - robustness and segment numbers;
  - runtime table;
  - python and transformers versions (printed in the setup cell);
  - remove every `[from run]`.
- [ ] **By 19:00:** update the Review cell numbers in the chosen notebook. Use `scripts/patch_markdown.py`, which changes markdown only.
- [ ] **By 21:00:** once Anadi builds the ZIP, run `python scripts/check_zip.py <zip> --team Team_Hampton --test <test.csv>` and report any error.

## Samprity
- [ ] **By 18:30:** in `submission/README.md` section 6, add the FT resources:
  - `intfloat/multilingual-e5-small` (MIT);
  - `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, run locally, no API);
  - the 4 Mendeley CC BY 4.0 datasets, with DOIs and attribution.

  Replace "No external datasets" (section 6 and the checklist in section 8) with an accurate line, for the case where the FT run is chosen. If it isn't chosen, list FT as "tested, not in the final".
- [ ] **By 18:30:** write one paragraph for the README describing how the 625 Qwen rows were made (prompt inputs = official class definitions only, no sample/test rows) and the near-duplicate filter.
- [ ] Keep the teacher / pseudo-label code (`models/finetune/llm.py` teacher part, `synthetic.load_pseudo`) out of everything we submit. It is banned by v1.3.

## Anadi
- [ ] Make the GitHub repo **private** (Settings → Change visibility).
- [ ] ~16:40: check both runs, upload the best one, select 2 finals by CV + public score before 18:00.
- [ ] Build the ZIP with `scripts/build_zip.py` from the chosen run's CSV and notebook; upload it before 22:00.
