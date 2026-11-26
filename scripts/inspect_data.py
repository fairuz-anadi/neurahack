"""EDA for the labeled sample + aggregate-only statistics for test.csv (if present).

    python scripts/inspect_data.py            -> outputs/eda_report.md, outputs/eda.json

Test data rule: this script never prints or saves an individual test record. Only counts,
shares and quantiles are computed for test.csv.
"""
import _bootstrap  # noqa: F401

import json
import re

import numpy as np
import pandas as pd

from common.data_loader import load_config, load_sample_submission, load_test, load_train, output_dirs, test_available
from common.folds import build_groups, detect_brands, similarity_matrix, template_key
from common.labels import TASK_A_NAMES, TASK_C_NAMES, TECHNIQUES
from common.preprocessing import is_junk_token, is_noise_only, script_profile


def noise_stats(raw: pd.Series) -> dict:
    prof = raw.map(script_profile)
    junk = raw.map(lambda t: any(is_junk_token(w) for w in re.findall(r"[A-Za-z]+", t)))
    return {
        "homoglyph_rows": int(prof.map(lambda p: p["other_letters"] > 0).sum()),
        "invisible_char_rows": int(prof.map(lambda p: p["invisible_chars"] > 0).sum()),
        "rows_with_junk_token": int(junk.sum()),
        "letter_repeat_rows": int(raw.map(lambda t: bool(re.search(r"([A-Za-z])\1\1", t))).sum()),
        "emoji_rows": int(raw.map(lambda t: any(ord(c) > 0x2600 and not 0x0980 <= ord(c) <= 0x09FF for c in t)).sum()),
        "noise_only_rows": int(raw.map(is_noise_only).sum()),
        "url_rows": int(prof.map(lambda p: p["has_url"]).sum()),
    }


def length_stats(s: pd.Series) -> dict:
    q = s.quantile([0, 0.25, 0.5, 0.75, 0.95, 1]).round(1)
    return {f"q{int(k * 100)}": float(v) for k, v in q.items()} | {"mean": round(float(s.mean()), 1)}


def main():
    cfg = load_config()
    _, out_dir = output_dirs(cfg)
    tr = load_train(cfg)
    rep = {"train_file": tr.attrs["source"], "train_fingerprint": tr.attrs["fingerprint"],
           "text_column": tr.attrs["text_col"], "n_train": len(tr)}

    rep["task_a_counts"] = {f"{k} {TASK_A_NAMES[k]}": int(v) for k, v in tr.label.value_counts().sort_index().items()}
    rep["task_c_counts"] = {f"{k} {TASK_C_NAMES[k]}": int(v) for k, v in tr.action.value_counts().sort_index().items()}
    rep["technique_counts"] = {t: int(tr[t].sum()) for t in TECHNIQUES}
    rep["techniques_none_rows"] = int((tr.techniques == "NONE").sum())
    rep["techniques_per_row"] = tr[TECHNIQUES].sum(axis=1).value_counts().sort_index().to_dict()
    rep["class_x_technique"] = tr.groupby("label")[TECHNIQUES].sum().to_dict(orient="index")
    rep["class_x_action"] = pd.crosstab(tr.label, tr.action).to_dict(orient="index")
    rep["nonfraud_consistency"] = {
        "class0_none_share": float((tr[tr.label == 0].techniques == "NONE").mean()),
        "class5_none_share": float((tr[tr.label == 5].techniques == "NONE").mean()),
        "class0_action0_share": float((tr[tr.label == 0].action == 0).mean()),
        "class5_action0_share": float((tr[tr.label == 5].action == 0).mean()),
        "fraud_rows_with_no_technique": int(((~tr.label.isin([0, 5])) & (tr.techniques == "NONE")).sum()),
    }
    rep["imbalance_ratio_A"] = round(tr.label.value_counts().max() / tr.label.value_counts().min(), 2)
    rep["script_counts"] = tr.script.value_counts().to_dict()
    rep["script_x_class"] = pd.crosstab(tr.script, tr.label).to_dict(orient="index")
    rep["turns"] = {"conversation_rows": int(tr.is_conversation.sum()),
                    "n_turns_counts": tr.n_turns.value_counts().sort_index().to_dict(),
                    "speaker_patterns": tr.speakers[tr.is_conversation].value_counts().head(10).to_dict()}
    rep["char_length"] = length_stats(tr.raw_text.str.len())
    rep["word_length"] = length_stats(tr.raw_text.str.split().map(len))
    rep["missing_text"] = int((tr.raw_text.str.strip() == "").sum())
    rep["exact_duplicate_texts"] = int(tr.raw_text.duplicated().sum())
    rep["duplicate_normalized_texts"] = int(tr.normalized_text.duplicated().sum())
    S = similarity_matrix(tr.normalized_text.tolist())
    rep["near_dup_pairs"] = {f">={t}": int((np.triu(S, 1) >= t).sum()) for t in (0.5, 0.6, 0.7, 0.8, 0.9)}
    rep["template_key_duplicates"] = int(tr.raw_text.map(template_key).duplicated().sum())
    for t in (0.8, 0.6, 0.45, 0.3):
        g = build_groups(tr.normalized_text.tolist(), t, S)
        rep[f"groups_at_{t}"] = {"n_groups": int(g.max() + 1), "multi_row_groups": int((np.bincount(g) > 1).sum()),
                                 "largest_group": int(np.bincount(g).max())}
    up = np.triu(S, 1)[np.triu_indices(len(S), 1)]
    rep["max_pair_cosine"] = round(float(up.max()), 3)
    rep["noise_train"] = noise_stats(tr.raw_text)
    brands = tr.raw_text.map(detect_brands)
    rep["brand_mentions_train"] = pd.Series([b for bs in brands for b in bs]).value_counts().to_dict()

    ss = load_sample_submission(cfg)
    rep["sample_submission"] = {"rows": len(ss), "columns": list(ss.columns), "id_min": int(ss.id.min()),
                                "id_max": int(ss.id.max()), "fingerprint": ss.attrs["fingerprint"]}

    if test_available(cfg):
        te = load_test(cfg)
        rep["test_aggregate"] = {
            "n_test": len(te), "text_column": te.attrs["text_col"], "extra_columns": te.attrs["extra_columns"],
            "ids_match_sample_submission": set(te.id.tolist()) == set(ss.id.tolist()),
            "script_share": te.script.value_counts(normalize=True).round(4).to_dict(),
            "conversation_share": round(float(te.is_conversation.mean()), 4),
            "n_turns_counts": te.n_turns.value_counts().sort_index().to_dict(),
            "char_length": length_stats(te.raw_text.str.len()),
            "empty_text_rows": int((te.raw_text.str.strip() == "").sum()),
            "noise": noise_stats(te.raw_text),
        }
    else:
        rep["test_aggregate"] = "test.csv not available in this environment (expected locally)."

    with open(out_dir / "eda.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    write_md(rep, out_dir / "eda_report.md")
    print(f"wrote {out_dir / 'eda_report.md'} and eda.json")


def write_md(r: dict, path):
    L = ["# EDA report (generated by scripts/inspect_data.py)", "",
         f"Train: {r['n_train']} labeled rows, text column `{r['text_column']}`, fingerprint `{r['train_fingerprint']}`.",
         f"sample_submission: {r['sample_submission']['rows']} rows, ids {r['sample_submission']['id_min']}..{r['sample_submission']['id_max']}.", "",
         "## Labels", f"- Task A: {r['task_a_counts']} (max/min ratio {r['imbalance_ratio_A']})",
         f"- Task C: {r['task_c_counts']}", f"- Techniques: {r['technique_counts']}; NONE rows: {r['techniques_none_rows']}",
         f"- Techniques per row: {r['techniques_per_row']}", f"- Non-fraud consistency: {r['nonfraud_consistency']}", "",
         "## Class x technique", "```", json.dumps(r["class_x_technique"]), "```",
         "## Class x action", "```", json.dumps(r["class_x_action"]), "```", "",
         "## Text", f"- Script (Latin = English OR Banglish; script alone cannot separate them): {r['script_counts']}",
         f"- Conversations: {r['turns']}", f"- Char length: {r['char_length']}", f"- Word length: {r['word_length']}",
         f"- Missing text: {r['missing_text']}; exact dup texts: {r['exact_duplicate_texts']}; template-key dups: {r['template_key_duplicates']}",
         f"- Near-dup pairs (char TF-IDF cosine): {r['near_dup_pairs']}",
         f"- Grouping: 0.8 -> {r['groups_at_0.8']}; 0.6 -> {r['groups_at_0.6']}; 0.45 -> {r['groups_at_0.45']}; 0.3 -> {r['groups_at_0.3']}",
         f"- Noise/corruption: {r['noise_train']}", f"- Brand mentions: {r['brand_mentions_train']}", "",
         "## Test (aggregate only)", f"{r['test_aggregate']}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
