# Final submission from the LLM family: label / techniques / action from the LLM; rows the LLM missed fall back to a
# model submission. Confidence: LLM accuracy on sample.csv when LLM and model agree, lower when they disagree.
#   python llm_combine.py LLM_TEST_CSV FALLBACK_SUB_CSV OUT_CSV [acc_agree] [acc_disagree]
import sys
import numpy as np, pandas as pd

D = "D:/NSU/"
llm_csv, fb_csv, out = sys.argv[1:4]
c_agree = float(sys.argv[4]) if len(sys.argv) > 4 else 0.97
c_dis = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
test = pd.read_csv(D + "test.csv", usecols=["id"])
L = pd.read_csv(llm_csv, keep_default_na=False).set_index("id").loc[test["id"]].reset_index()
F = pd.read_csv(fb_csv, keep_default_na=False).set_index("id").loc[test["id"]].reset_index()
ok = L["llm_label"].astype(int).between(0, 5).values
fraud = L["llm_label"].astype(int).isin([1, 2, 3, 4]).values
tech = np.where(fraud, L["llm_tech"].replace("", "NONE"), "NONE")
tech = np.where(fraud & (tech == "NONE"), "T5", tech)          # a fraud row needs at least one technique
act = np.where(fraud, L["llm_action"].astype(int), 0)
sub = pd.DataFrame({"id": test["id"],
                    "label": np.where(ok, L["llm_label"].astype(int), F["label"].astype(int)),
                    "techniques": np.where(ok, tech, F["techniques"]),
                    "action": np.where(ok, act, F["action"].astype(int))})
agree = sub["label"].values == F["label"].astype(int).values
sub["confidence"] = np.where(~ok, F["confidence"].astype(float), np.where(agree, c_agree, c_dis)).round(4)
cols = list(pd.read_csv(D + "sample_submission.csv", nrows=1).columns)
sub = sub[cols]
assert len(sub) == len(test) and sub["id"].is_unique and not sub.isna().any().any()
assert sub["techniques"].str.fullmatch(r"NONE|T[1-6](\|T[1-6])*").all()
sub.to_csv(out, index=False)
print(f"wrote {out} | LLM rows {ok.mean():.4f} | agree with fallback {agree.mean():.4f}")
print("label mix:", sub["label"].value_counts(normalize=True).sort_index().round(3).to_dict())
