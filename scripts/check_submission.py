"""Validate a ScamShield BD submission.csv (standard library only; runs anywhere, incl. Kaggle).

    python scripts/check_submission.py submission.csv --test test.csv --sample sample_submission.csv

Checks: exact columns id,label,techniques,action,confidence (and equal to sample_submission's header);
every test id exactly once (vs test.csv and sample_submission.csv); no blank fields; label/action
integers 0-5; techniques NONE or T1..T6 joined by '|' with no repeats; confidence finite in [0,1];
labeling-guide consistency (labels 0/5 -> NONE + action 0; labels 1-4 -> >=1 technique + action != 0).
Prints label / action / technique / confidence distributions. Never prints test text, only ids and counts.
Exit code 0 = valid, 1 = invalid.
"""
import argparse
import csv
import math
import re
import sys
from collections import Counter

EXPECTED = ["id", "label", "techniques", "action", "confidence"]
TECH_RE = re.compile(r"^(NONE|T[1-6](\|T[1-6])*)$")
INT05_RE = re.compile(r"^[0-5]$")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_ids(path):
    """Header + list of ids from a CSV with an 'id' column (other columns are not kept or printed)."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        k = header.index("id")
        return header, [row[k] for row in r if row]


def check(sub_path, test_path=None, sample_path=None, quiet=False):
    errors, warns = [], []
    out = (lambda *a: None) if quiet else print

    with open(sub_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return ["submission is empty"], warns
        rows = [row for row in r]
    if header != EXPECTED:
        return [f"columns must be exactly {EXPECTED}, got {header}"], warns

    bad = Counter(); ex = {}
    def flag(kind, rid):
        bad[kind] += 1
        ex.setdefault(kind, []).append(rid) if len(ex.get(kind, [])) < 3 else None

    ids, labels, actions, techs, confs = [], [], [], [], []
    for n, row in enumerate(rows, 2):
        if len(row) != 5:
            flag("wrong number of fields", f"line {n}"); continue
        rid, lab, tech, act, conf = row
        ids.append(rid)
        if any(v.strip() == "" for v in row): flag("blank field", rid)
        if not INT05_RE.match(lab): flag("label not an int 0-5", rid)
        if not INT05_RE.match(act): flag("action not an int 0-5", rid)
        if not TECH_RE.match(tech): flag("malformed techniques", rid)
        elif len(tech.split("|")) != len(set(tech.split("|"))): flag("repeated technique code", rid)
        try:
            c = float(conf)
            if not math.isfinite(c) or not 0 <= c <= 1: flag("confidence outside [0,1]", rid)
        except ValueError:
            c = float("nan"); flag("confidence not a number", rid)
        if INT05_RE.match(lab) and INT05_RE.match(act):
            L, A = int(lab), int(act)
            if L in (0, 5) and (tech != "NONE" or A != 0): flag("label 0/5 without NONE + action 0", rid)
            if L in (1, 2, 3, 4) and (tech == "NONE" or A == 0): flag("label 1-4 with NONE or action 0", rid)
            labels.append(L); actions.append(A)
        techs.append(tech); confs.append(c)

    for k, v in bad.items():
        errors.append(f"{v} rows: {k} (e.g. {', '.join(ex[k])})")

    dup = len(ids) - len(set(ids))
    if dup: errors.append(f"{dup} duplicate ids")

    for name, path in (("test.csv", test_path), ("sample_submission.csv", sample_path)):
        if not path:
            warns.append(f"{name} not given: id coverage not checked against it"); continue
        h, ref = read_ids(path)
        if name == "sample_submission.csv":
            if h != EXPECTED: errors.append(f"sample_submission columns {h} differ from {EXPECTED}")
            elif ids != ref and set(ids) == set(ref): warns.append("row order differs from sample_submission (allowed, ids match)")
        s, t = set(ids), set(ref)
        if len(ids) != len(ref): errors.append(f"row count {len(ids)} != {name} rows {len(ref)}")
        if t - s: errors.append(f"{len(t - s)} {name} ids missing from submission")
        if s - t: errors.append(f"{len(s - t)} submission ids not in {name}")

    # ---- distributions (aggregate only) ----
    N = max(len(rows), 1)
    share = lambda cnt: {k: round(v / N, 4) for k, v in sorted(cnt.items())}
    out(f"rows: {len(rows)}")
    out("label share:  ", share(Counter(labels)))
    out("action share: ", share(Counter(actions)))
    codes = Counter(c for t in techs if t != "NONE" for c in t.split("|"))
    out("NONE share:   ", round(sum(t == "NONE" for t in techs) / N, 4))
    out("technique share (rows containing code):", share(codes))
    out("codes per row:", share(Counter(0 if t == "NONE" else len(t.split("|")) for t in techs)))
    fc = [c for c in confs if math.isfinite(c)]
    if fc:
        hist = Counter(min(int(c * 10), 9) for c in fc)
        out(f"confidence mean {sum(fc)/len(fc):.4f} | min {min(fc):.4f} | max {max(fc):.4f}")
        out("confidence deciles:", {f"{k/10:.1f}": round(v / len(fc), 3) for k, v in sorted(hist.items())})
    return errors, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--test", help="test.csv (only the id column is read)")
    ap.add_argument("--sample", help="sample_submission.csv")
    a = ap.parse_args()
    errors, warns = check(a.submission, a.test, a.sample)
    for w in warns: print("WARN:", w)
    if errors:
        print("INVALID:"); [print(" -", e) for e in errors]; sys.exit(1)
    print("VALID submission")


if __name__ == "__main__":
    main()
