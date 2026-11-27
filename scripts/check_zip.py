"""Validate the final package NeuraHack_Preli_<TeamName>.zip (standard library only).

    python scripts/check_zip.py NeuraHack_Preli_Team_Hampton.zip --team Team_Hampton \
        --test test.csv --sample sample_submission.csv

Checks:
  * file name NeuraHack_Preli_<TeamName>.zip (and matches --team if given)
  * EXACTLY 4 top-level entries: submission.csv, notebook.ipynb, README.md, .env (no folders, no extras)
  * .env is empty or only blank / comment lines
  * submission.csv passes check_submission.py (ids vs --test / --sample when given)
  * notebook.ipynb: first cell is the markdown Declaration with all 5 parts, no placeholders;
    no tokens / secrets / git / curl / wget in sources OR saved outputs; unpinned pip = WARN;
    no huge outputs; with --test: no saved output contains test text (reports counts only, never text);
    the printed 'test label share' in the outputs matches the csv (same Kaggle version)
  * README.md has no placeholders and names the team
Exit code 0 = OK (warnings allowed), 1 = problems.
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_submission import check as check_submission  # noqa: E402

EXPECTED = {"submission.csv", "notebook.ipynb", "README.md", ".env"}
NAME_RE = re.compile(r"^NeuraHack_Preli_([A-Za-z0-9_]+)\.zip$")
SECRET_PATTERNS = {
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    "Hugging Face token": r"hf_[A-Za-z0-9]{20,}",
    "OpenAI/Anthropic-style key": r"sk-(ant-)?[A-Za-z0-9_-]{20,}",
    "Google API key": r"AIza[0-9A-Za-z_-]{30,}",
    "Kaggle secrets": r"UserSecretsClient|kaggle_secrets|get_secret\(",
    "GH_TOKEN": r"GH_TOKEN",
    "api_key assignment": r"api[_-]?key\s*[=:]",
    "credential in URL": r"https?://[^/\s'\"]+@",
    "Authorization header": r"Authorization:\s*(token|bearer)",
}
SIDE_EFFECT_PATTERNS = {"git command": r"(^|[!\s;&])git\s+(clone|push|pull|commit|config|checkout)\b",
                        "curl/wget": r"(^|[!\s;&])(curl|wget)\s", "requests/urllib": r"\b(import requests|requests\.(get|post)|urllib\.request)"}
PIP_RE = re.compile(r"pip['\"]?,?\s*['\"]?install[^\n]*")
PLACEHOLDERS = r"<version>|<X>|<…>|<\.\.\.>|\[from run\]|<Team ?Name>|<v>"
DECL_PARTS = {"1 libraries": r"librar", "2 external resources": r"external|resources|license",
              "3 method + validation": r"method|validation", "4 seeds + reproduction": r"seed",
              "5 API steps/costs": r"\bAPI"}
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def output_text(cell):
    parts = []
    for o in cell.get("outputs", []):
        parts.append("".join(o.get("text", [])) if isinstance(o.get("text"), list) else str(o.get("text", "")))
        for k, v in (o.get("data") or {}).items():
            if k.startswith("text/") or k == "application/json":
                parts.append("".join(v) if isinstance(v, list) else str(v))
        if o.get("output_type") == "error":
            parts.append("\n".join(o.get("traceback", [])))
    return "\n".join(parts)


def test_prefixes(test_path, k=40):
    """Set of the first k chars of every test text with >= k chars (kept in memory only, never printed)."""
    P = set()
    with open(test_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f); h = next(r); ti = h.index("text")
        for row in r:
            if len(row) > ti and len(row[ti].strip()) >= k:
                P.add(row[ti].strip()[:k])
    return P


def leaks(text, P, k=40):
    n = 0
    for line in text.splitlines():
        starts = [0] + [m.end() for m in re.finditer(r"[\s'\"(\[,:|>]", line)]
        n += sum(line[s:s + k] in P for s in starts if len(line) - s >= k)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--team", help="registered team name, e.g. Team_Hampton")
    ap.add_argument("--test", help="test.csv: enables id coverage + test-text-in-outputs scan")
    ap.add_argument("--sample", help="sample_submission.csv")
    a = ap.parse_args()
    E, W = [], []

    m = NAME_RE.match(os.path.basename(a.zip))
    if not m: E.append(f"zip name must be NeuraHack_Preli_<TeamName>.zip, got {os.path.basename(a.zip)}")
    elif a.team and m.group(1) != re.sub(r"[^A-Za-z0-9_]+", "_", a.team.strip()).strip("_"):
        E.append(f"zip team name {m.group(1)!r} != --team {a.team!r}")

    z = zipfile.ZipFile(a.zip)
    names = z.namelist()
    if set(names) != EXPECTED or len(names) != 4:
        E.append(f"zip must contain exactly {sorted(EXPECTED)} at top level, got {names}")
    have = lambda n: n in names

    # ---- .env ----
    if have(".env"):
        env = z.read(".env").decode("utf-8", "replace")
        real = [l for l in env.splitlines() if l.strip() and not l.strip().startswith("#")]
        if real: E.append(f".env must be empty or comments only; it has {len(real)} non-comment line(s)")
        print(f".env: {len(env.encode())} bytes, {len(real)} non-comment lines")

    # ---- submission.csv ----
    sub_share = None
    if have("submission.csv"):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "submission.csv")
            open(p, "wb").write(z.read("submission.csv"))
            print("---- submission.csv ----")
            errs, warns = check_submission(p, a.test, a.sample)
            E += [f"submission.csv: {e}" for e in errs]; W += [f"submission.csv: {w}" for w in warns]
            with open(p, newline="", encoding="utf-8") as f:
                labs = [row[1] for row in csv.reader(f)][1:]
            c = Counter(labs); sub_share = {int(k): round(v / max(len(labs), 1), 3) for k, v in c.items() if k.isdigit()}

    # ---- notebook.ipynb ----
    if have("notebook.ipynb"):
        print("---- notebook.ipynb ----")
        try:
            nb = json.loads(z.read("notebook.ipynb").decode("utf-8"))
        except Exception as ex:
            E.append(f"notebook.ipynb is not valid JSON: {ex}"); nb = {"cells": []}
        cells = nb.get("cells", [])
        if not cells or cells[0].get("cell_type") != "markdown":
            E.append("notebook: first cell must be markdown (the Declaration)")
        else:
            d = "".join(cells[0]["source"])
            if "declaration" not in d.lower(): E.append("notebook: first cell does not say 'Declaration'")
            for part, pat in DECL_PARTS.items():
                if not re.search(pat, d, re.I): E.append(f"notebook declaration: part {part} not found")
            ph = sorted(set(re.findall(PLACEHOLDERS, d)))
            if ph: E.append(f"notebook declaration still has placeholders: {ph}")
        P = test_prefixes(a.test) if a.test else None
        total_out, printed_share = 0, None
        for i, c in enumerate(cells, 1):
            s = "".join(c.get("source", [])); o = output_text(c) if c.get("cell_type") == "code" else ""
            total_out += len(o)
            for where, txt in (("source", s), ("output", o)):
                for k, pat in SECRET_PATTERNS.items():
                    if re.search(pat, txt, re.I if k == "api_key assignment" else 0):
                        E.append(f"notebook cell {i} {where}: {k}")
                if c.get("cell_type") == "code":
                    for k, pat in SIDE_EFFECT_PATTERNS.items():
                        if re.search(pat, txt, re.M): (E if where == "source" else W).append(f"notebook cell {i} {where}: {k}")
            if c.get("cell_type") == "code":
                for pm in PIP_RE.findall(s):
                    if "==" not in pm: W.append(f"notebook cell {i}: unpinned pip install")
            if len(o) > 200_000: W.append(f"notebook cell {i}: large saved output ({len(o)//1000} KB)")
            if P is not None and o:
                n = leaks(o, P)
                if n: E.append(f"notebook cell {i} output: {n} line(s) contain test.csv text (not shown)")
            mm = re.findall(r"test label share:\s*(\{[^}]*\})", o)
            if mm: printed_share = mm[-1]
        print(f"cells: {len(cells)} | saved output: {total_out // 1000} KB"
              + ("" if P is not None else " | test-text scan skipped (no --test)"))
        if not any(c.get("outputs") for c in cells):
            W.append("notebook has no saved outputs: use the notebook downloaded from the same Kaggle version as the csv")
        if sub_share is not None:
            if printed_share is None:
                W.append("no 'test label share' line in notebook outputs: cannot confirm csv and notebook come from the same run")
            else:
                plain = re.sub(r"np\.\w+\(([^)]*)\)", r"\1", printed_share)          # numpy 2 reprs
                nb_share = {int(k): float(v) for k, v in re.findall(r"(\d+):\s*([0-9.]+)", plain)}
                diff = max(abs(nb_share.get(k, 0) - sub_share.get(k, 0)) for k in range(6))
                if diff > 0.0015: E.append(f"csv label share {sub_share} != notebook's printed {nb_share}: csv and notebook are from different runs")
                else: print("csv label share matches the notebook's final output")

    # ---- README.md ----
    if have("README.md"):
        rd = z.read("README.md").decode("utf-8", "replace")
        ph = sorted(set(re.findall(PLACEHOLDERS, rd)))
        if ph: E.append(f"README.md still has placeholders: {ph}")
        if a.team and a.team not in rd: W.append(f"README.md does not mention team {a.team!r}")
        for k, pat in SECRET_PATTERNS.items():
            if re.search(pat, rd, re.I if k == "api_key assignment" else 0): E.append(f"README.md: {k}")

    print("----")
    for w in W: print("WARN:", w)
    if E:
        print("PROBLEMS:"); [print(" -", e) for e in E]; sys.exit(1)
    print(f"OK: {os.path.basename(a.zip)} is ready to upload")


if __name__ == "__main__":
    main()
