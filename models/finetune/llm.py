"""Local instruction LLM for (1) a held-out few-shot evaluation and (2) synthetic training data.

    python -m models.finetune.llm eval  [--model Qwen/Qwen2.5-7B-Instruct] [--shots 2]
    python -m models.finetune.llm synth --n 900 --out /kaggle/working/synth_v1.csv

Model: Qwen/Qwen2.5-7B-Instruct (Apache-2.0), fp16, split over the two Kaggle T4s with device_map="auto".
Greedy decoding everywhere (reproducible). Diversity in synthetic data comes from a seeded grid of specs
(class x language x format x scenario hint), not from sampling.

Leakage rules:
  eval  - few-shot examples for fold k are drawn only from folds != k; a row never appears in its own prompt.
  synth - prompts contain ONLY the official class definitions and a seeded spec. No sample.csv rows (and none
          of the Data-tab illustrative examples, which duplicate sample.csv rows). So one synthetic set can be
          added to every training fold without leaking validation rows.
Never run on test.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.labels import NONE_TOKEN, TECHNIQUES, LabelError, parse_techniques

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LABEL_GUIDE = """You label SMS / chat messages from Bangladesh (English, Bangla script, Banglish, mixed).
Labels follow the intent of the WHOLE message or conversation, not keywords.

Task A label (if several apply, use this priority and stop at the first match: 1, 2, 4, 3, 5, 0):
1 MFS/OTP scam: asks for PIN, OTP, password, NID, card details or other credentials; claims a mistaken transfer and asks for money back; or impersonates support to get account access. A genuine OTP delivery that warns not to share the code is 0.
2 Prize/lottery scam: claims an unearned prize, gift, cashback or reward and requires payment, clicking, sharing details or another action to claim it.
4 Delivery/parcel phishing: deceptive or unexpected parcel/delivery claim used to get payment, credentials, address confirmation or a link click.
3 Fake job/investment: unrealistic income, upfront recruitment fee, task-based earning, money doubling, crypto/investment schemes.
5 Promotional spam: marketing, sales, discounts, product or app promotion that is spam-like but NOT fraudulent.
0 Legitimate: genuine transactional, informational, personal or service message (incl. real OTP notices, payment confirmations, delivery updates, bill reminders, and people warning others about scams).
Priority examples: a prize message asking for OTP/PIN/NID is 1; a "gift" parcel demanding customs payment is 2; a job offer asking for an OTP is 1.

Task B techniques (all that apply, joined by |; NONE for classes 0 and 5):
T1 authority/organization impersonation (a bank, MFS, courier, government, telecom, employer... is claimed)
T2 credential/OTP/PIN/sensitive-information harvesting
T3 advance-fee or payment deception
T4 prize/reward/lottery/cashback baiting
T5 urgency, threat, pressure, trust-building or other social engineering
T6 task-based earning, fake job, investment or financial manipulation

Task C action (the most consequential thing the sender wants; 0 for classes 0 and 5):
0 none / informational, 1 share credentials or confidential info, 2 transfer money or pay a fee,
3 click/open a link, 4 install an app, join a group or complete a task, 5 contact/call/reply to someone.
An explicit request to pay is 2; only opening a link is 3; handing over credentials is 1 even via a link.
Conversations use ' ||| ' between messages and may carry speaker markers such as 'S:' / 'R:'."""

CLASS_NAMES = {0: "Legitimate", 1: "MFS/OTP scam", 2: "Prize/lottery scam", 3: "Fake job/investment",
               4: "Delivery/parcel phishing", 5: "Promotional spam"}


# ------------------------------------------------------------------ model
def load_llm(model_name: str = DEFAULT_MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tok, model


def _first_digit_prob(step_tokens) -> float:
    """step_tokens: list of (decoded_token, prob). Probability of the first generated 0-5 digit (= Task A label)."""
    for txt, p in step_tokens:
        if txt.strip() in {"0", "1", "2", "3", "4", "5"}:
            return float(p)
    return 0.0


def make_generate_fn(tok, model, max_new_tokens: int, batch_size: int = 8, with_conf: bool = False):
    """Returns gen(list_of_message_lists) -> list[str] (or (texts, label_confidences) with with_conf).
    Greedy, batched, length-sorted."""
    import torch

    if with_conf:
        def gen_conf(conversations):
            prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
            order = np.argsort([len(p) for p in prompts])
            out, conf = [None] * len(prompts), [0.0] * len(prompts)
            for i in range(0, len(prompts), batch_size):
                idx = order[i:i + batch_size]
                enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
                new = g.sequences[:, enc["input_ids"].shape[1]:]
                probs = [torch.softmax(sc.float(), -1) for sc in g.scores]
                for r, j in enumerate(idx):
                    out[j] = tok.decode(new[r], skip_special_tokens=True)
                    steps = [(tok.decode([int(new[r, t])]), float(probs[t][r, new[r, t]])) for t in range(new.shape[1])]
                    conf[j] = _first_digit_prob(steps)
            return out, conf
        return gen_conf

    def gen(conversations):
        prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
        order = np.argsort([len(p) for p in prompts])
        out = [None] * len(prompts)
        for i in range(0, len(prompts), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j, row in zip(idx, ids[:, enc["input_ids"].shape[1]:]):
                out[j] = tok.decode(row, skip_special_tokens=True)
        return out
    return gen


def parse_json(text: str):
    """First JSON object or array in the text, or None."""
    if text is None:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        s = t.find(open_c)
        e = t.rfind(close_c)
        if s >= 0 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


def clean_record(obj: dict, want_label: int | None = None) -> dict | None:
    """Validate one {text,label,techniques,action} record and apply the official 0/5 convention."""
    try:
        text = str(obj["text"]).strip()
        label = int(obj["label"])
        action = int(obj["action"])
        tech = obj.get("techniques", NONE_TOKEN)
        if isinstance(tech, list):
            tech = "|".join(str(x) for x in tech) or NONE_TOKEN
        codes = parse_techniques(str(tech).replace(",", "|").replace(" ", ""))
    except (KeyError, TypeError, ValueError, LabelError):
        return None
    if not text or len(text) < 8 or label not in range(6) or action not in range(6):
        return None
    if want_label is not None and label != want_label:
        return None
    if label in (0, 5):
        codes, action = [], 0
    elif not codes or action == 0:
        return None  # fraud rows need at least one technique and a suspicious action
    codes = [c for c in TECHNIQUES if c in codes]
    return {"text": text, "label": label, "techniques": "|".join(codes) if codes else NONE_TOKEN, "action": action}


# ------------------------------------------------------------------ eval
def classify_messages(text: str, shots: pd.DataFrame) -> list[dict]:
    ex = "\n".join(json.dumps({"text": r.raw_text, "label": int(r.label), "techniques": r.techniques,
                               "action": int(r.action)}, ensure_ascii=False) for r in shots.itertuples())
    return [{"role": "system", "content": LABEL_GUIDE},
            {"role": "user", "content": f"Labeled examples:\n{ex}\n\nLabel this record. Reply with ONLY one JSON "
                                        f'object: {{"label": int, "techniques": "T1|T2" or "NONE", "action": int}}\n\n'
                                        f"Record: {text}"}]


def pick_shots(pool: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in pool.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def llm_eval(train: pd.DataFrame, fold: np.ndarray, gen, per_class: int = 2, seed: int = 42) -> dict:
    from common.data_loader import get_targets
    from common.labels import techniques_to_binary
    from common.metrics import evaluate

    preds = [None] * len(train)
    for k in sorted(np.unique(fold)):
        shots = pick_shots(train[fold != k], per_class, seed + int(k))     # never contains fold-k rows
        va = np.where(fold == k)[0]
        assert not set(shots.id) & set(train.id.iloc[va]), "validation row leaked into its own prompt"
        outs = gen([classify_messages(train.raw_text.iloc[i], shots) for i in va])
        for i, o in zip(va, outs):
            j = parse_json(o)
            preds[i] = j if isinstance(j, dict) else None
    def _int(p, key):
        try:
            return int(p.get(key, 0)) if p else 0
        except (TypeError, ValueError):
            return 0
    n_bad = sum(p is None for p in preds)
    lab = np.array([_int(p, "label") for p in preds]).clip(0, 5)
    act = np.array([_int(p, "action") for p in preds]).clip(0, 5)
    tech = []
    for p in preds:
        try:
            tech.append("|".join(parse_techniques(str(p.get("techniques", "NONE"))) or ["NONE"]) if p else "NONE")
        except LabelError:
            tech.append("NONE")
    onehot = lambda y: np.eye(6)[y] * 0.9 + 0.1 / 6
    yA, yB, yC = get_targets(train)
    m = evaluate(yA, yB, yC, onehot(lab), techniques_to_binary(tech).astype(float), onehot(act))
    m["unparseable"] = n_bad
    return m


# ------------------------------------------------------------------ synthetic
LANGS = ["English", "Bangla script (বাংলা)", "Banglish (Bangla written in Latin letters)",
         "mixed: Bangla script and English words in the same message"]
FORMATS = ["a single short SMS", "a single SMS with informal spelling, abbreviations and a few typos",
           "a chat conversation of 3-6 messages separated by ' ||| ' with speaker markers like 'S:' and 'R:'",
           "a chat conversation of 5-10 messages separated by ' ||| ' where the intent only becomes clear late",
           "a longer formal message (email, notice, or call transcript)",
           "a group-chat excerpt separated by ' ||| ' with names like '[Group] Rafi:'"]
SCENARIOS = {
    0: ["genuine OTP delivery that warns not to share the code", "genuine MFS cash-in/cash-out/send-money confirmation",
        "genuine courier delivery or pickup update", "utility or telecom bill reminder from the real provider",
        "a friend warning someone about a scam message they received", "personal chat between family or friends about money or plans",
        "bank transaction alert", "office, university or doctor appointment notice", "real refund credited notice",
        "customer support reply that tells the user never to share PIN", "noise-only text with no meaning (random letters)"],
    1: ["fake MFS agent claims money was sent by mistake and asks for it back", "fake bank/MFS support asks for PIN or OTP to unblock account",
        "KYC/NID update threat asking for NID photo and PIN", "fake refund asking for card number and CVV",
        "relative in trouble asks for PIN to send money", "SIM-swap or telecom offer that asks for the OTP just received",
        "prize or job message that ends by asking for an OTP (priority -> class 1)"],
    2: ["lucky draw winner must pay a processing fee", "free data/talk-time gift via a link", "cashback reward requires clicking a link",
        "spin-and-win app download", "smartphone gift that requires a delivery charge", "anniversary voucher quiz with a link",
        "gift parcel that demands customs payment (priority -> class 2)"],
    3: ["work-from-home job with a registration/security fee", "like/subscribe task earning via Telegram group",
        "crypto or forex signal group promising guaranteed profit", "money doubling in days", "abroad job with visa fee",
        "freelancing course guaranteeing income", "e-commerce order-boosting task scheme"],
    4: ["parcel held at hub, pay storage fee via link", "customs duty for an unexpected international parcel",
        "address confirmation link for failed delivery", "courier asks to call a number to reschedule",
        "fake tracking link that asks for card details"],
    5: ["Eid / Puja / Boishakh sale discounts", "telecom data pack offer with dial code", "coaching or course admission ad",
        "real-estate or apartment installment ad", "restaurant or food-delivery combo deal", "app install promotion with a coupon code",
        "clothing brand end-of-season sale"],
}
BRANDS = ["bKash", "Nagad", "Rocket", "Upay", "Grameenphone", "Robi", "Banglalink", "Teletalk", "Daraz", "Chaldal",
          "Pathao", "Foodpanda", "RedX", "Steadfast", "Paperfly", "eCourier", "Sundarban Courier", "SA Paribahan", "DHL",
          "BRAC Bank", "Dutch-Bangla Bank", "City Bank", "Islami Bank", "DESCO", "Titas Gas", "Bdjobs", "Shwapno", "Aarong",
          "a new unknown brand"]


def synth_specs(n_prompts: int, seed: int) -> list[dict]:
    """Seeded, class-balanced spec list. Each spec asks for `per_prompt` records of one class."""
    rng = random.Random(seed)
    specs = []
    for i in range(n_prompts):
        label = i % 6
        specs.append({"label": label, "lang": rng.choice(LANGS), "format": rng.choice(FORMATS),
                      "scenario": rng.choice(SCENARIOS[label]), "brand": rng.choice(BRANDS),
                      "amount": rng.choice([45, 99, 150, 300, 500, 1200, 2000, 3500, 5000, 10000, 25000, 50000])})
    return specs


def synth_messages(spec: dict, per_prompt: int) -> list[dict]:
    lab = spec["label"]
    want = (f"Write {per_prompt} different, realistic records of class {lab} ({CLASS_NAMES[lab]}).\n"
            f"Scenario: {spec['scenario']}. Brand/sender idea: {spec['brand']} (vary it). Amount idea: Tk {spec['amount']}.\n"
            f"Language: {spec['lang']}. Format: {spec['format']}.\n"
            "Mask phone numbers as 01XXXXXXXXX and defang links like hxxps://example[.]com. No real personal data.\n"
            "Make each record different in wording, sender and details. Label techniques and action correctly "
            "(typical: class 1 uses T2, class 2 uses T4, class 3 uses T6, a claimed organization adds T1, pressure or "
            "trust-building adds T5, a fee or payment adds T3).\n"
            'Reply with ONLY a JSON array: [{"text": str, "label": int, "techniques": "T1|T5" or "NONE", "action": int}, ...]')
    return [{"role": "system", "content": LABEL_GUIDE}, {"role": "user", "content": want}]


def quality_filter(rows: list[dict], near_dup: float = 0.9) -> tuple[pd.DataFrame, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.DataFrame(rows)
    rep = {"raw": len(df)}
    if df.empty:
        return df, rep
    key = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[~key.duplicated()].reset_index(drop=True)
    rep["after_exact_dedup"] = len(df)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(df["text"])
    S = (X @ X.T).tocsr()
    keep = np.ones(len(df), bool)
    for i in range(len(df)):
        if keep[i]:
            row = S.getrow(i)
            dup = row.indices[(row.data >= near_dup) & (row.indices > i)]
            keep[dup] = False
    df = df[keep].reset_index(drop=True)
    rep["after_near_dedup"] = len(df)
    rep["class_counts"] = df["label"].value_counts().sort_index().to_dict()
    rep["action_counts"] = df["action"].value_counts().sort_index().to_dict()
    rep["technique_counts"] = {t: int(df["techniques"].str.contains(t).sum()) for t in TECHNIQUES}
    rep["conversation_share"] = round(float(df["text"].str.contains(r"\|\|\|").mean()), 3)
    return df, rep


def _parse_spec_output(spec: dict, text: str) -> list[dict] | None:
    arr = parse_json(text)
    if isinstance(arr, dict):
        arr = [arr]
    if not isinstance(arr, list):
        return None
    rows = []
    for obj in arr:
        r = clean_record(obj, want_label=spec["label"]) if isinstance(obj, dict) else None
        if r:
            r["spec_lang"], r["spec_format"] = spec["lang"], spec["format"]
            rows.append(r)
    return rows


def generate_synthetic(gen, n_target: int, seed: int = 42, per_prompt: int = 5, chunk: int = 32,
                       checkpoint_path: str | None = None, max_minutes: float | None = None,
                       retry: bool = True) -> tuple[pd.DataFrame, dict]:
    """Prompts are processed in chunks; valid rows are checkpointed after each chunk so a time limit or crash keeps
    the finished work. Prompts whose output cannot be parsed get ONE retry (bounded)."""
    specs = synth_specs(int(np.ceil(n_target / per_prompt)), seed)
    t0, rows, failed, stopped = time.time(), [], [], False
    for i in range(0, len(specs), chunk):
        part = specs[i:i + chunk]
        for spec, o in zip(part, gen([synth_messages(sp, per_prompt) for sp in part])):
            got = _parse_spec_output(spec, o)
            if got is None:
                failed.append(spec)
            else:
                rows.extend(got)
        if checkpoint_path:
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        print(f"[synth] {i + len(part)}/{len(specs)} prompts, {len(rows)} raw rows, {len(failed)} failed", flush=True)
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print(f"[synth] time limit {max_minutes} min reached - keeping finished prompts", flush=True)
            stopped = True
            break
    retried_ok = 0
    if retry and failed and not stopped:
        again = [dict(sp, scenario=sp["scenario"] + " (write valid JSON only)") for sp in failed]
        for i in range(0, len(again), chunk):
            for spec, o in zip(again[i:i + chunk], gen([synth_messages(sp, per_prompt) for sp in again[i:i + chunk]])):
                got = _parse_spec_output(spec, o)
                if got:
                    rows.extend(got)
                    retried_ok += 1
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                break
    df, rep = quality_filter(rows)
    rep.update(unparseable_prompts=len(failed), retried_ok=retried_ok, prompts=len(specs), time_limited=stopped)
    if not df.empty:
        df.insert(0, "id", [f"syn{seed}_{i:05d}" for i in range(len(df))])
    return df, rep


# ------------------------------------------------------------------ teacher (pseudo-labels for test rows)
def make_vllm_generate_fn(model_name: str, max_new_tokens: int, seed: int = 42, with_conf: bool = False):
    """Fast batched greedy generation with vLLM (tensor-parallel over all GPUs, prefix caching so the shared
    instructions + few-shot block is computed once)."""
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, tensor_parallel_size=max(1, torch.cuda.device_count()), dtype="half",
              max_model_len=4096, enable_prefix_caching=True, gpu_memory_utilization=0.90, seed=seed)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=1 if with_conf else None)

    def gen(conversations):
        outs = llm.chat(conversations, sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        if not with_conf:
            return texts
        conf = []
        for o in outs:
            c = o.outputs[0]
            steps = []
            for tid, lp in zip(c.token_ids, c.logprobs or []):
                entry = lp.get(tid)
                if entry is not None:
                    steps.append((entry.decoded_token or "", float(np.exp(entry.logprob))))
            conf.append(_first_digit_prob(steps))
        return texts, conf
    return gen


def teacher_label(test: pd.DataFrame, shots: pd.DataFrame, gen, n: int, seed: int, out_path: str,
                  chunk: int = 512, max_minutes: float | None = None) -> dict:
    """Label a seeded random sample of TEST rows with the LLM (programmatic pseudo-labeling). Writes
    id,label,techniques,action incrementally so a time limit keeps whatever is finished. Prints counts only."""
    sample = test.sample(min(n, len(test)), random_state=seed)
    t0, rows, bad = time.time(), [], 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(sample), chunk):
        part = sample.iloc[i:i + chunk]
        res = gen([classify_messages(t, shots) for t in part["raw_text"]])
        outs, confs = res if isinstance(res, tuple) else (res, [1.0] * len(part))
        for rid, o, cf in zip(part["id"], outs, confs):
            j = parse_json(o)
            rec = None
            if isinstance(j, dict):
                rec = clean_record({"text": "placeholder text",
                                    **{k: j.get(k) for k in ("label", "techniques", "action")}})
            if rec is None:
                bad += 1
                continue
            rows.append({"id": rid, "label": rec["label"], "techniques": rec["techniques"], "action": rec["action"],
                         "conf": round(float(cf), 4)})
        pd.DataFrame(rows, columns=["id", "label", "techniques", "action", "conf"]).to_csv(out_path, index=False)
        el = (time.time() - t0) / 60
        print(f"[teacher] {i + len(part)}/{len(sample)} rows, {len(rows)} kept, {bad} unusable, "
              f"{(i + len(part)) / max(el * 60, 1e-9):.1f} rows/s", flush=True)
        if max_minutes and el > max_minutes:
            print(f"[teacher] time limit {max_minutes} min reached - keeping {len(rows)} labels", flush=True)
            break
    df = pd.DataFrame(rows, columns=["id", "label", "techniques", "action", "conf"])
    secs = time.time() - t0
    return {"requested": int(n), "labeled": len(df), "unusable": bad, "minutes": round(secs / 60, 1),
            "rows_per_sec": round(min(n, len(test)) / max(secs, 1e-9), 2) if len(df) else 0.0,
            "conf_quantiles": df["conf"].quantile([0.1, 0.25, 0.5]).round(3).tolist() if len(df) else [],
            "share_conf_ge_0.8": round(float((df["conf"] >= 0.8).mean()), 4) if len(df) else 0.0,
            "label_counts": df["label"].value_counts().sort_index().to_dict() if len(df) else {},
            "action_counts": df["action"].value_counts().sort_index().to_dict() if len(df) else {},
            "shot_ids": [str(x) for x in shots["id"].tolist()]}


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "synth", "teacher"])
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="hf")
    ap.add_argument("--max-minutes", type=float, default=None, help="teacher: stop after this many minutes")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--shots", type=int, default=2, help="few-shot examples per class (eval)")
    ap.add_argument("--n", type=int, default=900, help="target synthetic records before filtering")
    ap.add_argument("--per-prompt", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synth_v1.csv")
    args = ap.parse_args(argv)
    from common.data_loader import load_config, load_train, output_dirs
    from common.folds import load_folds
    from common.metrics import save_json

    cfg = load_config()
    _, out_dir = output_dirs(cfg)
    t0 = time.time()
    if args.cmd == "teacher":
        from common.data_loader import load_test
        backend = args.backend
        if backend == "auto":
            try:
                import vllm  # noqa: F401
                backend = "vllm"
            except ImportError:
                backend = "hf"
        if backend == "vllm":
            gen = make_vllm_generate_fn(args.model, max_new_tokens=40, seed=args.seed, with_conf=True)
        else:
            tok, model = load_llm(args.model)
            gen = make_generate_fn(tok, model, max_new_tokens=40, batch_size=args.batch, with_conf=True)
        print(f"teacher backend {backend}, model loaded in {time.time() - t0:.0f}s", flush=True)
        train, test = load_train(cfg), load_test(cfg)
        shots = pick_shots(train, args.shots, args.seed)          # fixed shot set (ids recorded in the report)
        rep = teacher_label(test, shots, gen, args.n, args.seed, args.out, max_minutes=args.max_minutes)
        rep.update(model=args.model, backend=backend)
        save_json(rep, out_dir / "teacher_report.json")
        print(json.dumps(rep))
        return
    tok, model = load_llm(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.0f}s", flush=True)
    if args.cmd == "eval":
        train = load_train(cfg)
        gen = make_generate_fn(tok, model, max_new_tokens=60, batch_size=args.batch)
        t1 = time.time()
        m = llm_eval(train, load_folds(train), gen, per_class=args.shots, seed=args.seed)
        m["seconds"] = round(time.time() - t1, 1)
        save_json(m, out_dir / "eval_llm_fewshot.json")
        print(json.dumps({k: m[k] for k in ("F_A", "F_B", "F_C", "accuracy_A", "unparseable", "seconds")}))
        print("per-class A F1:", {k: v["f1"] for k, v in m["per_class_A"].items()})
    else:
        gen = make_generate_fn(tok, model, max_new_tokens=220 * args.per_prompt, batch_size=args.batch)
        t1 = time.time()
        df, rep = generate_synthetic(gen, args.n, seed=args.seed, per_prompt=args.per_prompt,
                                     checkpoint_path=str(args.out) + ".partial.csv", max_minutes=args.max_minutes)
        rep["seconds"] = round(time.time() - t1, 1)
        rep["model"] = args.model
        df.to_csv(args.out, index=False)
        save_json(rep, out_dir / "synth_report.json")
        print(json.dumps(rep, ensure_ascii=False))
        print("wrote", args.out, len(df), "rows")


if __name__ == "__main__":
    main()
