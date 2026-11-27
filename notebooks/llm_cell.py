# LLM family: OpenAI chat model labels each message from the official task definitions (zero-shot, no test labels,
# no hand labeling). Key is read from the environment / .env (OPENAI_API_KEY). Results are cached so a re-run resumes.
#   python llm_cell.py sample   -> scores the 86 labeled rows (go/no-go check)
#   python llm_cell.py test     -> labels every unique normalized test text, writes llm_test_<model>.csv
import os, re, sys, json, time, asyncio, unicodedata
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(os.environ.get("ENV_FILE", ".env"))
D = os.environ.get("DATA_DIR", "D:/NSU/")
OUT = os.environ.get("LLM_OUT", D)
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
BATCH = int(os.environ.get("LLM_BATCH", 40))
CONC = int(os.environ.get("LLM_CONC", 48))
SEED = 42

SYSTEM = """You label SMS messages and chat conversations from Bangladesh (English, Bangla script, Banglish, mixed; may be noisy, misspelled, obfuscated, with lookalike characters or junk strings). Conversations use ||| between messages. Judge the WHOLE message/conversation by intent and behaviour, not keywords.

LABEL (one of 0-5). If several apply, use this priority and stop at the first match: 1, 2, 4, 3, 5, 0.
1 MFS/OTP scam: asks for PIN, OTP, password, NID, card details or other credentials; claims a mistaken transfer and asks for money back; or impersonates support to get account access/info. A genuine OTP delivery that warns not to share the code is 0.
2 Prize/lottery scam: prize, gift, cashback or reward the person did not legitimately earn/enter, needing payment, a click, details or another action to claim.
4 Delivery/parcel phishing: deceptive/unexpected parcel or delivery claim to get payment, credentials, address confirmation or a link click.
3 Fake job/investment: unrealistic income, upfront recruitment fee, task-based earning, money doubling, crypto/investment schemes.
5 Promotional spam: marketing, sales, discounts, product/app promotions or business offers that are spam-like but not fraudulent.
0 Legitimate: genuine transactional, informational, personal or service communication (including real OTP notices, payment confirmations, real delivery updates, legitimate discounts/cashback promotions, service notices with links).

Examples: "Apnar cash-out 500 taka successful. Balance 1,240 taka." -> 0. "Your OTP is 482913. Do not share it with anyone." -> 0. "Ami agent, bhul number-e taka pathaisi. Ferot din, PIN ta bolen." -> 1. "Apni 5 lakh taka jitechen. Claim korte link e click korun." -> 2. "Ghor bose din e 3000 taka income. WhatsApp e join korun." -> 3. "Parcel atke ase. Delivery fee dite link e jan." -> 4. "Your order #4471 is out for delivery today." -> 0. "Eid offer! Sob item e 30% discount." -> 5. "Get 10% cashback on your next top-up with code EID10." -> 5. "You won cashback. Click to claim within 1 hour." -> 2.

TECHNIQUES (only for labels 1-4; otherwise NONE; can be several): T1 impersonating an authority/organization (bank, MFS, courier, government, telecom, employer); T2 harvesting credentials/OTP/PIN/sensitive info; T3 advance-fee or payment deception; T4 prize/reward/lottery/cashback bait; T5 urgency, threat, pressure, trust-building or other manipulation; T6 task-based earning, fake job, investment or financial manipulation.

ACTION (principal action the sender wants; 0 if label is 0 or 5 with nothing suspicious): 0 none/informational; 1 share credentials or confidential info; 2 transfer money / pay a fee; 3 click/open/visit a link; 4 install an app, join a group or do a task; 5 contact/call/reply to the sender or another party.

Return JSON only: {"r": [[index, label, "T1|T5" or "NONE", action], ...]} with one entry per input, in order."""


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip()


def parse(content, n):
    r = json.loads(content)["r"]
    out = {}
    for e in r:
        i, l, t, a = int(e[0]), int(e[1]), str(e[2]), int(e[3])
        if 0 <= i < n and 0 <= l <= 5 and 0 <= a <= 5:
            t = "|".join(sorted({x for x in re.findall(r"T[1-6]", t)})) or "NONE"
            out[i] = (l, t, a)
    return out


async def label_batch(client, texts, sem):
    user = "\n".join(f"{i}. {t[:1500]}" for i, t in enumerate(texts))
    for attempt in range(6):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL, temperature=0, seed=SEED, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            got = parse(resp.choices[0].message.content, len(texts))
            if len(got) >= len(texts) * 0.9 or len(texts) == 1:
                return got, resp.usage
        except Exception as e:
            if attempt == 5:
                print("batch failed:", repr(e)[:200])
        await asyncio.sleep(2 ** attempt)
    return {}, None


async def run(texts, cache_path):
    done = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            k, l, t, a = json.loads(line); done[k] = (l, t, a)
    todo = [t for t in dict.fromkeys(texts) if t not in done]
    print(f"{MODEL}: {len(done)} cached, {len(todo)} to label")
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=2, timeout=120)
    sem = asyncio.Semaphore(CONC)
    t0, tok_in, tok_out = time.time(), 0, 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), BATCH * CONC * 4):
            chunk = todo[start:start + BATCH * CONC * 4]
            batches = [chunk[i:i + BATCH] for i in range(0, len(chunk), BATCH)]
            res = await asyncio.gather(*[label_batch(client, b, sem) for b in batches])
            for b, (got, usage) in zip(batches, res):
                for i, v in got.items():
                    done[b[i]] = v; f.write(json.dumps([b[i], *v], ensure_ascii=False) + "\n")
                if usage: tok_in += usage.prompt_tokens; tok_out += usage.completion_tokens
            f.flush()
            el = time.time() - t0; k = start + len(chunk)
            print(f"  {k}/{len(todo)} in {el / 60:.1f} min | tokens in {tok_in / 1e6:.2f}M out {tok_out / 1e6:.2f}M", flush=True)
    return done


def main():
    mode = sys.argv[1]
    if mode == "sample":
        from sklearn.metrics import f1_score
        s = pd.read_csv(D + "sample.csv")
        txt = s["text"].map(norm).tolist()
        done = asyncio.run(run(txt, OUT + f"llm_cache_sample_{MODEL}.jsonl"))
        pred = [done.get(t, (0, "NONE", 0)) for t in txt]
        pa = [p[0] for p in pred]; pc = [p[2] for p in pred]
        print("missing:", sum(t not in done for t in txt))
        print("Task A macro F1:", round(f1_score(s["label"], pa, average="macro"), 4),
              "| acc:", round((s["label"].values == pa).mean(), 4),
              "| Task C macro F1:", round(f1_score(s["action"], pc, average="macro"), 4))
        pd.DataFrame({"id": s["id"], "label": s["label"], "llm": pa, "text": s["text"].str[:90]})[
            lambda d: d.label != d.llm].to_csv(OUT + f"llm_sample_errors_{MODEL}.csv", index=False)
    else:
        t = pd.read_csv(D + "test.csv")
        txt = t["text"].fillna("").map(norm)
        done = asyncio.run(run(txt.tolist(), OUT + f"llm_cache_test_{MODEL}.jsonl"))
        pred = [done.get(x) for x in txt]
        print("missing:", sum(p is None for p in pred))
        pd.DataFrame({"id": t["id"], "llm_label": [p[0] if p else -1 for p in pred],
                      "llm_tech": [p[1] if p else "" for p in pred],
                      "llm_action": [p[2] if p else -1 for p in pred]}).to_csv(OUT + f"llm_test_{MODEL}.csv", index=False)


if __name__ == "__main__":
    main()
