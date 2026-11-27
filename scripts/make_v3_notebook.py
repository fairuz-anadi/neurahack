"""v3 = v2 (no pseudo-labeling) + extra templates for the error types seen on sample.csv + averaging of
2 XLM-R seeds for the final test predictions. Works on the base or the large v2 notebook.

    python scripts/make_v3_notebook.py IN_V2.ipynb OUT_V3.ipynb

The new templates were written from the out-of-fold errors on the 86 labeled rows (sample.csv) and from
general knowledge of Bangladeshi SMS. No test text was read to write them.
"""
import json
import sys

V3_TEMPLATES = r'''
    # ---------------- v3: extra templates for error types seen on sample.csv (no test text used) ----------------
    # legitimate notices that look like telecom / courier / job / institution messages
    add(0, "", 0,
        "Dear customer, your {telco} {gb} pack will expire on {date}. To renew, dial {ussd} or use the {telco} app.",
        "{telco}: Your remaining data is 200MB. Your pack is valid till {date}. Thank you for staying with {telco}.",
        "{telco}: Your SIM has been re-registered successfully with your NID. No further action is needed.",
        "Priyo grahok, apnar {telco} internet pack {date} e sesh hobe. Renew korte {ussd} dial korun.",
        "{telco}: Apnar minute pack er meyad ajke rat 12 tay sesh hobe.",
        "{courier}: Parcel {pid} has reached our {name} Road branch. Please collect it within 3 days with your NID.",
        "{courier}: Your parcel {pid} could not be delivered today because nobody was home. We will try again tomorrow.",
        "{courier}: Apnar parcel {pid} branch e pouche geche. Office time e eshe niye jan.",
        "{courier} rider {name}: bhai ami apnar basar niche, parcel ta niye jan please.",
        "Your interview at {company} is confirmed for {date}, {time}. Please bring your CV and certificates. Reception will guide you.",
        "{company} HR: Thank you for applying. Your written test is on {date} at {time} at our head office. No fee is required.",
        "HR, {company}: Your offer letter has been sent to your email. Please sign and return it by {date}.",
        "Interview kal {time} e, {company} office e. CV er hard copy niye ashben.",
        "University notice: Mid-term exams start on {date}. Check the routine on the student portal.",
        "Library notice: Please return the borrowed book by {date} to avoid a late fine.",
        "Dear student, your tuition payment of {amt} has been received. Receipt no. {pid}.",
        "Hospital: Your appointment with Dr. {name} is on {date} at {time}. Please arrive 15 minutes early.",
        "Team, the meeting is moved to {time}. Same Zoom link as last week.",
        "Bhai kal class {time} theke shift kore {time} e hobe, sir group e janaise.",
        "Amar train late, {time} er dike pouchabo. Tumi kheye nio.",
        "Doctor er serial dite hobe kal, tumi ektu call kore dite parba?",
        "R: I forgot my {mfs} PIN, what should I do? ||| S: Dial {ussd} and follow the reset steps, or visit a customer care centre with your NID. Never share your PIN with anyone.",
        "R: amar {mfs} account lock hoye geche ||| S: {short} e call korun ba nearest service point e NID niye jan. Kauke PIN ba OTP diben na.",
        "{mfs}: Your PIN was changed successfully. If you did not do this, call {short} immediately.",
        "{bank}: Your new debit card is ready for collection at the branch. Please bring your NID.",
        "{bank}: Your monthly e-statement for A/C XXXX{code4} has been sent to your email.",
        "{gov}: Your application {pid} has been received. You will be notified by SMS about the next step.",
        "আপনার {bntelco} ইন্টারনেট প্যাকের মেয়াদ আগামীকাল শেষ হবে। নবায়ন করতে অ্যাপ ব্যবহার করুন।",
        "{bncourier}: আপনার পার্সেল শাখায় পৌঁছেছে। জাতীয় পরিচয়পত্র নিয়ে সংগ্রহ করুন।",
        "আগামীকাল সকাল ১০টায় ইন্টারভিউ, সিভি ও সার্টিফিকেট সাথে আনবেন।",
    )
    # promotions that are not scams (real-estate, installments, services)
    add(5, "", 0,
        "Book your flat in {name} Tower, Bashundhara. Easy installment, handover in 2027. Call {short}.",
        "Ready flat in Uttara! 1350 sqft, easy EMI. Visit our project this Friday.",
        "{bank} credit card: 0% EMI on electronics at {shop} this month. T&C apply.",
        "{shop}: New arrivals in store! Visit your nearest outlet this weekend.",
        "Admission open for HSC batch at our coaching centre. Free trial class on {date}.",
        "Gym membership {pct}% off this month only. Visit us today.",
        "{telco}: Get {gb} bonus on every recharge of {fee} this week. Dial {ussd}",
        "Flat booking cholche! Sohoj kistite nijer flat. Details er jonno call korun {short}.",
    )
    # credential request inside a prize message -> class 1 (credential requests take priority)
    add(1, "T1|T2|T4|T5", 1,
        "Congratulations! Apnar number {mfs} lucky draw e {big} jitese. Taka pete OTP ta amader bolun.",
        "You won {big} in the {telco} Eid draw! Reply with the OTP we just sent to claim it.",
        "Abhinandan! Apni {amt} cashback peyechen. Pete {mfs} PIN ar OTP pathan {phone} e.",
        "অভিনন্দন! আপনি {bnamt} পুরস্কার জিতেছেন। টাকা পেতে আপনার পিন ও ওটিপি পাঠান।",
    )
    # task-based earning that looks like shopping promotions -> class 3
    add(3, "T3|T6", 4,
        "Shop on {shop} with our link and get a full refund plus {pct}% commission per order. Finish 3 tasks a day.",
        "Order task: buy any item on {shop} through our app, we refund the price and pay {income} commission.",
        "Product review job: buy, review and get the money back with {pct}% extra. Join our WhatsApp group.",
        "{shop} er product order kore review din, taka ferot + {pct}% commission. Telegram e join korun: {link}",
    )
'''

SEED_OLD = "    xm = XlmrModel().fit(d, TBd, seed=SEED); Px = xm.predict(uniq); xm.free()\n"
SEED_NEW = ("    Pxs = []\n"
            "    for s in range(FINAL_SEEDS):                 # average several XLM-R seeds for steadier test predictions\n"
            "        xm = XlmrModel().fit(d, TBd, seed=SEED + s); Pxs.append(xm.predict(uniq)); xm.free()\n"
            "    Px = tuple(np.mean([p[k] for p in Pxs], axis=0) for k in range(len(Pxs[0])))\n")


def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


def set_src(cell, s):
    lines = s.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


TAIL = open(__file__.replace("make_v3_notebook.py", "v3_final_tail.py"), encoding="utf-8").read()


def v3_edits(nb):
    C = nb["cells"]
    get = lambda i: "".join(C[i]["source"])

    s = get(3)
    s = rep(s, 'BATCH        = int(os.environ.get("SS_BATCH",',
            'FINAL_SEEDS  = int(os.environ.get("SS_FINAL_SEEDS", 2))       # XLM-R seeds averaged for the final test predictions\n'
            'BATCH        = int(os.environ.get("SS_BATCH",')
    set_src(C[3], s)

    s = get(5)
    s = rep(s, "\n    # ---------------- composition helpers ----------------", V3_TEMPLATES.rstrip() + "\n\n    # ---------------- composition helpers ----------------")
    set_src(C[5], s)

    s = get(13)
    s = rep(s, SEED_OLD, SEED_NEW)
    set_src(C[13], s)

    s = get(len(C) - 1)
    i = s.index("# ---- confidence: temperature by Brier ----")
    set_src(C[-1], s[:i] + TAIL.rstrip() + "\n")

    s = get(0)
    s = rep(s, "* Consistency rules from the labeling guide:",
            "* **v3 additions:** extra templates for the error types seen in out-of-fold predictions on `sample.csv` (telecom, courier,\n"
            "  HR and institution notices that are legitimate; non-scam real-estate and service promotions; prize messages that ask\n"
            "  for an OTP, which are class 1; shopping-task earning schemes, which are class 3). They were written from the labeled\n"
            "  rows and general knowledge, without reading test text. The final XLM-R test predictions average 2 seeds (42, 43).\n"
            "  The last cell scores two candidates out-of-fold (the shrunk 3-family ensemble and the TF-IDF + XLM-R blend)\n"
            "  and writes the one with the higher CV proxy to `submission.csv`; both are also saved as `sub_<name>.csv`.\n"
            "* Consistency rules from the labeling guide:")
    set_src(C[0], s)
    return nb


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    nb = json.load(open(src, encoding="utf-8"))
    nb = v3_edits(nb)
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
