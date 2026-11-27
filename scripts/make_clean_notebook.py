"""Clean notebook: rewrite the 48 synthetic-template strings that Rabib flagged (flagged_templates.md: phrases found
verbatim in test.csv and never in sample.csv) in new wording. Only our own template text is edited; test.csv is not read.

    python scripts/make_clean_notebook.py notebooks/09_v3.ipynb notebooks/13_clean.ipynb [--no-part-b]

--no-part-b drops the embedding cells (Part B) and makes the final cell treat the embedding family as optional,
for a faster Kaggle run; the embedding family does not use the templates, so its outputs are unchanged.
"""
import argparse
import json

R = {
    '"{mfs}: {shop} e {amt} payment successful. TrxID {trx}."':
        '"{mfs}: {amt} paid to {shop}. Ref {trx}. Thanks for using {mfs}."',
    '"Reminder: your loan installment {amt} is due tomorrow. Pay via {mfs} Pay Bill or at branch."':
        '"{bank}: EMI of {amt} falls due on {date}. You can pay at any branch or with {mfs} bill pay."',
    '"{courier}: Your parcel is waiting. Delivery fee {fee} will be collected in cash by our rider. No online payment is needed."':
        '"{courier}: Parcel {pid} goes out for delivery today. Keep {fee} cash ready for the delivery man."',
    '"{mfs}: You earned {fee} cashback on your bill payment. It has been credited. Balance {amt}."':
        '"{mfs}: {fee} cashback added for your electricity bill. Current balance {amt}."',
    '"Registration for Spring starts {date}. Pay tuition through the student portal before the deadline."':
        '"Course enrolment for the new semester opens {date}. Fees go through the university portal only."',
    '"{mfs} Alert: your account will be blocked within {hours} hours. Confirm your password and OTP here: {link}"':
        '"{mfs} security: unusual login found. Unlock within {hours} hrs by entering PIN + OTP at {link}"',
    '"Notice: your account will be blocked within {hours} hours. Tell us your card number, expiry and CVV."':
        '"Your bank access stops in {hours} hrs unless you reply with full card no, expiry date & the 3-digit code."',
    '"{mfs} customer care: apnar account {hours} ghontar moddhe band hoye jabe. je 6 digit code ashche seta bolen."':
        '"{mfs} office theke: {hours} ghonta por apnar account off hobe. SMS e asha code ta amake bolen ekhuni."',
    '"{mfs} theke bolchi. apnar wallet limit barano hobe. code ta {phone} e pathan."':
        '"ami {mfs} agent. limit update korte ekta code jabe, oita {phone} number e forward korun."',
    '"{bank} security department theke apnar wallet limit barano hobe, PIN ta ekhoni reply korun."':
        '"{bank} security team: transaction limit boshate apnar PIN lagbe, ekhoni reply den."',
    '"Dear user, your card ending {code4} has been flagged. Verify your card number and CVV at {link} to avoid suspension."':
        '"Card XXXX{code4} is on hold for security. Re-enter the card details with CVV at {link} today."',
    '"Dear user, we are verifying a payment of {amt} to {shop}. Confirm your password and OTP here: {link}"':
        '"Was the {amt} purchase at {shop} you? If not, cancel it by giving your login PIN and OTP: {link}"',
    '"Western Union: {amt} remittance from your brother in {country} is ready. Verify by sending NID photo and wallet PIN."':
        '"Probashi transfer of {amt} from {country} is pending. Send a picture of your NID and your {mfs} PIN to release it."',
    '"{courier}: Apnar parcel {pid} atke ase. Release korte PIN ar OTP den."':
        '"{courier}: parcel {pid} hold e. chharate apnar {mfs} PIN o OTP lagbe, reply din."',
    '"{mfs} Eid Bonus: {amt} is yours! Send your PIN to receive."':
        '"Eid gift from {mfs}: {amt} waiting for you. Reply with PIN to get it now."',
    '"Congratulations! You won a {big} prize in the {mfs} draw. Enter your card number and CVV at {link}"':
        '"Your number won {big} in the {mfs} raffle. Claim by filling card no. and CVV at {link}"',
    '"bhai bhul kore tomar number e {amt} recharge diye felsi, ferot dao pls {phone} e flexiload kore"':
        '"vai {amt} vul number e chole gese, apnar number e. ektu {phone} e pathay den please"',
    '"YOU HAVE BEEN SELECTED FOR A {big} PRIZE! CLICK {link} BEFORE MIDNIGHT."':
        '"WINNER ALERT!!! {big} UNCLAIMED. OPEN {link} TONIGHT ONLY."',
    '"FREE iPhone for {telco} customers! Take the 2-min survey: {link}"':
        '"{telco} gift: answer 5 quick questions and get a free smartphone {link}"',
    '"{mfs} Lucky Draw: Apnar {mfs} account e {big} reward ashche. Delivery charge {fee} {mfs} korun."':
        '"{mfs} lottery te apni {big} jitsen! pathanor khoroch {fee} age {mfs} korte hobe."',
    '"{gov}: Your child is selected for a {amt} stipend. Pay a {fee} verification fee to receive it."':
        '"{gov}: {amt} education allowance approved for your son/daughter. Send {fee} as file charge first."',
    '"Dear winner, you won cashback of {amt}! Pay {fee} processing fee to receive it."':
        '"{amt} cashback reward unlocked! A small {fee} service charge is needed before transfer."',
    '"{company} is hiring! Salary {salary}/month, work from home. Pay {fee} registration fee to start. Limited seats!"':
        '"Home-based job at {company}, monthly pay {salary}. Joining charge {fee} only, few posts left!"',
    '"Selected as brand ambassador for {shop}! Earn {income} per post. Pay {fee} for your kit."':
        '"{shop} wants you as a promoter: {income} for each post. Starter kit costs {fee}."',
    '"Selected for data entry at {company}. Pay {fee} registration fee to start. Apply today, offer closes tonight."':
        '"Typing job confirmed at {company}. Deposit {fee} for your ID card today, seats close at night."',
    '"Crypto bot trading, 30% daily profit guaranteed. Pay {fee} registration fee to start."':
        '"Auto trading robot on USDT, sure profit every day. Activation costs {fee}."',
    '"{courier}: Customs duty is pending for your package. Pay {fee} via {mfs} {phone}."':
        '"{courier}: import tax {fee} due on your overseas item. Send by {mfs} to {phone}."',
    '"{courier}: {country} theke ekta parcel apnar jonno ashche. Release korte {fee} fee din: {link}"':
        '"{courier}: {country} er gift box airport e. charge {fee} pay korle delivery hobe: {link}"',
    '"{courier}: Parcel {pid} er address adhura. Release korte {fee} fee din: {link} Taratari korun."':
        '"{courier}: {pid} parcel er thikana bhul. thik korte {fee} lagbe, {link} e jan."',
    '"Parcel {pid} deliver kora jayni. Release korte {fee} fee din: {link}"':
        '"{pid} pouchano gelo na. abar pathate {fee} charge din: {link}"',
    '"{courier}: ডেলিভারি ব্যর্থ হয়েছে। ঠিকানা আপডেট করতে {link} এ যান।"':
        '"{courier}: আপনার পার্সেল পৌঁছানো যায়নি। সঠিক ঠিকানা দিন এই লিংকে: {link}"',
    '"{shop}: Flash sale ends tonight! Extra {pct}% off. Use code DEAL{pct}. T&C apply."':
        '"{shop}: Last day of mega deals, {pct}% off with promo SAVE{pct}. Conditions apply."',
    '"{telco}: Winter sale is live! Up to {pct}% off."':
        '"{telco}: Shoit offer cholche, {pct}% porjonto chhar."',
    '"{brand}: Admission going on for Digital Marketing course. Details: {glink}"':
        '"{brand}: New batch of our freelancing class starts soon. Info: {glink}"',
    '"{shop}: Recharge {fee} and get {gb} extra. Open the app to apply."':
        '"{shop}: Top up {fee}, enjoy bonus {gb}. Activate from the app."',
    '"{brand}: Book your dream flat, easy installment. Details: {link} T&C apply."':
        '"{brand}: Apartments near the main road, pay monthly. Visit {link}"',
    '"{shop}: End of season sale up to {pct}% off. Visit your nearest outlet. T&C apply."':
        '"{shop}: Clearance going on, {pct}% less on selected items in all stores."',
    '"{mfs}: Apnar jonno special offer: {pct}% off. {short} e call korun. Shortto projojjo."':
        '"{mfs}: shudhu apnar jonno {pct}% chhar! jante phone korun {short}."',
    '"{telco}: Eid offer! Sob item e {pct}% discount. {short} e call korun. Limited time!"':
        '"{telco}: Eid e shob kichute {pct}% kom. aro jante {short}."',
    '"Refer a friend and earn {fee} credit on your next order. Dial {ussd}. Limited time only."':
        '"Invite friends, get {fee} wallet credit for each one. Dial {ussd} to share."',
    '"Order 3 times this week and unlock a {fee} voucher. Offer ends Sunday."':
        '"Place three orders before the weekend and receive a {fee} coupon."',
    '"{brand}: New app update with exciting features. Call {short}. Limited time only."':
        '"{brand}: Our app just got better, update from Play Store. Help: {short}."',
    '"{fee} recharge e {gb} extra pan. Nikotbortee showroom e ashun."':
        '"{fee} er recharge korle bonus {gb}. kache store e ashen."',
    '("assalamu alaikum", "walaikum assalam, ke bolchen?")':
        '("assalamu alaikum", "wa alaikum, ji bolun")',
    '"We hope this message finds you well. "': '"Greetings. "',
    '"This is an automated notification from our system. "': '"[Auto SMS] "',
    '" Thank you for your cooperation."': '" Thanks."',
    '" Please do not reply to this automated message."': '" No reply needed."',
    '"eta scam bhai, kauke PIN ba OTP dibi na, block kor"': '"bhai eta fraud, PIN OTP share korish na, number ta block de"',
}
FLAGGED = ["payment successful. TrxID", "Reminder: your loan installment", "collected in cash by our rider",
           "cashback on your bill payment", "Registration for Spring starts", "Confirm your password and OTP here",
           "your account will be blocked within", "customer care: apnar account", "ghontar moddhe band hoye jabe",
           "wallet limit barano hobe", "Dear user, your card ending", "Dear user, we are verifying a payment of",
           "remittance from your brother in", "Verify by sending NID photo", "atke ase. Release korte PIN",
           "is yours! Send your PIN to receive", "Congratulations! You won a", "draw. Enter your card number",
           "bhai bhul kore tomar number e", "recharge diye felsi", "YOU HAVE BEEN SELECTED FOR A",
           "customers! Take the 2-min survey", "reward ashche. Delivery charge", "Your child is selected for a",
           "processing fee to receive it", "registration fee to start", "/month, work from home. Pay",
           "Selected as brand ambassador for", "Selected for data entry at", "Crypto bot trading",
           "Customs duty is pending for your package", "theke ekta parcel apnar jonno ashche", "er address adhura",
           "deliver kora jayni", "ডেলিভারি ব্যর্থ হয়েছে। ঠিকানা আপডেট করতে", "Flash sale ends tonight",
           "Winter sale is live", "Admission going on for Digital Marketing", "extra. Open the app to apply",
           "Book your dream flat, easy installment", "Visit your nearest outlet. T&C apply", "Apnar jonno special offer",
           "e call korun. Shortto projojjo", "e call korun. Limited time!", "credit on your next order. Dial",
           "Order 3 times this week and unlock a", "voucher. Offer ends Sunday", "New app update with exciting features",
           "extra pan. Nikotbortee showroom", "walaikum assalam, ke bolchen", "This is an automated notification from our system",
           "We hope this message finds you well", "Please do not reply to this automated message",
           "Thank you for your cooperation", "eta scam bhai, kauke PIN ba OTP dibi na"]


def src(c):
    return "".join(c["source"])


def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--no-part-b", action="store_true")
    a = ap.parse_args()
    nb = json.load(open(a.src, encoding="utf-8"))
    C = nb["cells"]
    k = next(i for i, c in enumerate(C) if c["cell_type"] == "code" and "def make_synth" in src(c))
    s = src(C[k])
    for old, new in R.items():
        s = rep(s, old, new)
    left = [p for p in FLAGGED if p.lower() in s.lower()]
    assert not left, left
    compile(s, "synth", "exec")
    C[k]["source"] = s.splitlines(keepends=True)
    p = next(i for i, c in enumerate(C) if c["cell_type"] == "markdown" and src(c).startswith("# Part A"))
    C[p]["source"] = (src(C[p]).rstrip("\n") + "\n\n"
                      "**Clean rerun:** 49 synthetic-template strings whose wording matched phrases that occur in `test.csv`\n"
                      "(found by an n-gram audit of our own templates) were rewritten in new wording. No test text was used\n"
                      "to write the replacements.\n").splitlines(keepends=True)
    if a.no_part_b:
        b0 = next(i for i, c in enumerate(C) if c["cell_type"] == "markdown" and src(c).startswith("# Part B"))
        c0 = next(i for i, c in enumerate(C) if c["cell_type"] == "markdown" and src(c).startswith("# Part C"))
        fin = C[c0 + 1]
        f = rep(src(fin), "O, TS = {'emb': emb_family('oof')}, {'emb': emb_family('test')}",
                "O, TS = {}, {}\nif all(os.path.exists(f'{OUT}{s}_{k}_{t}.npy') for k in EMB for s in ['oof', 'test'] for t in 'ABC'):\n"
                "    O['emb'], TS['emb'] = emb_family('oof'), emb_family('test')")
        compile(f, "final", "exec")
        fin["source"] = f.splitlines(keepends=True)
        nb["cells"] = C[:b0] + C[c0:]
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1); fh.write("\n")
    print("wrote", a.out, "| rewrote", len(R), "strings | cells", len(nb["cells"]))


if __name__ == "__main__":
    main()
