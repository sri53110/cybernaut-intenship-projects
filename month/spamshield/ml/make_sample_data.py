"""Generate a synthetic spam/ham corpus so the project runs out of the box.

This is NOT a substitute for a real dataset: metrics on it are only a smoke test.
Train on the UCI SMS Spam Collection or SpamAssassin for real numbers (see README).
"""
import csv, random, sys
from pathlib import Path

R = random.Random(7)
pick = R.choice

S_OPEN = ["WINNER!!", "Congratulations!", "URGENT:", "Final notice:", "Dear customer,", "Hot offer!", "ALERT:", "You are selected!",
          "Limited time only!!!", "Attention:", "Free gift!", "Act now!"]
S_HOOK = ["you have won a ${n} cash prize", "your account has been suspended", "you are pre-approved for a ${n} loan with no credit check",
          "claim your free {item} today", "we detected unusual activity on your account", "your parcel is held and a ${m} fee is due",
          "earn ${n} per day working from home", "get {m}0% discount on cheap meds, no prescription",
          "double your bitcoin in {m} days guaranteed", "your {item} subscription renews for ${m}",
          "you were chosen to receive a free {item}", "hot singles in your area want to chat",
          "your bank details need to be verified immediately", "exclusive deal: buy one {item} get one free"]
S_CTA = ["Click http://{d}/{w} now", "Call {p} to claim", "Reply WIN to {n2}", "Verify at www.{d}", "Text CLAIM to {n2}",
         "Visit http://{d} before midnight", "Send your password to {w}@{d}", "Act now, offer expires tonight",
         "Pay now at http://{d}/pay", "Order today and save big", "T&Cs apply. 18+ only"]
ITEMS = ["iPhone", "holiday trip", "gift card", "laptop", "cruise", "TV", "cash voucher", "PS5", "watch"]
DOMS = ["win-prize.biz", "secure-bank-login.top", "freegift.xyz", "cheap-deals.info", "verify-now.co", "bigwin.click", "promo-hub.online"]
WORDS = ["claim", "verify", "login", "offer", "reward", "secure", "bonus"]

H_OPEN = ["Hey,", "Hi,", "Hello,", "Good morning,", "Hey there,", "Hi all,", "", "Quick one:", "Btw,", "Ok so"]
H_CORE = ["are we still meeting for {meal} at {t} {day}?", "can you send me the notes from the {sub} class",
          "the {sub} assignment deadline is {day}, did you finish?", "team meeting moved to {t} on {day}, room {room}",
          "thanks for the update, I'll review the report and reply by end of day",
          "I'm running late, the {veh} is stuck in traffic, start without me",
          "please find the attached invoice for last month, tell me if anything looks off",
          "{name} called, wants you to ring back after dinner", "dinner at my place on {day}? I'm cooking {food}",
          "the project review went well, great job on the presentation", "can we reschedule our call to {t}? work came up",
          "don't forget your laptop charger for the lab {day}", "just landed, will text when I reach the hotel",
          "could you pick up {food} on your way home?", "here are the slides for the {sub} seminar, add comments if any",
          "happy birthday {name}! hope your day is great", "I'll be at the library until {t}, join me if free",
          "the {sub} exam is on {day}, want to study together?", "I left my keys at your place, can I pick them up {day}?",
          "the code review for the {sub} project is done, a few small comments"]
H_END = ["", "", "Thanks!", "See you then.", "Let me know.", "Cheers, {name}", "Free to talk now?", "Call me when you can.",
         "The offer from the other company sounds good.", "Please confirm.", "Talk later."]
NAMES = ["Mom", "Dad", "Priya", "Arjun", "Sam", "Rahul", "Anita", "Karthik", "Meera", "John"]
SUBS = ["physics", "maths", "chemistry", "history", "machine learning", "networks", "biology", "economics"]
TIMES = ["12:30", "1 pm", "3:15", "5 pm", "10 am", "9:45", "6 pm", "2 pm"]
DAYS = ["tomorrow", "on Friday", "this weekend", "on Monday", "tonight", "next week", "on Saturday"]
FOODS = ["pasta", "biryani", "pizza", "milk and bread", "rice", "vegetables", "dosa"]
MEALS = ["lunch", "coffee", "dinner", "breakfast"]


def fmt(s):
    return s.format(n=pick([50, 100, 500, 1000, 5000, 10000]), m=R.randint(2, 30), n2=R.randint(80000, 89999),
                    p="0" + str(R.randint(8000000000, 8999999999)), item=pick(ITEMS), d=pick(DOMS), w=pick(WORDS),
                    name=pick(NAMES), sub=pick(SUBS), t=pick(TIMES), day=pick(DAYS), food=pick(FOODS), meal=pick(MEALS),
                    room=pick(["4B", "12", "A3", "201"]), veh=pick(["bus", "metro", "train", "cab"]))


def spam():
    s = f"{pick(S_OPEN)} {pick(S_HOOK)}. {pick(S_CTA)}"
    if R.random() < .3: s += " " + pick(S_CTA)
    s = fmt(s)
    if R.random() < .2: s = s.upper()
    if R.random() < .2: s += " !!!"
    return s


def ham():
    s = f"{pick(H_OPEN)} {pick(H_CORE)}. {pick(H_END)}"
    return fmt(" ".join(s.split()))


def main(out="data/sample_sms.csv", n_ham=1500, n_spam=500):
    seen, rows = set(), []
    for gen, lab, n in ((ham, "ham", n_ham), (spam, "spam", n_spam)):
        tries = 0
        while sum(1 for r in rows if r[1] == lab) < n and tries < n * 50:
            tries += 1
            t = gen()
            if t not in seen:
                seen.add(t); rows.append((t, lab))
    rows = [(t, ("spam" if l == "ham" else "ham") if R.random() < .01 else l) for t, l in rows]  # ~1% label noise
    R.shuffle(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["text", "label"]); w.writerows(rows)
    print(f"wrote {len(rows)} unique rows to {out}")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
