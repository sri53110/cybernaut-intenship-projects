"""SafeTriage core: synthetic corpus -> dedup (FR-01) -> rules / TF-IDF LogReg / RandomForest (FR-02)
-> campaign holdout (FR-03) -> abstain + safe cues (FR-04) -> urgent-FPR + robustness drift (FR-05).
Run `python pipeline.py` to rebuild artifacts/. Nothing here ever opens, fetches or executes a link (NFR-01)."""
import json, random, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 7
ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)
HOLDOUT = ["p_ceo", "p_doc", "b_payroll", "b_meet"]  # mentor-assigned campaigns go here (FR-03)
ABSTAIN_BAND = (0.35, 0.65)
URGENT = r"\b(urgent\w*|immediately|today|expires?|final notice|within \d+ hours?|cutoff)\b"
DISCLAIMER = "Decision support only, not a guarantee of safety. Links are never opened."

# ---------- synthetic, defanged corpus (NFR-02: no real data) ----------
NAMES = ["Sam", "Priya", "Liam", "Aisha", "Noah", "Mei"]
NAME_RE = re.compile(r"\b(" + "|".join(NAMES) + r")\b")
GREET = ["Hi {name},", "Hello {name},", "Dear {name},", "{name} -"]
CLOSE = ["Thanks.", "Regards, Service Desk", "- Team"]
URLS = {"short": ["hxxp://bit[.]ly/3xK9a", "hxxps://t[.]co/Qa81z"],
        "odd": ["hxxps://secure-login[.]acme-portal[.]test/v"],
        "internal": ["hxxps://intranet[.]acme[.]test/it"]}
# campaign: (label, channel, sender_type, subjects, cores, asks)
C = {
 "p_suspend": (1, "email", "lookalike", ["Account notice", "Action needed"],
   ["Your mailbox has been suspended due to unusual activity.", "We detected a new sign-in and locked your account.",
    "Your password expires today and access will be removed.", "Final notice: your account closes within {n} hours."],
   ["Verify your password here: {url}", "Confirm your credentials at {url} immediately.", "Click {url} to restore access."]),
 "p_prize": (1, "sms", "unknown", ["Reward", "You won"],
   ["You have been selected for a {n}0 dollar reward.", "Claim your free gift card before midnight.",
    "Your loyalty points are about to expire.", "Congratulations, a prize is waiting."],
   ["Tap {url} to claim.", "Reply with your card details or open {url}.", "Claim now: {url}"]),
 "p_invoice": (1, "email", "lookalike", ["Unpaid invoice #{n}", "Payment overdue"],
   ["Invoice {n} is overdue and a penalty applies.", "We could not process your last payment.",
    "Your subscription renewal failed.", "The attached invoice requires wire payment today."],
   ["Pay at {url} to avoid suspension.", "Confirm your bank login at {url}.", "Review the payment portal: {url}"]),
 "p_mfa": (1, "email", "lookalike", ["Security update", "Device check"],
   ["Your multi-factor device needs re-enrolment.", "We are migrating authentication apps this week.",
    "Approve the pending sign-in request now.", "Enter the code we just sent to confirm it is you."],
   ["Re-enrol at {url}.", "Send us the code in reply.", "Start here: {url}"]),
 "p_ceo": (1, "email", "unknown", ["Quick favour", "Task"],
   ["Are you at your desk? I need a quick favour.", "I'm in a meeting and can't talk, can you help discreetly?",
    "Could you pick up a few vouchers for a client today?", "Small task before the board call, keep this between us."],
   ["Reply here and I will send details.", "Text me your number so I can explain.", "Let me know when you are free."]),
 "p_doc": (1, "email", "lookalike", ["Shared with you", "Documents"],
   ["A colleague shared a document with you.", "Please review the updated roster when you can.",
    "Your scan from the printer is ready to view.", "A voice note was left for you."],
   ["Open it here: {url}", "Sign in to view: {url}", "Preview: {url}"]),
 "b_reset": (0, "email", "internal", ["Password reset", "Account access"],
   ["You asked to reset your password.", "Your password expires in {n} days; please update it.",
    "A password reset was requested from the helpdesk portal.", "Your temporary password is ready for first login."],
   ["Use the intranet portal: {url}", "If this wasn't you, call the helpdesk.", "Reset it at {url}; we never ask for it by email."]),
 "b_payroll": (0, "email", "internal", ["Payroll", "Payslip ready"],
   ["Your payroll for this month is processed.", "Payslips are available in the HR portal.",
    "Please confirm your bank details change with HR in person.", "Payroll cutoff is today; submit timesheets by 5 pm."],
   ["View it at {url}.", "Contact payroll if anything looks wrong.", "Submit timesheet: {url}"]),
 "b_it": (0, "email", "internal", ["IT notice", "Maintenance"],
   ["Scheduled maintenance tonight from 10 pm.", "The VPN client update is available.",
    "Printer on floor {n} is back online.", "Wi-Fi will be restarted urgently at noon."],
   ["No action needed.", "Details: {url}", "Save your work beforehand."]),
 "b_hr": (0, "email", "internal", ["HR update", "Reminder"],
   ["The annual leave policy was updated.", "Security awareness training is due this month.",
    "Open enrolment starts next week.", "Reminder: complete the mandatory course."],
   ["Read it at {url}.", "Questions? Ask your manager.", "Register: {url}"]),
 "b_meet": (0, "email", "internal", ["Meeting", "Agenda"],
   ["Team meeting moved to {n} pm.", "Please send agenda items before Friday.",
    "The quarterly review is in room {n}.", "Lunch and learn tomorrow."],
   ["Calendar invite attached.", "See {url}.", "Reply to confirm."]),
 "b_parcel": (0, "sms", "vendor", ["Delivery", "Order update"],
   ["Your parcel was delivered to reception.", "Courier pickup is booked for tomorrow.",
    "Your order has shipped.", "A delivery attempt was made today."],
   ["Track it at {url}.", "Collect it from the front desk.", "See {url}."]),
}


def url_feat(t):  # sanitized_url_features
    if re.search(r"bit\[?\.\]?ly|\bt\[?\.\]?co/", t): return "shortened"
    if "intranet[.]acme" in t: return "internal"
    return "other_url" if re.search(r"hxxp|https?:", t) else "no_url"


def make_corpus():
    rng, rows = random.Random(SEED), []
    for camp, (label, ch, snd, subs, cores, asks) in C.items():
        combos = sorted({(rng.choice(GREET), rng.choice(subs), rng.choice(cores), rng.choice(asks), rng.choice(CLOSE))
                         for _ in range(150)})
        for g, s, c, a, cl in rng.sample(combos, 40):
            kind = rng.choices(["short", "odd", "internal"], [.5, .4, .1] if label else [.1, .1, .8])[0]
            fill = dict(name=rng.choice(NAMES), n=rng.randint(2, 9), url=rng.choice(URLS[kind]))
            rows.append(dict(channel=ch, subject_or_title=s.format(**fill), body_text=" ".join([g, c, a, cl]).format(**fill),
                             sender_domain_type=snd, label=label, campaign_group=camp,
                             collected_date=f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}", provenance="locally authored synthetic"))
    for r in rng.sample(rows, 30):  # inject near-identical copies so FR-01 has something to remove
        rows.append({**r, "body_text": NAME_RE.sub(rng.choice(NAMES), r["body_text"], count=1)})
    df = pd.DataFrame(rows)
    df.insert(0, "message_id", [f"M{i:04d}" for i in range(len(df))])
    df["sanitized_url_features"] = df.body_text.map(url_feat)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def textify(df):
    return df.subject_or_title + " " + df.body_text + " url_" + df.sanitized_url_features


def dedup(df, thr=0.95):
    norm = (df.subject_or_title + " " + df.body_text).str.replace(NAME_RE, "", regex=True).str.replace(r"\d+", "0", regex=True)
    S = linear_kernel(TfidfVectorizer().fit_transform(norm))
    keep, dropped = [], []
    for i in range(len(df)):
        j = next((k for k in keep if S[i, k] >= thr), None)
        if j is None: keep.append(i)
        else: dropped.append([df.message_id[i], df.message_id[j]])
    (ART / "dedup_proof.json").write_text(json.dumps(dropped))
    return df.iloc[keep].reset_index(drop=True), len(dropped)


# ---------- model 1: explainable rules ----------
CUES = {
 "Credential or code request": (r"\b(verify|confirm|re-?enter|sign in|enter the code|send us the code|card details|bank login)\b", .4,
                                "Asks you to verify, sign in or share a code or card details."),
 "Urgency or threat": (r"\b(suspend\w*|locked|expires? today|final notice|within \d+ hours?|immediately|before midnight|about to expire)\b", .3,
                       "Pressures you to act fast or warns of lost access."),
 "Money, reward or gift offer": (r"\b(gift cards?|prize|reward|wire payment|penalty|overdue|vouchers?)\b", .35,
                                 "Mentions payments, rewards or vouchers."),
 "Shortened or look-alike link": (r"bit\[?\.\]?ly|\bt\[?\.\]?co/|secure-login|acme-portal", .4,
                                  "The link hides or imitates its real destination."),
 "Secrecy or authority pressure": (r"\b(keep this between us|at your desk|discreet\w*|can't talk)\b", .45,
                                   "Asks for discretion or a quick favour from someone senior."),
}


def rules(text):
    hits = [(n, w, why) for n, (rx, w, why) in CUES.items() if re.search(rx, text, re.I)]
    return float(1 - np.prod([1 - w for _, w, _ in hits])), [{"cue": n, "why": why} for n, _, why in hits]


# ---------- models 2 and 3 ----------
def build():
    vec = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    return {"tfidf_logreg": make_pipeline(vec(), LogisticRegression(C=2, max_iter=1000, class_weight="balanced")),
            "random_forest": make_pipeline(vec(), RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=SEED))}


def scorer(name, models):
    if name == "rules": return lambda X: np.array([rules(x)[0] for x in X])
    return lambda X: models[name].predict_proba(list(X))[:, 1]


def metrics(y, p):
    y, yh = np.asarray(y), (np.asarray(p) >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(f1=round(float(f1_score(y, yh, zero_division=0)), 3), recall=round(tp / max(tp + fn, 1), 3),
                fpr=round(fp / max(fp + tn, 1), 3), cm=[[int(tn), int(fp)], [int(fn), int(tp)]])


PERTURB = {  # FR-05 / AT-03: controlled, harmless text perturbations
    "defang_variant": lambda t: t.replace("[.]", "(dot)").replace("hxxp", "hXXp"),
    "punctuation": lambda t: t.replace(".", "...").replace(",", " ,") + " !!",
    "hyphenated_words": lambda t: re.sub("(?i)password", "pass-word", t),
}


def train():
    raw = make_corpus()
    raw.to_csv(ART / "raw_corpus.csv", index=False)  # unchanged raw records (T2)
    df, removed = dedup(raw)
    df["text"] = textify(df)
    te_mask = df.campaign_group.isin(HOLDOUT)
    tr, te = df[~te_mask], df[te_mask]
    rs_tr, rs_te = train_test_split(df, test_size=.25, stratify=df.label, random_state=SEED)
    models = build()
    report = {"corpus": dict(raw=len(raw), duplicates_removed=removed, after_dedup=len(df), train=len(tr), test=len(te),
                             holdout_campaigns=HOLDOUT), "models": {}}
    for name in ["rules", *models]:
        if name in models: models[name].fit(rs_tr.text, rs_tr.label)
        fn, e = scorer(name, models), {}
        e["random_split"] = metrics(rs_te.label, fn(rs_te.text))  # shows the inflated, leaky number
        if name in models: models[name].fit(tr.text, tr.label)
        e["campaign_holdout"] = metrics(te.label, fn(te.text))
        urg = te[(te.label == 0) & te.text.str.contains(URGENT, case=False, regex=True)]
        e["legit_urgent"] = dict(n=len(urg), fpr=round(float((fn(urg.text) >= .5).mean()), 3) if len(urg) else None)
        p0, rob = fn(te.text), {}
        for k, f in PERTURB.items():
            p1 = fn(te.text.map(f))
            rob[k] = dict(mean_abs_drift=round(float(np.abs(p1 - p0).mean()), 3), flip_rate=round(float(((p1 >= .5) != (p0 >= .5)).mean()), 3))
        report["models"][name] = {**e, "robustness": rob}
    p = scorer("tfidf_logreg", models)(te.text)
    wrong = (p >= .5) != te.label.to_numpy().astype(bool)
    report["error_examples"] = te[wrong].assign(p=p[wrong].round(2)).head(3)[["message_id", "campaign_group", "label", "p", "body_text"]].to_dict("records")
    joblib.dump({"models": models, "version": "phishshift-1.0"}, ART / "models.joblib")
    (ART / "results.json").write_text(json.dumps(report, indent=2))
    return report


# ---------- inference ----------
def sanitize(t):  # links become inert defanged text; they are never requested
    return re.sub(r"https?://\S+", lambda m: m.group().replace("http", "hxxp", 1).replace(".", "[.]"), t)


_bundle = None
def bundle():
    global _bundle
    if _bundle is None:
        if not (ART / "models.joblib").exists(): train()
        _bundle = joblib.load(ART / "models.joblib")
    return _bundle


def analyse(body, subject=""):
    body = sanitize(body[:5000])
    text, b = f"{subject} {body} url_{url_feat(body)}", bundle()
    p_rules, cues = rules(text)
    scores = {"rules": p_rules, **{n: float(m.predict_proba([text])[0, 1]) for n, m in b["models"].items()}}
    p = scores["tfidf_logreg"]
    unsure = ABSTAIN_BAND[0] < p < ABSTAIN_BAND[1] or abs(p - scores["random_forest"]) > .5
    return dict(verdict="review" if unsure else "phishing" if p >= .5 else "legitimate", risk=round(p, 3),
                scores={k: round(v, 3) for k, v in scores.items()}, cues=cues,
                abstain_band=ABSTAIN_BAND, model_version=b["version"], disclaimer=DISCLAIMER)


if __name__ == "__main__":
    print(json.dumps(train()["corpus"], indent=2))
