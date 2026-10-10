import csv
import math
import os
import re
from collections import Counter

LABEL_MAP = {
    "spam": "spam",
    "ham": "ham",
    "not spam": "ham",
    "legitimate": "ham",
    "non-spam": "ham",
    "non spam": "ham",
}

FALLBACK_ROWS = [
    ("ham", "Hi John, can we meet tomorrow at 5?"),
    ("ham", "Thanks for your help with the project today."),
    ("ham", "I will call you after lunch."),
    ("ham", "The meeting is moved to Friday morning."),
    ("ham", "Please review the files before the deadline."),
    ("spam", "Free prize! Claim your cash reward now"),
    ("spam", "Win a vacation and get money back immediately"),
    ("spam", "Huge offer, act now and claim your bonus"),
    ("spam", "Urgent: you have won a free gift card"),
    ("spam", "Limited time offer, cash reward waiting"),
]


def _tokenize(text):
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _normalize_label(raw_label):
    return LABEL_MAP.get(str(raw_label).strip().lower(), None)


def _read_dataset(path):
    basename = os.path.basename(path).lower()
    rows = []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        if basename == "smsspamcollection" or path.lower().endswith((".tsv", ".txt")):
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if not row or len(row) < 2:
                    continue
                label = _normalize_label(row[0])
                message = row[1].strip()
                if label and message:
                    rows.append((label, message))
        else:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError("Dataset file is missing a header row.")
            field_map = {str(name).strip().lower(): name for name in reader.fieldnames}
            label_field = next((field_map[k] for k in ["v1", "label", "category", "class", "type"] if k in field_map), None)
            message_field = next((field_map[k] for k in ["v2", "message", "text", "sms", "email"] if k in field_map), None)
            if label_field is None or message_field is None:
                raise ValueError("Dataset must have a label and message column.")
            for row in reader:
                label = _normalize_label(row.get(label_field, ""))
                message = str(row.get(message_field, "") or "").strip()
                if label and message:
                    rows.append((label, message))

    if len(rows) < 10:
        raise ValueError("Dataset has too few valid rows.")

    seen = {"spam": 0, "ham": 0}
    cleaned = []
    for label, message in rows:
        if label not in seen:
            continue
        seen[label] += 1
        cleaned.append((label, message))

    if seen["spam"] == 0 or seen["ham"] == 0:
        raise ValueError("Dataset must contain both spam and ham labels.")

    return cleaned


def _train_naive_bayes(rows):
    spam_counts = Counter()
    ham_counts = Counter()
    spam_total = 0
    ham_total = 0

    for label, message in rows:
        tokens = _tokenize(message)
        if not tokens:
            continue
        if label == "spam":
            spam_counts.update(tokens)
            spam_total += len(tokens)
        else:
            ham_counts.update(tokens)
            ham_total += len(tokens)

    vocab = set(spam_counts.keys()) | set(ham_counts.keys())
    spam_prior = math.log((sum(1 for label, _ in rows if label == "spam") + 1) / (len(rows) + 2))
    ham_prior = math.log((sum(1 for label, _ in rows if label == "ham") + 1) / (len(rows) + 2))

    return {
        "spam_counts": spam_counts,
        "ham_counts": ham_counts,
        "spam_total": spam_total,
        "ham_total": ham_total,
        "vocab_size": max(len(vocab), 1),
        "spam_prior": spam_prior,
        "ham_prior": ham_prior,
    }


def load_or_train_model(model_path, dataset_candidates):
    chosen = next((path for path in dataset_candidates if os.path.isfile(path)), None)
    if chosen:
        try:
            rows = _read_dataset(chosen)
            status = f"Model trained using {os.path.basename(chosen)} with {len(rows):,} labeled messages."
            return _train_naive_bayes(rows), None, status
        except Exception as exc:
            rows = FALLBACK_ROWS
            status = f"Could not read dataset: {exc}. Using the built-in fallback demo dataset instead."
            return _train_naive_bayes(rows), None, status

    rows = FALLBACK_ROWS
    status = "No dataset file was found. Using the built-in demo dataset so the app can still run."
    return _train_naive_bayes(rows), None, status


def predict_message(message, model, vectorizer):
    if model is None:
        return "ham", 0.0

    tokens = _tokenize(message)
    if not tokens:
        return "ham", 0.0

    spam_counts = model["spam_counts"]
    ham_counts = model["ham_counts"]
    spam_total = model["spam_total"]
    ham_total = model["ham_total"]
    vocab_size = model["vocab_size"]
    spam_prior = model["spam_prior"]
    ham_prior = model["ham_prior"]

    spam_score = spam_prior
    ham_score = ham_prior
    for token in tokens:
        spam_score += math.log((spam_counts.get(token, 0) + 1) / (spam_total + vocab_size + 1))
        ham_score += math.log((ham_counts.get(token, 0) + 1) / (ham_total + vocab_size + 1))

    if spam_score >= ham_score:
        label = "spam"
        confidence = 1 / (1 + math.exp(ham_score - spam_score))
    else:
        label = "ham"
        confidence = 1 / (1 + math.exp(spam_score - ham_score))

    return label, float(confidence)
