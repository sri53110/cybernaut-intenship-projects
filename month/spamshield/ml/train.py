"""Train Naive Bayes, Logistic Regression and SVM spam classifiers.

Usage:
    python ml/train.py                              # uses data/sample_sms.csv
    python ml/train.py --data data/spam.csv         # UCI SMS Spam Collection (v1,v2 columns)
    python ml/train.py --data my.csv --label-col category --text-col message

Outputs (in models/): spam_models.joblib, metrics.json, keywords.json
"""
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load(path, label_col, text_col):
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    cols = {c.lower(): c for c in df.columns}
    if label_col is None:
        label_col = next((cols[c] for c in ("label", "v1", "category", "class", "target") if c in cols), None)
    if text_col is None:
        text_col = next((cols[c] for c in ("text", "v2", "message", "email", "body", "sms") if c in cols), None)
    if label_col is None or text_col is None:
        sys.exit(f"Could not detect columns in {list(df.columns)}; pass --label-col/--text-col")
    df = df[[label_col, text_col]].dropna()
    df.columns = ["label", "text"]
    lab = df["label"].astype(str).str.strip().str.lower()
    df["y"] = lab.isin({"spam", "1", "true"}).astype(int)
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"loaded {before} rows, {len(df)} after de-duplication; spam ratio {df.y.mean():.3f}")
    return df


def vectorizer():
    return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, stop_words="english", strip_accents="unicode")


def make_models():
    return {
        "naive_bayes": Pipeline([("tfidf", vectorizer()), ("clf", MultinomialNB(alpha=0.1))]),
        "logistic_regression": Pipeline([("tfidf", vectorizer()),
                                         ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))]),
        # LinearSVC has no predict_proba, so calibrate it to get probabilities
        "svm": Pipeline([("tfidf", vectorizer()),
                         ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=5))]),
    }


LABELS = {"naive_bayes": "Naive Bayes", "logistic_regression": "Logistic Regression", "svm": "Support Vector Machine"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sample_sms.csv"))
    ap.add_argument("--label-col"); ap.add_argument("--text-col")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default=str(ROOT / "models"))
    a = ap.parse_args()

    df = load(a.data, a.label_col, a.text_col)
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.y, test_size=a.test_size, stratify=df.y, random_state=SEED)
    print(f"train {len(Xtr)} / test {len(Xte)}")

    models, results = {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for key, pipe in make_models().items():
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        cv_f1 = cross_val_score(make_models()[key], Xtr, ytr, cv=cv, scoring="f1")
        results[key] = {
            "name": LABELS[key],
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "false_positive_rate": fp / max(fp + tn, 1),
            "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "train_seconds": round(time.time() - t0, 2),
        }
        models[key] = pipe
        r = results[key]
        print(f"{LABELS[key]:24s} acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
              f"f1={r['f1']:.4f} auc={r['roc_auc']:.4f} fpr={r['false_positive_rate']:.4f}")

    # Pick the default model: spam filters must avoid blocking real mail, so weight precision
    # higher than recall (F0.5-style), breaking ties on F1.
    def score(r):
        p, rc = r["precision"], r["recall"]
        f05 = 1.25 * p * rc / (0.25 * p + rc) if (0.25 * p + rc) else 0
        return (f05, r["f1"])
    best = max(results, key=lambda k: score(results[k]))
    print("default model:", best)

    # Keyword insight from the logistic regression weights
    lr = models["logistic_regression"]
    names = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    coef = lr.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    kw = {"spam": [{"term": names[i], "weight": float(coef[i])} for i in order[::-1][:25]],
          "ham": [{"term": names[i], "weight": float(coef[i])} for i in order[:25]]}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "default": best, "labels": LABELS}, out / "spam_models.joblib", compress=3)
    meta = {"dataset": Path(a.data).name, "rows": int(len(df)), "spam_ratio": float(df.y.mean()),
            "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "default_model": best,
            "is_sample_data": Path(a.data).name == "sample_sms.csv", "models": results}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    (out / "keywords.json").write_text(json.dumps(kw, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
