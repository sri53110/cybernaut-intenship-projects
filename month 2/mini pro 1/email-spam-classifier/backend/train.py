"""Train Naive Bayes, Logistic Regression and SVM, compare them, and save everything.

Usage:  python backend/train.py [path/to/spam.csv]
Dataset: UCI SMS Spam Collection (Kaggle: uciml/sms-spam-collection-dataset)
         columns v1 (ham/spam) + v2 (text), latin-1 encoded. A CSV with
         `label,text` columns (e.g. SpamAssassin export) also works.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import clean

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Dataset not found at {path}. Download spam.csv from Kaggle "
                 "(uciml/sms-spam-collection-dataset) and put it in /data.")
    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna().drop_duplicates()
    df["y"] = (df["label"].str.lower().str.strip() == "spam").astype(int)
    df["clean"] = df["text"].map(clean)
    return df


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "spam.csv"
    df = load(path)
    print(f"Loaded {len(df)} messages ({df.y.sum()} spam, {(1 - df.y).sum()} ham)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["y"], test_size=0.2, stratify=df["y"], random_state=42)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)

    models = {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(C=10, max_iter=1000,
                                                  class_weight="balanced"),
        "svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
    }

    metrics, matrices = {}, {}
    for name, m in models.items():
        m.fit(Xtr, y_tr)
        pred = m.predict(Xte)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
        }
        matrices[name] = confusion_matrix(y_te, pred)
        print(f"{name:20s}", metrics[name])

    best = max(metrics, key=lambda k: metrics[k]["f1"])

    # Global spam / ham indicators from Logistic Regression coefficients
    lr = models["logistic_regression"]
    words = vec.get_feature_names_out()
    order = np.argsort(lr.coef_[0])
    keywords = {
        "spam": [{"word": words[i], "weight": round(float(lr.coef_[0][i]), 3)}
                 for i in order[::-1][:15]],
        "ham": [{"word": words[i], "weight": round(float(-lr.coef_[0][i]), 3)}
                for i in order[:15]],
    }

    joblib.dump({"vectorizer": vec, "models": models, "best": best},
                MODELS / "spam_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(
        {"best": best, "metrics": metrics, "keywords": keywords,
         "dataset": {"total": int(len(df)), "spam": int(df.y.sum())}}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (name, cm) in zip(axes, matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    plt.tight_layout()
    plt.savefig(MODELS / "confusion_matrices.png", dpi=150)
    print(f"Best model: {best}. Saved to {MODELS}")


if __name__ == "__main__":
    main()
