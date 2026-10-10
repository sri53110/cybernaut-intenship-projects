
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_model.joblib"


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset missing. Add data/spam.csv before training."
        )

    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Support the UCI dataset format and a standard label,text CSV.
    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "label", "v2": "text"})
    elif not {"label", "text"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain v1,v2 or label,text columns."
        )

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"] != ""]

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=30000
            )
        ),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("\nEvaluation report:")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()