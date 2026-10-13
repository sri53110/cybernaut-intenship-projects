"""Flask API + static frontend for the spam classifier."""
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

from preprocess import clean

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "spam_model.joblib"
METRICS_FILE = ROOT / "models" / "metrics.json"
MAX_CHARS = 10_000

app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
bundle = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


def need_model():
    return jsonify(error="Model not trained yet. Run: python backend/train.py"), 503


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_loaded=bundle is not None)


@app.get("/api/metrics")
def metrics():
    if not METRICS_FILE.exists():
        return need_model()
    return jsonify(json.loads(METRICS_FILE.read_text()))


@app.post("/api/predict")
def predict():
    if bundle is None:
        return need_model()
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Paste the email text to check."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Email is too long. Limit is {MAX_CHARS} characters."), 400

    name = data.get("model") or bundle["best"]
    if name not in bundle["models"]:
        return jsonify(error=f"Unknown model '{name}'."), 400

    vec = bundle["vectorizer"]
    x = vec.transform([clean(text)])
    spam_prob = float(bundle["models"][name].predict_proba(x)[0][1])

    # Explain with Logistic Regression: contribution = tf-idf value x coefficient
    lr = bundle["models"]["logistic_regression"]
    words = vec.get_feature_names_out()
    row = x.tocoo()
    contrib = [(words[c], float(v * lr.coef_[0][c])) for c, v in zip(row.col, row.data)]
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)

    return jsonify(
        label="spam" if spam_prob >= 0.5 else "ham",
        spam_probability=round(spam_prob, 4),
        model=name,
        signals=[{"word": w, "score": round(s, 3)} for w, s in contrib[:8]],
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
