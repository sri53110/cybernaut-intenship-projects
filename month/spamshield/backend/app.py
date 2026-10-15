```python
"""SpamShield API — Flask application for Vercel."""

import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Project paths
ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT / "models")))
ML_DIR = ROOT / "ml"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import preprocess  # noqa: E402

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Check required files
required_files = [
    MODEL_DIR / "spam_models.joblib",
    MODEL_DIR / "metrics.json",
    MODEL_DIR / "keywords.json",
    FRONTEND / "index.html",
]

missing = [str(path) for path in required_files if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "Required SpamShield files are missing: " + ", ".join(missing)
    )

# Load trained models and metadata
_bundle = joblib.load(MODEL_DIR / "spam_models.joblib")
MODELS = _bundle["models"]
DEFAULT = _bundle["default"]
LABELS = _bundle["labels"]

METRICS = json.loads(
    (MODEL_DIR / "metrics.json").read_text(encoding="utf-8")
)
KEYWORDS = json.loads(
    (MODEL_DIR / "keywords.json").read_text(encoding="utf-8")
)

_lr = MODELS["logistic_regression"]
_vec = _lr.named_steps["tfidf"]
_coef = _lr.named_steps["clf"].coef_[0]
_vocab_inv = np.array(_vec.get_feature_names_out())

_hits = defaultdict(deque)
STATS = {"started": time.time(), "scans": 0, "spam": 0}

MAX_CHARS = 20_000
MAX_BATCH = 100
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "120"))
RATE_WINDOW = 60


@app.before_request
def limit():
    if not request.path.startswith("/api/"):
        return None

    ip = (
        request.headers.get("X-Forwarded-For", request.remote_addr) or "?"
    ).split(",")[0].strip()

    now = time.time()
    queue = _hits[ip]

    while queue and now - queue[0] > RATE_WINDOW:
        queue.popleft()

    if len(queue) >= RATE_LIMIT:
        return jsonify(error="Rate limit exceeded. Try again in a minute."), 429

    queue.append(now)
    return None


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def verdict(probability):
    if probability >= 0.85:
        return "spam", "high"
    if probability >= 0.5:
        return "spam", "medium"
    if probability >= 0.2:
        return "ham", "uncertain"
    return "ham", "high"


def explain(text, top=8):
    row = _vec.transform([text])
    indices = row.indices
    values = row.data
    contributions = values * _coef[indices]
    order = np.argsort(contributions)

    spam_terms = [
        {
            "term": str(_vocab_inv[indices[i]]),
            "score": round(float(contributions[i]), 4),
        }
        for i in order[::-1]
        if contributions[i] > 0
    ][:top]

    ham_terms = [
        {
            "term": str(_vocab_inv[indices[i]]),
            "score": round(float(contributions[i]), 4),
        }
        for i in order
        if contributions[i] < 0
    ][:top]

    return spam_terms, ham_terms


def build_text(payload):
    subject = str(payload.get("subject") or "").strip()
    body = str(payload.get("text") or payload.get("body") or "").strip()
    return f"{subject}\n{body}".strip() if subject else body


def pick_model(name):
    key = name or DEFAULT
    if key not in MODELS:
        raise ValueError(
            f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}"
        )
    return key


def score_many(texts, key):
    probabilities = MODELS[key].predict_proba(texts)[:, 1]
    STATS["scans"] += len(texts)
    results = []

    for text, probability in zip(texts, probabilities):
        label, confidence = verdict(float(probability))
        if label == "spam":
            STATS["spam"] += 1

        results.append({
            "label": label,
            "spam_probability": round(float(probability), 4),
            "confidence": confidence,
            "chars": len(text),
        })

    return results


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        default_model=DEFAULT,
        uptime_s=int(time.time() - STATS["started"]),
    )


@app.get("/api/models")
def models():
    return jsonify(
        default=DEFAULT,
        models=[{"id": key, "name": LABELS[key]} for key in MODELS],
    )


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.get("/api/keywords")
def keywords():
    return jsonify(KEYWORDS)


@app.get("/api/stats")
def stats():
    return jsonify(
        scans=STATS["scans"],
        spam=STATS["spam"],
        uptime_s=int(time.time() - STATS["started"]),
    )


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = build_text(payload)

    if not text:
        return jsonify(error="Provide a non-empty 'text' field."), 400

    if len(text) > MAX_CHARS:
        return jsonify(error=f"Message too long (max {MAX_CHARS} characters)."), 413

    try:
        key = pick_model(payload.get("model"))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    start = time.perf_counter()
    result = score_many([text], key)[0]
    spam_terms, ham_terms = explain(text)

    result.update(
        model=key,
        model_name=LABELS[key],
        spam_signals=spam_terms,
        ham_signals=ham_terms,
        votes={
            name: round(float(model.predict_proba([text])[0, 1]), 4)
            for name, model in MODELS.items()
        },
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )

    return jsonify(result)


@app.post("/api/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")

    if not isinstance(texts, list) or not texts:
        return jsonify(error="Provide 'texts' as a non-empty list."), 400

    if len(texts) > MAX_BATCH:
        return jsonify(error=f"Max {MAX_BATCH} messages per batch."), 413

    texts = [str(text)[:MAX_CHARS] for text in texts]

    try:
        key = pick_model(payload.get("model"))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    results = score_many(texts, key)

    for result, text in zip(results, texts):
        result["preview"] = re.sub(r"\s+", " ", text)[:90]

    return jsonify(
        model=key,
        results=results,
        summary={
            "total": len(results),
            "spam": sum(r["label"] == "spam" for r in results),
        },
    )


# Homepage
@app.get("/")
def index():
    return send_from_directory(str(FRONTEND), "index.html")


# Frontend assets such as CSS, JavaScript and images
@app.get("/<path:path>")
def assets(path):
    if path.startswith("api/"):
        return jsonify(error="Not found"), 404

    file_path = (FRONTEND / path).resolve()

    # Prevent access outside the frontend directory
    if not file_path.is_relative_to(FRONTEND.resolve()):
        return jsonify(error="Not found"), 404

    if not file_path.is_file():
        return jsonify(error="Not found"), 404

    return send_from_directory(str(FRONTEND), path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
    )
```
