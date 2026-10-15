"""SpamShield API (Flask). Serves the trained models and the static frontend.

Run:  python backend/app.py        (dev)
      gunicorn -w 2 -b 0.0.0.0:8000 backend.app:app   (prod)
"""
import json, os, re, sys, time
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
import preprocess  # noqa: E402,F401  (needed so joblib can unpickle clean_text)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
FRONTEND = ROOT / "frontend"
MAX_CHARS, MAX_BATCH = 20_000, 100
RATE_LIMIT, RATE_WINDOW = int(os.environ.get("RATE_LIMIT", 120)), 60  # requests / seconds / IP

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

_bundle = joblib.load(MODEL_DIR / "spam_models.joblib")
MODELS, DEFAULT, LABELS = _bundle["models"], _bundle["default"], _bundle["labels"]
METRICS = json.loads((MODEL_DIR / "metrics.json").read_text())
KEYWORDS = json.loads((MODEL_DIR / "keywords.json").read_text())

# Logistic regression doubles as the explainer: weight x tf-idf value per term.
_lr = MODELS["logistic_regression"]
_vec, _coef = _lr.named_steps["tfidf"], _lr.named_steps["clf"].coef_[0]
_vocab_inv = np.array(_vec.get_feature_names_out())

_hits = defaultdict(deque)
STATS = {"started": time.time(), "scans": 0, "spam": 0}


@app.before_request
def limit():
    if not request.path.startswith("/api/"):
        return None
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr) or "?").split(",")[0].strip()
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return jsonify(error="Rate limit exceeded. Try again in a minute."), 429
    q.append(now)


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


def verdict(p):
    if p >= 0.85: return "spam", "high"
    if p >= 0.5: return "spam", "medium"
    if p >= 0.2: return "ham", "uncertain"
    return "ham", "high"


def explain(text, top=8):
    """Top terms pushing the message toward spam (and toward ham)."""
    row = _vec.transform([text])
    idx, val = row.indices, row.data
    contrib = val * _coef[idx]
    order = np.argsort(contrib)
    spam_terms = [{"term": _vocab_inv[idx[i]], "score": round(float(contrib[i]), 4)} for i in order[::-1] if contrib[i] > 0][:top]
    ham_terms = [{"term": _vocab_inv[idx[i]], "score": round(float(contrib[i]), 4)} for i in order if contrib[i] < 0][:top]
    return spam_terms, ham_terms


def build_text(payload):
    subject = (payload.get("subject") or "").strip()
    body = (payload.get("text") or payload.get("body") or "").strip()
    return f"{subject}\n{body}".strip() if subject else body


def pick_model(name):
    key = name or DEFAULT
    if key not in MODELS:
        raise ValueError(f"Unknown model '{key}'. Choose one of: {', '.join(MODELS)}")
    return key


def score_many(texts, key):
    probs = MODELS[key].predict_proba(texts)[:, 1]
    STATS["scans"] += len(texts)
    out = []
    for t, p in zip(texts, probs):
        label, conf = verdict(float(p))
        STATS["spam"] += label == "spam"
        out.append({"label": label, "spam_probability": round(float(p), 4), "confidence": conf, "chars": len(t)})
    return out


@app.get("/api/health")
def health():
    return jsonify(status="ok", default_model=DEFAULT, uptime_s=int(time.time() - STATS["started"]))


@app.get("/api/models")
def models():
    return jsonify(default=DEFAULT, models=[{"id": k, "name": LABELS[k]} for k in MODELS])


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.get("/api/keywords")
def keywords():
    return jsonify(KEYWORDS)


@app.get("/api/stats")
def stats():
    return jsonify(scans=STATS["scans"], spam=STATS["spam"], uptime_s=int(time.time() - STATS["started"]))


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
    except ValueError as e:
        return jsonify(error=str(e)), 400
    t0 = time.perf_counter()
    res = score_many([text], key)[0]
    spam_terms, ham_terms = explain(text)
    res.update(model=key, model_name=LABELS[key], spam_signals=spam_terms, ham_signals=ham_terms,
               # every model's opinion, so the UI can show agreement
               votes={k: round(float(m.predict_proba([text])[0, 1]), 4) for k, m in MODELS.items()},
               latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return jsonify(res)


@app.post("/api/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify(error="Provide 'texts' as a non-empty list."), 400
    if len(texts) > MAX_BATCH:
        return jsonify(error=f"Max {MAX_BATCH} messages per batch."), 413
    texts = [str(t)[:MAX_CHARS] for t in texts]
    try:
        key = pick_model(payload.get("model"))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    results = score_many(texts, key)
    for r, t in zip(results, texts):
        r["preview"] = re.sub(r"\s+", " ", t)[:90]
    return jsonify(model=key, results=results,
                   summary={"total": len(results), "spam": sum(r["label"] == "spam" for r in results)})


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/<path:path>")
def assets(path):
    if path.startswith("api/"):
        return jsonify(error="Not found"), 404
    return send_from_directory(FRONTEND, path)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)), debug=bool(os.environ.get("DEBUG")))
