"""SafeTriage API. Run: python app.py  ->  http://127.0.0.1:5000"""
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from pipeline import ART, analyse, bundle

FRONT = Path(__file__).parent.parent / "frontend"
app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(FRONT, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_version=bundle()["version"])


@app.post("/api/analyze")
def analyze():
    d = request.get_json(silent=True) or {}
    body, subject = str(d.get("body", "")).strip(), str(d.get("subject", ""))[:200]
    if len(body) < 10:
        return jsonify(error="Paste at least 10 characters of message text."), 400
    return jsonify(analyse(body, subject))  # text is scored only; links are never opened (NFR-01)


@app.get("/api/report")
def report():
    bundle()  # trains on first run if artifacts are missing
    return send_from_directory(ART, "results.json")


@app.errorhandler(Exception)
def fail(e):
    return jsonify(error="The scanner hit a problem. Check the server log."), 500


if __name__ == "__main__":
    bundle()
    app.run(host="127.0.0.1", port=5000)
