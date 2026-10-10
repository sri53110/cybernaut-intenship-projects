
from pathlib import Path

import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"

# Load the trained model.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "error": "Please enter a message."
        }), 400

    if len(message) > 20000:
        return jsonify({
            "error": "Message must be 20,000 characters or fewer."
        }), 400

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = round(float(max(probabilities)) * 100, 1)

    label = "Spam" if prediction == "spam" else "Ham"

    return jsonify({
        "label": label,
        "confidence": confidence,
        "message": f"This message was classified as {label}."
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
