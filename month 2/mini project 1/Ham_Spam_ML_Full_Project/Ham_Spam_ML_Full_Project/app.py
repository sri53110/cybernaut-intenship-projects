from flask import Flask, render_template, request
from ml_model import load_or_train_model, predict_message

app = Flask(__name__)

MODEL_PATH = "spam_classifier.joblib"
DATASET_CANDIDATES = ["SMSSpamCollection", "spam.csv", "spam.tsv"]

model, vectorizer, dataset_status = load_or_train_model(MODEL_PATH, DATASET_CANDIDATES)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            result = {"label": "Please enter a message.", "kind": "warning", "confidence": None}
        elif model is None:
            result = {
                "label": "Model is not ready. Add a supported labeled dataset and restart the app.",
                "kind": "warning",
                "confidence": None,
            }
        else:
            prediction, confidence = predict_message(message, model, vectorizer)
            if prediction == "spam":
                result = {"label": "SPAM", "kind": "spam", "confidence": confidence}
            else:
                result = {"label": "HAM (NOT SPAM)", "kind": "ham", "confidence": confidence}

    return render_template(
        "index.html",
        message=message,
        result=result,
        dataset_status=dataset_status,
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
