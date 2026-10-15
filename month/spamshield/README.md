# SpamShield: Email Spam Classifier with a web app

Machine-learning spam detector (TF-IDF + Naive Bayes / Logistic Regression / SVM) with a Flask API
and a single-page web frontend. Paste a message, get a verdict, a spam probability, and the exact
words that drove the decision.

## Project layout

```
spamshield/
├── ml/
│   ├── preprocess.py        # text cleaning (shared by training + API)
│   ├── train.py             # trains 3 models, evaluates, saves artifacts
│   └── make_sample_data.py  # synthetic demo dataset (smoke test only)
├── backend/app.py           # Flask API + serves the frontend
├── frontend/                # index.html, styles.css, app.js (no build step)
├── models/                  # spam_models.joblib, metrics.json, keywords.json
├── data/                    # sample_sms.csv (replace with a real dataset)
├── tests/test_api.py
├── Dockerfile
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ml/train.py            # trains on data/sample_sms.csv, writes models/
python backend/app.py         # http://localhost:8000
```

## Train on a real dataset (do this before showing real numbers)

The bundled `data/sample_sms.csv` is synthetic and only proves the pipeline works.

1. Download the [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle and save it as `data/spam.csv`.
2. Train: `python ml/train.py --data data/spam.csv` (columns `v1`/`v2` are auto-detected).
3. For SpamAssassin or other CSVs: `python ml/train.py --data mail.csv --label-col category --text-col message`.
4. Restart the server. The "sample data" notice in the Accuracy section disappears automatically.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/predict` | `{"subject"?, "text", "model"?}` | label, spam_probability, confidence, spam/ham signals, per-model votes, latency |
| POST | `/api/predict/batch` | `{"texts": [...], "model"?}` (max 100) | per-message results + summary |
| GET | `/api/models` `/api/metrics` `/api/keywords` `/api/stats` `/api/health` | | |

`model` is one of `naive_bayes`, `logistic_regression`, `svm`.
Inputs are capped at 20,000 characters, and requests at 120 per minute per IP (`RATE_LIMIT` env var).

## How it works

1. **Clean**: strip HTML; map URLs, e-mails, phone numbers, money and digits to placeholder tokens; flag ALL CAPS and repeated `!`.
2. **Vectorize**: TF-IDF over unigrams and bigrams, sublinear tf, English stop words.
3. **Classify**: Multinomial Naive Bayes, Logistic Regression, and a calibrated LinearSVC.
4. **Choose the default**: the model with the best precision-weighted F-score, because blocking real mail is worse than missing some spam.
5. **Explain**: per-message term contributions (weight x TF-IDF) from the logistic regression model.

## Test

```bash
python tests/test_api.py      # or: pytest
```

## Deploy

```bash
docker build -t spamshield . && docker run -p 8000:8000 spamshield
```
Or `gunicorn -w 2 -b 0.0.0.0:8000 backend.app:app` behind any reverse proxy. Render, Railway and Fly.io all work with the Dockerfile.

## Ideas to extend

Phishing and scam detection, a Gmail/Outlook add-in calling `/api/predict`, user feedback to retrain, a transformer model (DistilBERT) as a fourth option.
