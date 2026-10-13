# Sortbox: Email Spam Classifier

A full-stack machine learning app that classifies emails as **spam** or **ham**.

- **Backend:** Python, Flask, scikit-learn (TF-IDF + Naive Bayes / Logistic Regression / SVM)
- **Frontend:** single-page HTML/CSS/JS served by Flask (no build step)
- **Extras:** per-message word explanations, model comparison table, spam keyword charts, confusion-matrix image

## Project structure
```
email-spam-classifier/
├── backend/   app.py, train.py, preprocess.py, requirements.txt
├── frontend/  index.html
├── data/      spam.csv (you add this)
├── models/    created by train.py
└── Procfile   for Render / Railway / Heroku
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```
1. Download `spam.csv` from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset and put it in `data/`.
2. Train: `python backend/train.py`
3. Serve: `python backend/app.py` and open http://localhost:5000

## API
| Method | Route | Purpose |
|---|---|---|
| POST | `/api/predict` | Body `{"text": "...", "model": "svm"}` returns label, spam probability, top signal words |
| GET | `/api/metrics` | Model scores, top keywords, dataset size |
| GET | `/api/health` | Health check |

## Push to GitHub
```bash
git init && git add . && git commit -m "Email spam classifier"
git branch -M main
git remote add origin https://github.com/<you>/email-spam-classifier.git
git push -u origin main
```

## Deploy
Run `python backend/train.py` before deploying, and commit `models/spam_model.joblib` (remove it from `.gitignore`) or train in your build command. Start command: `gunicorn --chdir backend app:app`.

## Extending
Swap in the SpamAssassin corpus (CSV with `label,text` columns), or retrain on phishing data to catch scams and ads.
