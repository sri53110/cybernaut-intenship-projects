# Ham / Spam Classifier — Full-Stack ML Project

This project contains:
- **Frontend:** HTML, CSS, small JavaScript helper for example messages
- **Backend:** Python + Flask
- **Machine learning:** scikit-learn TF-IDF + Multinomial Naive Bayes
- **Dataset:** UCI SMS Spam Collection or a compatible labeled `spam.csv`

> The common UCI dataset is SMS message text, not full email headers. It is suitable for a learning prototype that classifies message text.

## Step 1 — Install Python
Use Python 3.10+ and VS Code.

## Step 2 — Open the folder
Extract the ZIP and open the `Ham_Spam_ML_Full_Project` folder in VS Code.

## Step 3 — Create a virtual environment (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked, open Command Prompt and run:
```bat
.venv\Scripts\activate.bat
```

## Step 4 — Install packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 5 — Download dataset
Download the UCI SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Extract it and copy the file named `SMSSpamCollection` into the project root (same folder as `app.py`).
Alternatively, add a compatible `spam.csv` with columns like `v1`/`v2`, `label`/`message`, or `category`/`text`.

The project folder should look like:
```
Ham_Spam_ML_Full_Project/
  app.py
  ml_model.py
  requirements.txt
  templates/
    index.html
  static/
    style.css
  SMSSpamCollection
```

## Step 6 — Start the app
```powershell
python app.py
```
Open http://127.0.0.1:5000 in your browser.

The app trains the model when it starts, evaluates it on a held-out 20% test split, and reports test accuracy on the page. It also saves a `spam_classifier.joblib` file.

## How the prediction works
1. Read labeled messages from the dataset.
2. Split messages into training and test data (80% / 20%).
3. TF-IDF converts text into word and word-pair features.
4. Multinomial Naive Bayes learns from the training messages.
5. The backend receives a message from the form and predicts `spam` or `ham`.
6. Flask returns the result to the HTML page.

## Troubleshooting
- **Model not ready:** Dataset file is missing or not in the project root.
- **ModuleNotFoundError:** Activate `.venv`, then run `pip install -r requirements.txt`.
- **Dataset parse error:** Ensure the UCI file is named `SMSSpamCollection`, or use a compatible CSV with both labels.
- **Port conflict:** Change the last line in `app.py` to use `port=5001`, then open http://127.0.0.1:5001.

## Important
This is an educational prototype. The score is not a guarantee. Do not use the classifier as the only security control for real email or phishing protection.
