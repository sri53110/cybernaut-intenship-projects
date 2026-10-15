"""Text normalisation shared by training and the API (must stay identical)."""
import re

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"[$£€₹]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|gbp|eur|inr|dollars?|pounds?)\b", re.I)
_PHONE = re.compile(r"\b(?:\+?\d[\d\s\-().]{7,}\d)\b")
_NUM = re.compile(r"\b\d+\b")
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case, strip HTML, and replace volatile entities with stable tokens.

    URLs, e-mail addresses, money amounts, phone numbers and bare numbers are
    strong spam signals but are almost never identical twice, so they are
    mapped to placeholder tokens the model can learn from.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = _HTML.sub(" ", text)
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = _MONEY.sub(" moneytoken ", t)
    t = _PHONE.sub(" phonetoken ", t)
    t = _NUM.sub(" numtoken ", t)
    t = t.lower()
    # flag shouting before lower-casing would lose it, so measure on the original
    caps = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    if letters >= 8 and caps / letters > 0.5:
        t += " allcapstoken"
    if text.count("!") >= 3:
        t += " manyexclaimtoken"
    return _WS.sub(" ", t).strip()
