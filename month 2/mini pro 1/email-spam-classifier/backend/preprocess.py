"""Shared text cleaning, used by both training and the API so they never drift apart."""
import re


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)   # links
    text = re.sub(r"\b\d{5,}\b", " longnumber ", text)             # phone numbers, codes
    text = re.sub(r"[^a-z0-9£$!\s]", " ", text)                    # keep money/! signals
    return re.sub(r"\s+", " ", text).strip()
