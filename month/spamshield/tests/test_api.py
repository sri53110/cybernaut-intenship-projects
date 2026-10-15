import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import app  # noqa: E402

c = app.test_client()
SPAM = "WINNER!! You have won a $1000 prize. Click http://win-prize.biz/claim now to claim your reward"
HAM = "Hey, are we still meeting for lunch at 1 pm tomorrow? Let me know."


def test_health():
    assert c.get("/api/health").get_json()["status"] == "ok"


def test_spam_and_ham():
    s = c.post("/api/predict", json={"text": SPAM}).get_json()
    h = c.post("/api/predict", json={"text": HAM}).get_json()
    assert s["label"] == "spam" and s["spam_probability"] > 0.8 and s["spam_signals"]
    assert h["label"] == "ham" and h["spam_probability"] < 0.3


def test_each_model():
    for m in c.get("/api/models").get_json()["models"]:
        r = c.post("/api/predict", json={"text": SPAM, "model": m["id"]})
        assert r.status_code == 200 and r.get_json()["label"] == "spam"


def test_validation():
    assert c.post("/api/predict", json={}).status_code == 400
    assert c.post("/api/predict", json={"text": "x", "model": "nope"}).status_code == 400
    assert c.post("/api/predict", json={"text": "a" * 30000}).status_code == 413
    assert c.post("/api/predict/batch", json={"texts": []}).status_code == 400


def test_batch():
    r = c.post("/api/predict/batch", json={"texts": [SPAM, HAM]}).get_json()
    assert r["summary"] == {"total": 2, "spam": 1}


def test_metrics_keywords_frontend():
    assert "models" in c.get("/api/metrics").get_json()
    assert c.get("/api/keywords").get_json()["spam"]
    assert c.get("/api/nothing").status_code in (404,)


if __name__ == "__main__":  # run without pytest: python tests/test_api.py
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
