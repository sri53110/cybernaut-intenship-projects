# SafeTriage Message Risk Scanner (ML-07)

Educational triage tool for simulated emails and SMS. It scores text only. Links are defanged and never opened.

## Run

```bash
cd phishshift
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd backend
python pipeline.py     # optional: builds corpus, trains, writes artifacts/ (app.py does it on first run)
python app.py          # open http://127.0.0.1:5000
```

## Requirement map

| Req | Where |
|---|---|
| FR-01 dedup before split | `dedup()` in `pipeline.py`, proof in `artifacts/dedup_proof.json` |
| FR-02 rules, TF-IDF linear, third model | `CUES`/`rules()`, `tfidf_logreg`, `random_forest` in `build()` |
| FR-03 campaign holdout | `HOLDOUT` list (change it to the mentor-assigned campaigns) |
| FR-04 abstain + safe cues | `ABSTAIN_BAND`, `analyse()`; cues never echo message text |
| FR-05 urgent FPR + drift | `legit_urgent` and `PERTURB` in `train()` |
| NFR-01/02/03 | `sanitize()`, synthetic data only, disclaimer in API and footer |
| AT-01 / AT-02 / AT-03 | "Try" buttons in the UI, `results.json`, perturbation table |

## Still to do yourself

EDA charts, experiment journal, final_report.pdf, SRS.pdf, screenshots, 3-5 min demo, AI-usage disclosure, and replacing the synthetic corpus with your mentor's data. The synthetic data is generated from templates, so scores will look cleaner than real data would.
