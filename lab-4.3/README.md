# Lab 4.3 — API Validation and Test Suite

## Purpose
This lab validates the recommender API with unit, integration, and end-to-end tests.

It covers:
- score normalization
- routing logic
- purchase exclusion
- NDCG metric behavior
- API health and recommendation endpoints

## Setup
- Python 3.11+
- Required packages from `requirements.txt` and `requirements-app.txt`:
  - `fastapi`
  - `uvicorn`
  - `redis`
  - `mlflow`
  - `pytest`
  - `pytest-asyncio`
  - `pytest-cov`
  - `httpx`
  - `requests`
  - `evidently`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - plus the shared ML packages listed in the files
- Input artifacts expected under `data/`:
  - `als_artifacts.pkl`
  - `faiss_artifacts.pkl`
  - `faiss_index.bin`
  - `lightfm_serving.pkl`
  - `routing_split.pkl`
  - `events.csv`

## How to Run
First, create the LightFM serving artifact:

```bash
python create_lightfm_serving_artifact.py
```

Then start the API:

```bash
uvicorn app:app --reload
```

Finally, run the test suite:

```bash
pytest test_suite.py
```

## Outputs
The lab produces:
- a slim `data/lightfm_serving.pkl` artifact
- API responses from `/health` and `/recommend`
- test results from the validation suite
- optional coverage reports if enabled

## Key Design Choices
- Separated serving artifact creation from API runtime.
- Tested the core recommendation logic independently from the web layer.
- Included both unit tests and API-level checks.
- Used deterministic fixtures and sample users for repeatability.
- Validated routing, exclusion, and metric behavior.
- Kept the test suite compatible with the production-style API.

## Key Findings
Typical outcomes from this lab include:
- The API should respond correctly before deployment.
- Routing and exclusion logic are critical to recommendation quality.
- Unit tests help catch regressions in metric and filtering code.
- End-to-end checks confirm the service is wired correctly.

## Extra Info
- The script suppresses warnings for cleaner output.
- Run the prerequisite labs first so all artifacts are available.
- `requirements-app.txt` is the preferred dependency set for the API runtime.
- The test suite expects the API server to be running locally.
