# Lab 4.4 — Docker Deployment and CI Automation

## Purpose
This lab automates containerization and CI setup for the recommender API.

It generates:
- a Dockerfile
- a `docker-compose.yml` stack
- a `.dockerignore`
- a `.env` file
- a GitHub Actions CI workflow
- a cold-start verification script

## Setup
- Python 3.11+
- Required packages from `requirements.txt` and `requirements-app.txt`:
  - `fastapi`
  - `uvicorn`
  - `redis`
  - `mlflow`
  - `pytest`
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
From the lab directory, run:

```bash
python docker_deployment.py
```

The script writes the deployment files, then you can build and run the container stack with Docker Compose.

## Outputs
The script creates:
- `Dockerfile`
- `docker-compose.yml`
- `.env`
- `.dockerignore`
- `.github/workflows/ci.yml`
- `scripts/verify_coldstart.sh`

It also supports:
- containerized API startup
- Redis-backed caching
- CI validation for build and integration checks

## Key Design Choices
- Used a slim Python base image for smaller containers.
- Separated app and Redis into distinct services.
- Mounted data as a volume instead of baking large files into the image.
- Added health checks for both Redis and the API.
- Generated CI steps for linting, unit tests, Docker build, and integration tests.
- Added a cold-start verification script to validate the deployed service.
- Kept environment values configurable through `.env`.

## Key Findings
Typical outcomes from this lab include:
- Containerization makes the recommender easier to deploy consistently.
- Compose simplifies local multi-service orchestration.
- CI helps catch regressions before deployment.
- Health checks and verification scripts improve operational confidence.

## Extra Info
- The script suppresses warnings for cleaner output.
- Run the prerequisite labs first so all artifacts are available.
- The generated files should be reviewed before committing.
- Docker and Docker Compose must be installed to use the generated deployment stack.
