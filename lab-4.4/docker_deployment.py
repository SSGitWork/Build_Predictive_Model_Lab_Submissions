# =============================================================================
# MODULE 4 | LAB 4.4
# File: 04_docker_deploy.py
# Purpose: Automate the generation of containerization configurations,
#          multi-container compose manifests, .dockerignore settings,
#          GitHub Actions CI automation pipelines, and shell verifiers.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import json
import time
import subprocess
import requests
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 4 | LAB 4.4")
print("  Docker Containerization + CI Pipeline")
print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 1: Generate Dockerfile
# ---------------------------------------------------------------------------
print("\n[1] Generating Dockerfile...")

# TODO: Complete the Dockerfile multi-line configuration string.
# Requirements:
# - Base Image: python:3.11-slim
# - Workdir: /app
# - System dependencies to RUN: apt-get update and install libgomp1, gcc, g++
# - COPY and RUN pip install against 'requirements_module4.txt'
# - COPY source codes ('app.py') and 'data/' directory layers inside the container
# - EXPOSE port 8000
# - Configure ENV defaults: REDIS_HOST=redis, REDIS_PORT=6379, ARTIFACTS_DIR=data
# - CMD execution script layout calling uvicorn pointing to host 0.0.0.0 and port 8000
dockerfile_content = '''# Python 3.13 image matches the current Colab deployment environment
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# TODO: Add your multi-line commands installing system dependencies, python packages,
# copying source fields, exposing ports, and launching uvicorn wrappers.
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgomp1 \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \\
    python -m pip install -r requirements.txt

COPY app.py ./app.py
COPY data/ ./data/

RUN mkdir -p /app/output

EXPOSE 8000

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV ARTIFACTS_DIR=/app/data
ENV TOP_K=10
ENV INTERACTION_THRESHOLD=3

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

with open("Dockerfile", "w", encoding="utf-8") as f:
    f.write(dockerfile_content)

print("    Created -> Dockerfile")


# ---------------------------------------------------------------------------
# SECTION 2: Generate docker-compose.yml
# ---------------------------------------------------------------------------
print("\n[2] Generating docker-compose.yml...")

# TODO: Complete the docker-compose multi-service manifest structure string.
# Requirements:
# - Service 1: 'redis' using 'redis:7-alpine', exposing port 6379, map volume 'redis_data:/data'
# - Service 2: 'app' setting up a context build '.' using your Dockerfile, exposing port 8000
#   Map environmental flags: REDIS_HOST=redis, REDIS_PORT=6379, TOP_K=${TOP_K:-10}
#   Link volumes to persist output plots and mount data read-only ('./data:/app/data:ro')
#   Enforce strict creation ordering using 'depends_on' pointing to a healthy redis service container
# - Define standard driver networks and volumes at the root level boundary blocks
compose_content = '''services:
  redis:
    image: redis:7-alpine
    container_name: recommender-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - recommender-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 5s

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: hybrid-recommender-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_TTL: ${REDIS_TTL:-3600}
      TOP_K: ${TOP_K:-10}
      INTERACTION_THRESHOLD: ${INTERACTION_THRESHOLD:-3}
      ARTIFACTS_DIR: /app/data
    volumes:
      - ./data:/app/data:ro
      - ./output:/app/output
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - recommender-network
    healthcheck:
      test:
        [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
        ]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 60s

networks:
  recommender-network:
    driver: bridge

volumes:
  redis_data:
'''

with open("docker-compose.yml", "w", encoding="utf-8") as f:
    f.write(compose_content)

print("    Created -> docker-compose.yml")


# ---------------------------------------------------------------------------
# SECTION 3: Generate .env file
# ---------------------------------------------------------------------------
print("\n[3] Generating .env file...")

# TODO: Build an environment configuration string tracking local environment tokens
# Variables: REDIS_TTL=3600, REDIS_HOST=redis, REDIS_PORT=6379, TOP_K=10, INTERACTION_THRESHOLD=3
env_content = '''# TODO: Declare key-value pairs mapping environmental default configurations

REDIS_TTL=3600
REDIS_HOST=redis
REDIS_PORT=6379
TOP_K=10
INTERACTION_THRESHOLD=3
'''

with open(".env", "w", encoding="utf-8") as f:
    f.write(env_content)

print("    Created -> .env")


# ---------------------------------------------------------------------------
# SECTION 4: Generate .dockerignore
# ---------------------------------------------------------------------------
print("\n[4] Generating .dockerignore...")

# TODO: Construct a comprehensive exclusion list matching files that must stay out of the build matrix.
# Block patterns matching: __pycache__/, *.ipynb, .git/, venv/, .pytest_cache/, large raw data source
# files like 'data/events.csv' (which are mounted via volumes instead of hardcopied), and local 'output/' plots.
dockerignore_content = '''# TODO: Add pattern filters to exclude unnecessary local files from the build context

# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Notebooks and local virtual environments
*.ipynb
venv/
.venv/
env/
lab4-venv/

# Git and editor files
.git/
.github/
.gitignore
.vscode/
.idea/
.DS_Store

# Local output and experiment tracking
output/
mlruns/
*.db
*.log

# Large course data and model artifacts are mounted as volumes at runtime
data/events.csv
data/item_properties_part1.csv
data/item_properties_part2.csv
data/lightfm_artifacts.pkl
data/lightfm_serving.pkl
data/als_artifacts.pkl
data/faiss_artifacts.pkl
data/faiss_index.bin
data/routing_split.pkl
data/*.pkl

# Docker and environment local-only settings
.env
docker-compose.override.yml
'''

with open(".dockerignore", "w", encoding="utf-8") as f:
    f.write(dockerignore_content)

print("    Created -> .dockerignore")


# ---------------------------------------------------------------------------
# SECTION 5: Generate GitHub Actions CI Pipeline
# ---------------------------------------------------------------------------
print("\n[5] Generating GitHub Actions CI pipeline...")

os.makedirs(".github/workflows", exist_ok=True)

# TODO: Structure a continuous integration workflow configuration string saved to '.github/workflows/ci.yml'.
# Requirements:
# - Trigger: filter execution runs on push or pull requests targeting the 'main' branch
# - Job 1: 'lint' running ubuntu-latest, checking code format checks using flake8, black, or isort
# - Job 2: 'unit-tests' executing pytest targeting 'scripts/03_test_suite.py' with exclusions flags
#   configured to run structural code blocks only (-k "not TestAPI and not TestEndToEnd")
# - Job 3: 'build' running Docker build verification actions cleanly
# - Job 4: 'integration' spinning up containers via 'docker compose up -d' and verifying the /health endpoint contract
ci_content = '''# TODO: Map automated multi-stage CI pipelines tracking sequential check triggers

name: Recommender CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install lint tools
        run: |
          python -m pip install --upgrade pip
          python -m pip install flake8 black

      - name: Run flake8
        run: |
          flake8 app.py test_suite.py --max-line-length=100 --ignore=E203,W503

      - name: Validate Black formatting
        run: |
          black --check app.py test_suite.py

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: lint

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install minimal test dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install pytest numpy requests

      - name: Run unit tests only
        run: |
          pytest test_suite.py -v -k "not TestAPIHealth and not TestRecommendEndpoint and not TestEndToEnd"

  build:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: unit-tests

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t hybrid-recommender-api:ci .

  integration:
    name: Docker Integration Test
    runs-on: ubuntu-latest
    needs: build

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Build containers
        run: |
          docker compose build

      - name: Start services
        run: |
          docker compose up -d

      - name: Wait for API health endpoint
        run: |
          for i in $(seq 1 36); do
            if curl --fail --silent http://localhost:8000/health; then
              echo "API is healthy."
              exit 0
            fi
            echo "Waiting for API startup: attempt $i/36"
            sleep 5
          done
          echo "API did not become healthy."
          docker compose logs
          exit 1

      - name: Run cold-start verification
        run: |
          bash scripts/verify_coldstart.sh

      - name: Print container logs on failure
        if: failure()
        run: |
          docker compose logs

      - name: Shut down services
        if: always()
        run: |
          docker compose down -v
'''

with open(".github/workflows/ci.yml", "w", encoding="utf-8") as f:
    f.write(ci_content)

print("    Created -> .github/workflows/ci.yml")


# ---------------------------------------------------------------------------
# SECTION 6: Generate Cold-Start Verification Script
# ---------------------------------------------------------------------------
print("\n[6] Generating cold-start verification script...")

# TODO: Complete the bash automation script string to test live multi-container orchestration responses.
# Requirements:
# - Step 1: Use a 'while' loop with curl to ping 'http://localhost:8000/health' sequentially until ready
# - Step 2: Validate health JSON outputs ensuring 'als_loaded' and 'lfm_loaded' flags return True
# - Step 3: Trigger a test curl against the recommendation path '/recommend/125625?top_k=5&use_cache=true'
# - Step 4: Verify that a subsequent duplicate endpoint query logs data['cached'] as True via Redis memory lookups
verify_script = '''#!/bin/bash
set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"
MAX_ATTEMPTS=36
SLEEP_SECONDS=5
TEST_USER_ID="${TEST_USER_ID:-125625}"

echo "============================================================"
echo "  Cold-Start Verification — Student Lab Automation"
echo "============================================================"

# TODO: Implement bash testing validation routines pining endpoints and evaluating json responses

echo "[1] Waiting for recommendation API health endpoint..."

attempt=1

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    if health_response=$(curl --silent --fail "${BASE_URL}/health"); then
        echo "    API is ready after ${attempt} attempt(s)."
        break
    fi

    echo "    Waiting for API startup (${attempt}/${MAX_ATTEMPTS})..."
    sleep "$SLEEP_SECONDS"
    attempt=$((attempt + 1))
done

if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "ERROR: API did not become ready within timeout."
    exit 1
fi

echo "[2] Validating health payload..."

echo "$health_response"

echo "$health_response" | python -c '
import json
import sys

payload = json.load(sys.stdin)

assert payload.get("status") == "ok", "Health status is not ok"
assert payload.get("als_loaded") is True, "ALS model did not load"
assert payload.get("lfm_loaded") is True, "LightFM model did not load"

print("    Health validation passed.")
'

echo "[3] Calling recommendation endpoint..."

first_response=$(curl --silent --fail \
    "${BASE_URL}/recommend/${TEST_USER_ID}?top_k=5&use_cache=true")

echo "$first_response"

echo "$first_response" | python -c '
import json
import sys

payload = json.load(sys.stdin)

assert "recommendations" in payload, "Recommendations field missing"
assert len(payload["recommendations"]) > 0, "No recommendations returned"
assert len(payload["recommendations"]) <= 5, "Too many recommendations returned"

print("    First recommendation request passed.")
'

echo "[4] Calling duplicate request to verify caching..."

second_response=$(curl --silent --fail \
    "${BASE_URL}/recommend/${TEST_USER_ID}?top_k=5&use_cache=true")

echo "$second_response"

echo "$second_response" | python -c '
import json
import sys

payload = json.load(sys.stdin)

assert payload.get("cached") is True, "Expected a cache hit on duplicate request"

print("    Cache verification passed.")
'

echo "============================================================"
echo "  Cold-start verification completed successfully"
echo "============================================================"
'''

os.makedirs("scripts", exist_ok=True)

with open("scripts/verify_coldstart.sh", "w", encoding="utf-8") as f:
    f.write(verify_script)

os.chmod("scripts/verify_coldstart.sh", 0o755)

print("    Created -> scripts/verify_coldstart.sh")


# ---------------------------------------------------------------------------
# SECTION 7: Verify All Files Created
# ---------------------------------------------------------------------------
print("\n[7] Verifying all deployment files created...")

files_to_check = [
    ("Dockerfile", "Container image definition"),
    ("docker-compose.yml", "Service orchestration"),
    (".env", "Environment variables"),
    (".dockerignore", "Build context exclusions"),
    (".github/workflows/ci.yml", "GitHub Actions CI pipeline"),
    ("scripts/verify_coldstart.sh", "Cold-start verification"),
]

all_good = True

print(f"\n    {'File':<40} {'Status':>8}  Description")
print(f"    {'-' * 75}")

for filepath, description in files_to_check:
    exists = os.path.exists(filepath)
    status = "OK" if exists else "MISSING"

    if not exists:
        all_good = False

    print(f"    {filepath:<40} {status:>8}  {description}")


# ---------------------------------------------------------------------------
# SECTION 8: Try Docker Build (if Docker is available)
# ---------------------------------------------------------------------------
print("\n[8] Checking Docker availability...")

docker_available = False

try:
    result = subprocess.run(
        ["docker", "--version"],
        capture_output=True,
        text=True,
        timeout=10
    )

    if result.returncode == 0:
        docker_available = True
        print(f"    Docker found: {result.stdout.strip()}")
    else:
        print("    Docker not available locally.")

except Exception:
    print("    Docker daemon not found.")

if docker_available:
    print("\n    To build and start your containerized system:")
    print("    Step 1: docker compose build")
    print("    Step 2: docker compose up -d")
    print("    Step 3: bash scripts/verify_coldstart.sh")
    print("    Step 4: docker compose down")

else:
    print("\n    Docker is not available in this environment.")
    print("    Run these generated files locally using Docker Desktop.")

print("\n" + "=" * 60)
print("  LAB 4.4 COMPLETE — DOCKER DEPLOYMENT STACK")
print("=" * 60)

if all_good:
    print("  All deployment configuration files were generated successfully.")
else:
    print("  WARNING: One or more deployment files are missing.")
