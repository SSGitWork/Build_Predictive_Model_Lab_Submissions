# Python 3.13 image matches the current Colab deployment environment
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# TODO: Add your multi-line commands installing system dependencies, python packages,
# copying source fields, exposing ports, and launching uvicorn wrappers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \
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
