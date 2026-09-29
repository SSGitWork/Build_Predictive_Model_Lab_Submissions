#!/bin/bash
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

first_response=$(curl --silent --fail     "${BASE_URL}/recommend/${TEST_USER_ID}?top_k=5&use_cache=true")

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

second_response=$(curl --silent --fail     "${BASE_URL}/recommend/${TEST_USER_ID}?top_k=5&use_cache=true")

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
