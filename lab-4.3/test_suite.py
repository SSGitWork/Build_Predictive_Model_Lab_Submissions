# =============================================================================
# MODULE 4 | LAB 4.3
# File: test_suite.py
# Purpose: Unit, integration, and end-to-end testing for the recommender API.
# =============================================================================

import os
import time
import pickle
import numpy as np
import pytest
import requests
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = os.getenv("DATA_DIR", "data")
SERVER_UP = False
REQUEST_TIMEOUT = 90


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts():
    """
    Load all artifacts once for the entire test session.
    """
    # TODO: Load 'als_artifacts.pkl', 'lightfm_artifacts.pkl',
    # 'faiss_artifacts.pkl', and 'routing_split.pkl' from DATA_DIR.
    # Hint: Use pytest.skip() inside an except FileNotFoundError block if files are missing.
    try:
        with open(
            os.path.join(DATA_DIR, "als_artifacts.pkl"),
            "rb"
        ) as file:
            als_artifacts = pickle.load(file)

        # Uses the compact serving artifact created for Module 4 deployment.
        with open(
            os.path.join(DATA_DIR, "lightfm_serving.pkl"),
            "rb"
        ) as file:
            lightfm_artifacts = pickle.load(file)

        with open(
            os.path.join(DATA_DIR, "faiss_artifacts.pkl"),
            "rb"
        ) as file:
            faiss_artifacts = pickle.load(file)

        with open(
            os.path.join(DATA_DIR, "routing_split.pkl"),
            "rb"
        ) as file:
            routing_split = pickle.load(file)

    except FileNotFoundError as error:
        pytest.skip(f"Required artifact is missing: {error.filename}")

    except EOFError:
        pytest.skip("One or more artifacts are incomplete or corrupted.")

    return {
        "als": als_artifacts,
        "lightfm": lightfm_artifacts,
        "faiss": faiss_artifacts,
        "routing": routing_split,
    }


@pytest.fixture(scope="session")
def sample_users(artifacts):
    """
    Return a small set of known user IDs for testing.
    """
    # TODO: Extract the first 20 user ID keys from the faiss artifacts mapping layer
    user_mapping = artifacts["faiss"]["user_to_idx"]

    users = [
        int(user_id)
        for user_id in list(user_mapping.keys())[:20]
    ]

    if not users:
        pytest.skip("No users were available in FAISS artifact mappings.")

    return users


@pytest.fixture(scope="session")
def api_available():
    """
    Check if the API server is running by hitting the /health endpoint.
    """
    # TODO: Construct a requests.get query pointing to f"{BASE_URL}/health".
    # Return True if status_code == 200, else return False.
    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=10
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


# ===========================================================================
# UNIT TESTS — Pure Python, no API needed
# ===========================================================================

class TestScoreNormalization:
    """Unit tests for score normalization function."""

    def _normalize(self, scores):
        """Inline normalization tracker under test."""
        arr = np.array(scores, dtype=float)

        lo, hi = arr.min(), arr.max()

        if hi == lo:
            return [0.5] * len(scores)

        return ((arr - lo) / (hi - lo)).tolist()

    def test_all_scores_in_zero_one_range(self):
        # TODO: Define a dummy list of raw floating scores (e.g., negative, positive, or outside [0, 1])
        # Assert that after passing scores through self._normalize, ALL elements fall within [0.0, 1.0]
        raw_scores = [-5.0, -1.5, 0.0, 4.0, 10.0]

        normalized_scores = self._normalize(raw_scores)

        assert all(
            0.0 <= score <= 1.0
            for score in normalized_scores
        )

    def test_max_score_is_one(self):
        # TODO: Define a random list of asymmetric scores.
        # Assert that the maximum value of the normalized output list is exactly 1.0 (use pytest.approx)
        raw_scores = [3.1, 8.7, -2.5, 4.0]

        normalized_scores = self._normalize(raw_scores)

        assert max(normalized_scores) == pytest.approx(1.0)

    def test_min_score_is_zero(self):
        # TODO: Define a random list of scores.
        # Assert that the minimum value of the normalized output list is exactly 0.0 (use pytest.approx)
        raw_scores = [17.5, 10.0, 11.2, 90.0]

        normalized_scores = self._normalize(raw_scores)

        assert min(normalized_scores) == pytest.approx(0.0)

    def test_identical_scores_return_half(self):
        # TODO: Pass a list of identical values (e.g., [0.5, 0.5, 0.5]) into self._normalize
        # Assert that all resulting values equal 0.5
        raw_scores = [0.5, 0.5, 0.5]

        normalized_scores = self._normalize(raw_scores)

        assert normalized_scores == [0.5, 0.5, 0.5]

    def test_single_score_returns_half(self):
        # TODO: Pass a single element score list into self._normalize
        # Assert that the single output scalar equals 0.5
        normalized_scores = self._normalize([8.5])

        assert normalized_scores[0] == pytest.approx(0.5)

    def test_output_length_matches_input(self):
        # TODO: Assert that the size of the output list matches the input length exactly
        raw_scores = [-4.0, 0.2, 8.0, 10.1, 20.5]

        normalized_scores = self._normalize(raw_scores)

        assert len(normalized_scores) == len(raw_scores)

    def test_negative_scores_handled(self):
        # TODO: Pass a list containing negative numbers (e.g., [-1.0, 0.0, 2.0]) into self._normalize
        # Assert that elements are scaled into a valid [0, 1] range correctly
        normalized_scores = self._normalize([-1.0, 0.0, 2.0])

        assert normalized_scores[0] == pytest.approx(0.0)
        assert normalized_scores[-1] == pytest.approx(1.0)
        assert all(
            0.0 <= score <= 1.0
            for score in normalized_scores
        )


class TestRoutingLogic:
    """Unit tests for the routing decision logic."""

    def _route(self, user_id, als_users, n_interactions, threshold=3):
        """Inline routing logic tracker under test."""
        use_als = (
            user_id in als_users and
            n_interactions >= threshold
        )

        return "ALS" if use_als else "LightFM"

    def test_returning_user_routes_to_als(self, artifacts):
        # TODO: Extract 'als_users' from the routing split artifact. Grab a sample user_id.
        # Call self._route providing an interaction count above threshold (e.g., n_interactions=10).
        # Assert that the returned routing string equals "ALS".
        als_users = set(artifacts["routing"]["als_users"])

        if not als_users:
            pytest.skip("No ALS users exist in routing split.")

        user_id = next(iter(als_users))

        engine = self._route(
            user_id,
            als_users,
            n_interactions=10
        )

        assert engine == "ALS"

    def test_new_user_routes_to_lightfm(self, artifacts):
        # TODO: Define a fake user_id not present in the dataset (e.g., -99999).
        # Assert that calling self._route returns "LightFM".
        als_users = set(artifacts["routing"]["als_users"])

        fake_user_id = -99999

        engine = self._route(
            fake_user_id,
            als_users,
            n_interactions=0
        )

        assert engine == "LightFM"

    def test_returning_user_below_threshold_routes_to_lightfm(
        self,
        artifacts
    ):
        # TODO: Grab a valid returning user_id, but pass an interaction count below threshold (e.g., n_interactions=1).
        # Assert that the resulting routing engine equals "LightFM".
        als_users = set(artifacts["routing"]["als_users"])

        if not als_users:
            pytest.skip("No ALS users exist in routing split.")

        user_id = next(iter(als_users))

        engine = self._route(
            user_id,
            als_users,
            n_interactions=1
        )

        assert engine == "LightFM"

    def test_threshold_boundary_exact(self, artifacts):
        # TODO: Test exact threshold constraints. Pass n_interactions=3 with threshold=3.
        # Assert that the user is successfully routed to "ALS".
        als_users = set(artifacts["routing"]["als_users"])

        if not als_users:
            pytest.skip("No ALS users exist in routing split.")

        user_id = next(iter(als_users))

        engine = self._route(
            user_id,
            als_users,
            n_interactions=3,
            threshold=3
        )

        assert engine == "ALS"


class TestPurchaseExclusion:
    """Unit tests for already-purchased item exclusion."""

    def _exclude(self, items, scores, user_purchases):
        filtered = [
            (item_id, score)
            for item_id, score in zip(items, scores)
            if item_id not in user_purchases
        ]

        if not filtered:
            return items, scores

        filtered_items, filtered_scores = zip(*filtered)

        return list(filtered_items), list(filtered_scores)

    def test_purchased_items_removed(self):
        # TODO: Define list of items, matching score float keys, and a 'purchased' set containing elements to exclude.
        # Assert that excluded items do NOT exist inside the final cleaned items array.
        items = [101, 102, 103, 104]
        scores = [0.9, 0.8, 0.7, 0.6]
        purchased = {102, 104}

        filtered_items, _ = self._exclude(
            items,
            scores,
            purchased
        )

        assert 102 not in filtered_items
        assert 104 not in filtered_items

    def test_non_purchased_items_kept(self):
        # TODO: Define item arrays, scores, and a purchase exclusion list.
        # Assert that items not marked as purchased are retained in the filtered output list.
        items = [10, 20, 30, 40]
        scores = [0.95, 0.80, 0.60, 0.40]
        purchased = {20}

        filtered_items, _ = self._exclude(
            items,
            scores,
            purchased
        )

        assert filtered_items == [10, 30, 40]

    def test_no_purchases_returns_original(self):
        # TODO: Provide an empty set() for user purchases.
        # Assert that filtered outputs are identical to the raw inputs.
        items = [1, 2, 3]
        scores = [0.3, 0.2, 0.1]

        filtered_items, filtered_scores = self._exclude(
            items,
            scores,
            set()
        )

        assert filtered_items == items
        assert filtered_scores == scores

    def test_all_purchased_returns_original(self):
        # TODO: Test boundary conditions. Provide a purchase history containing all items in the candidate list.
        # Assert that the function falls back gracefully by returning the original lists.
        items = [100, 200, 300]
        scores = [0.9, 0.7, 0.5]
        purchased = {100, 200, 300}

        filtered_items, filtered_scores = self._exclude(
            items,
            scores,
            purchased
        )

        assert filtered_items == items
        assert filtered_scores == scores

    def test_scores_aligned_after_exclusion(self):
        # TODO: Define items, scores, and a partial exclusion set.
        # Loop through the outputs and assert that every retained item's score remains perfectly aligned with its original input score.
        items = [1, 2, 3, 4]
        scores = [0.99, 0.75, 0.45, 0.15]
        purchased = {2, 4}

        filtered_items, filtered_scores = self._exclude(
            items,
            scores,
            purchased
        )

        original_mapping = dict(zip(items, scores))

        for item_id, score in zip(
            filtered_items,
            filtered_scores
        ):
            assert score == original_mapping[item_id]


class TestNDCG:
    """Unit tests for NDCG@K metric."""

    def _ndcg(self, recommended, relevant, k=10):
        rec_k = recommended[:k]

        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, item_id in enumerate(rec_k)
            if item_id in relevant
        )

        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(relevant), k))
        )

        return dcg / idcg if idcg > 0 else 0.0

    def test_perfect_ranking_returns_one(self):
        # TODO: Define a relevant set and recommended list where the top items perfectly match the relevant set.
        # Assert that self._ndcg returns 1.0 (use pytest.approx).
        relevant = {101, 102, 103}
        recommended = [101, 102, 103, 999]

        score = self._ndcg(
            recommended,
            relevant,
            k=3
        )

        assert score == pytest.approx(1.0)

    def test_no_relevant_returns_zero(self):
        # TODO: Define completely disjoint relevant sets and recommendations.
        # Assert that self._ndcg returns 0.0.
        relevant = {10, 20}
        recommended = [30, 40, 50]

        score = self._ndcg(
            recommended,
            relevant,
            k=10
        )

        assert score == pytest.approx(0.0)

    def test_relevant_at_top_beats_bottom(self):
        # TODO: Contrast two lists: one with a relevant item at index 0, and another with the same item at the bottom.
        # Assert that the score for the top-ranked item is higher than the bottom-ranked item.
        relevant = {99}

        top_ranked = [99, 1, 2, 3, 4]
        bottom_ranked = [1, 2, 3, 4, 99]

        top_score = self._ndcg(
            top_ranked,
            relevant,
            k=5
        )

        bottom_score = self._ndcg(
            bottom_ranked,
            relevant,
            k=5
        )

        assert top_score > bottom_score


class TestMMRDiversity:
    """Unit tests for MMR reranking."""

    def _mmr(
        self,
        items,
        scores,
        embeddings,
        id_to_idx,
        top_k=5,
        lam=0.5
    ):
        norm_scores = np.array(scores, dtype=float)

        low, high = norm_scores.min(), norm_scores.max()

        if high > low:
            norm_scores = (
                norm_scores - low
            ) / (
                high - low
            )

        valid = [
            (
                item_id,
                score,
                embeddings[id_to_idx[item_id]]
            )
            for item_id, score in zip(items, norm_scores)
            if item_id in id_to_idx
        ]

        if not valid:
            return items[:top_k]

        selected = []
        selected_embeddings = []
        remaining = list(range(len(valid)))

        for _ in range(min(top_k, len(valid))):
            if not remaining:
                break

            if not selected_embeddings:
                best = max(
                    remaining,
                    key=lambda index: valid[index][1]
                )

            else:
                selected_matrix = np.array(
                    selected_embeddings
                )

                mmr_values = []

                for index in remaining:
                    relevance = lam * valid[index][1]

                    similarity = (
                        valid[index][2] @ selected_matrix.T
                    ).max()

                    mmr_values.append(
                        relevance - (1 - lam) * similarity
                    )

                best = remaining[np.argmax(mmr_values)]

            selected.append(valid[best][0])
            selected_embeddings.append(valid[best][2])
            remaining.remove(best)

        return selected

    @staticmethod
    def _unit_norm_embeddings():
        """Create a deterministic set of normalized test embeddings."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7, 0.2, 0.7],
            [0.2, 0.8, 0.5],
        ])

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        return embeddings / norms

    def test_output_length_equals_top_k(self):
        # TODO: Mock a list of candidate items, relevance scores, an id mapping dictionary, and unit-norm embeddings.
        # Assert that the output length of self._mmr matches top_k exactly.
        items = [1, 2, 3, 4, 5, 6]
        scores = [0.90, 0.85, 0.70, 0.60, 0.50, 0.40]

        embeddings = self._unit_norm_embeddings()

        item_mapping = {
            item_id: index
            for index, item_id in enumerate(items)
        }

        recommendations = self._mmr(
            items,
            scores,
            embeddings,
            item_mapping,
            top_k=4
        )

        assert len(recommendations) == 4

    def test_no_duplicate_items(self):
        # TODO: Pass a list of candidates into self._mmr.
        # Assert that the returned recommendations contain no duplicate IDs.
        items = [1, 2, 3, 4, 5, 6]
        scores = [0.91, 0.80, 0.70, 0.65, 0.40, 0.30]

        embeddings = self._unit_norm_embeddings()

        item_mapping = {
            item_id: index
            for index, item_id in enumerate(items)
        }

        recommendations = self._mmr(
            items,
            scores,
            embeddings,
            item_mapping,
            top_k=5
        )

        assert len(recommendations) == len(set(recommendations))

    def test_lambda_one_returns_highest_relevance_first(self):
        # TODO: Configure pure relevance by setting lam=1.0. Pass mismatched items/scores.
        # Assert that the first returned item is the one with the highest absolute score.
        items = [10, 20, 30, 40]
        scores = [0.20, 0.99, 0.55, 0.40]

        embeddings = self._unit_norm_embeddings()[:4]

        item_mapping = {
            item_id: index
            for index, item_id in enumerate(items)
        }

        recommendations = self._mmr(
            items,
            scores,
            embeddings,
            item_mapping,
            top_k=3,
            lam=1.0
        )

        assert recommendations[0] == 20


# ===========================================================================
# INTEGRATION TESTS — Require running API server
# ===========================================================================

class TestAPIHealth:
    """Integration tests for the /health endpoint."""

    def test_health_returns_200(self, api_available):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Issue a requests.get method call to f"{BASE_URL}/health". Assert status code is 200.
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=15
        )

        assert response.status_code == 200

    def test_health_response_schema(self, api_available):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Extract the JSON payload from the health endpoint.
        # Assert that keys like "status", "als_loaded", "lfm_loaded", and "redis_connected" exist in the response.
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=15
        )

        payload = response.json()

        required_keys = {
            "status",
            "als_loaded",
            "lfm_loaded",
            "redis_connected",
            "startup_time_s"
        }

        assert required_keys.issubset(payload.keys())

    def test_health_status_is_ok(self, api_available):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Assert that response JSON data["status"] == "ok".
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=15
        )

        assert response.json()["status"] == "ok"


class TestRecommendEndpoint:
    """Integration tests for the /recommend endpoint."""

    def test_valid_user_returns_200(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Query the recommendation endpoint using a valid user ID from the sample_users fixture.
        # Assert that the server handles the call with an HTTP status code of 200.
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

    def test_response_has_required_fields(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Assert that the endpoint response body contains keys like "engine", "recommendations", "scores", and "latency_ms".
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

        payload = response.json()

        required_keys = {
            "user_id",
            "engine",
            "recommendations",
            "scores",
            "cached",
            "latency_ms"
        }

        assert required_keys.issubset(payload.keys())

    def test_recommendations_count_matches_top_k(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Pass params={"top_k": 5} to the endpoint call.
        # Assert that the length of returned recommendations list is less than or equal to 5.
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

        recommendations = response.json()["recommendations"]

        assert 1 <= len(recommendations) <= 5

    def test_scores_in_zero_one_range(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Parse scores returned from a live /recommend call.
        # Assert that all scores are properly normalized within the [0.0, 1.0] interval.
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

        scores = response.json()["scores"]

        assert all(
            0.0 <= score <= 1.0
            for score in scores
        )

    def test_latency_under_threshold(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Query the recommendation endpoint for a sample user.
        # Assert that the returned latency field "latency_ms" is well under your production SLA limits (e.g., < 500ms).
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": True
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

        payload = response.json()

        # First query might be a cache miss and slower in Colab.
        # Test the warm cache path for the 500 ms service guardrail.
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": True
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

        warm_payload = response.json()

        assert warm_payload["cached"] is True
        assert warm_payload["latency_ms"] < 500


# ===========================================================================
# END-TO-END TESTS — Seeded dataset verification
# ===========================================================================

class TestEndToEnd:
    """End-to-end integration and system behavior assertions."""

    def test_e2e_recommendations_are_integers(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Call the recommendation endpoint. Assert that all item identifiers in the recommendations list are raw python integers.
        response = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 5,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response.status_code == 200

        recommendations = response.json()["recommendations"]

        assert all(
            isinstance(item_id, int)
            for item_id in recommendations
        )

    def test_e2e_multiple_users_get_different_recs(
        self,
        api_available,
        sample_users
    ):
        if not api_available:
            pytest.skip("API server not running")

        if len(sample_users) < 2:
            pytest.skip("At least two users are required.")

        # TODO: Query recommendations for two distinct users (sample_users[0] vs sample_users[1]).
        # Assert that they do not get identical recommendation lists (verify that the overlap is strict less than top_k).
        response_one = requests.get(
            f"{BASE_URL}/recommend/{sample_users[0]}",
            params={
                "top_k": 10,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        response_two = requests.get(
            f"{BASE_URL}/recommend/{sample_users[1]}",
            params={
                "top_k": 10,
                "use_cache": False
            },
            timeout=REQUEST_TIMEOUT
        )

        assert response_one.status_code == 200
        assert response_two.status_code == 200

        recs_one = response_one.json()["recommendations"]
        recs_two = response_two.json()["recommendations"]

        overlap = len(set(recs_one).intersection(recs_two))

        assert overlap < min(len(recs_one), len(recs_two))

    def test_e2e_stats_endpoint(self, api_available):
        if not api_available:
            pytest.skip("API server not running")

        # TODO: Issue a requests.get method target against f"{BASE_URL}/stats".
        # Assert response is 200, and fields like "als_users" or "lfm_items" exist.
        response = requests.get(
            f"{BASE_URL}/stats",
            timeout=15
        )

        assert response.status_code == 200

        payload = response.json()

        assert "als_users" in payload
        assert "lfm_items" in payload
        assert "faiss_items" in payload


# ===========================================================================
# STANDALONE RUNNER — run without pytest for demo
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 4 | LAB 4.3")
    print("  Running test suite in standalone mode")
    print("=" * 60)

    raise SystemExit(
        pytest.main([
            __file__,
            "-v",
            "--tb=short"
        ])
    )
