"""Test resale-prop-analysis REST API."""

import httpx
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_read_root() -> None:
    """Test that reading the root is successful."""
    with TestClient(app) as client:
        response = client.get("/")
        assert httpx.codes.is_success(response.status_code)


def test_make_prediction() -> None:
    """Test making a prediction."""
    with TestClient(app) as client:
        # Example request payload
        request_data = {
            "input_data": {
                "town": "SENGKANG",
                "flat_type": "4 ROOM",
                "storey_range": "04 TO 06",
                "floor_area_sqm": 93,
                "flat_model": "Model A",
                "remaining_lease": 95,
            }
        }

        # Sending a POST request to the /predict endpoint
        response = client.post("/predict", json=request_data)

        # Providing more information if the test fails
        assert response.status_code == httpx.codes.OK, f"Failed with response: {response.json()}"

        # Asserting that the response is in the expected format
        assert "prediction" in response.json()
