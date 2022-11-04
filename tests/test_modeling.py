import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


def test_fit(client: TestClient) -> None:
    response = client.post("/modeling/training", json={"texts": []})
    assert response.status_code == 200

    response_json = response.json()
    assert "model" in response_json
    assert "model_id" in response_json["model"]
    assert "version" in response_json["model"] and response_json["model"]["version"] == 1
    assert "predictions" in response_json
    assert len(response_json["predictions"]["topics"]) == len(
        response_json["predictions"]["probabilities"]
    )
