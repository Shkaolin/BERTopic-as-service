import sys

from fastapi.testclient import TestClient

sys.path.append("..")
from service.main import app

client = TestClient(app)


def test_fit():
    response = client.post("/fit", json={"texts": []})
    assert response.status_code == 200
    response_json = response.json()
    # assert len(response.json()) == 1
    assert "model_id" in response_json
    assert "version" in response_json and response_json["version"] == 1
