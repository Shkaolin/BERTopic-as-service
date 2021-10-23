import sys

from fastapi.testclient import TestClient

sys.path.append("..")
from service.main import app

client = TestClient(app)


def test_fit():
    response = client.post("/fit", json={"texts": []})
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert "model_id" in response.json()
