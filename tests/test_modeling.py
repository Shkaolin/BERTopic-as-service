import sys

from fastapi.testclient import TestClient

sys.path.append("..")
from service.main import app

client = TestClient(app)


def test_modeling():
    response = client.post("/modeling", json={"texts": []})
    assert response.status_code == 200
    assert len(response.json()) == 1
