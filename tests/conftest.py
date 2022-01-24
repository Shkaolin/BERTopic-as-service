import pytest
from fastapi.testclient import TestClient

from service.main import app


@pytest.fixture()
def client():
    with TestClient(app=app) as client:
        yield client
