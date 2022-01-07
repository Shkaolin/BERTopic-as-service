import sys

import pytest
from bertopic import BERTopic
from fastapi.testclient import TestClient

sys.path.append("..")
from service.main import app


@pytest.fixture()
def client():
    with TestClient(app=app) as client:
        yield client


@pytest.fixture()
def dummy_model():
    return BERTopic.load("tests/assets/model.mdl")
