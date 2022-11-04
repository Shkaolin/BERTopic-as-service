from typing import Generator

import pytest
from bertopic import BERTopic
from fastapi.testclient import TestClient

from service.main import app


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app=app) as client:
        yield client


@pytest.fixture()
def dummy_model() -> BERTopic:
    return BERTopic.load("tests/assets/model.mdl")
