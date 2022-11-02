import uuid

import pytest
from bertopic import BERTopic
from fastapi.testclient import TestClient
from plotly.io import from_json
from pytest_mock import MockFixture


@pytest.mark.unit
class TestVisualizers:
    def test_topics(self, client: TestClient, dummy_model: BERTopic, mocker: MockFixture) -> None:
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualizations/topics",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "topics": [0],
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500
        assert fig["data"][0]["type"] == "scatter"
        assert len(fig["data"][0]["customdata"]) == 1
        assert fig["data"][0]["customdata"][0] == [0, "the | to | is | for | and", 31]

    def test_barchart(
        self, client: TestClient, dummy_model: BERTopic, mocker: MockFixture
    ) -> None:
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualizations/barchart",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "top_n_topics": 2,
                "n_words": 3,
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert len(fig["data"]) == 2
        assert fig["data"][0]["type"] == "bar"
        assert fig["data"][0]["y"] == ("is  ", "to  ", "the  ")
        assert fig["data"][1]["type"] == "bar"
        assert fig["data"][1]["y"] == ("to  ", "of  ", "the  ")

    def test_hierarchy(
        self, client: TestClient, dummy_model: BERTopic, mocker: MockFixture
    ) -> None:
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualizations/hierarchy",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "top_n_topics": 2,
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["data"][0]["mode"] == "lines"
        assert fig["data"][0]["type"] == "scatter"
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["yaxis"]["ticktext"] == ("0_the_to_is", "1_the_of_to")

    def test_heatmap(self, client: TestClient, dummy_model: BERTopic, mocker: MockFixture) -> None:
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualizations/heatmap",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "top_n_topics": 2,
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500
        assert fig["data"][0]["type"] == "heatmap"
        assert fig["data"][0]["x"] == ("0_the_to_is", "1_the_of_to")
        assert fig["data"][0]["y"] == ("0_the_to_is", "1_the_of_to")

    def test_distribution(
        self, client: TestClient, dummy_model: BERTopic, mocker: MockFixture
    ) -> None:
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualizations/distribution",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "probabilities": [0.33, 0.41, 0.25],
                "min_probability": 0.3,
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["data"][0]["type"] == "bar"
        assert fig["data"][0]["x"] == (0.33, 0.41)
        assert fig["data"][0]["y"] == (
            "<b>Topic 0</b>: the_to_is_for_and",
            "<b>Topic 1</b>: the_of_to_and_in",
        )
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500

    def test_term_rank(
        self, client: TestClient, dummy_model: BERTopic, mocker: MockFixture
    ) -> None:
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualizations/term_rank",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "topics": [0, 1],
                "log_scale": True,
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500
        assert len(fig["data"]) == 4
        assert fig["data"][0]["type"] == "scatter"
        assert fig["data"][0]["hovertext"] == "<b>Topic -1</b>:the_and_to_of_that_you_for_in_it_i"
        assert fig["data"][1]["type"] == "scatter"
        assert fig["data"][1]["hovertext"] == "<b>Topic 0</b>:the_to_is_for_and_you_it_of_with_in"
