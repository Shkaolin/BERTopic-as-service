import uuid

import pytest
from plotly.io import from_json


@pytest.mark.unit
class TestVisualizers(object):
    def test_topics(self, client, dummy_model, mocker):
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualization/topics",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500
        assert fig["data"][0]["type"] == "scatter"
        assert len(fig["data"][0]["customdata"]) == 3
        assert fig["data"][0]["customdata"][0][2:] == [0, "the | to | of | and | in", 3842]
        assert fig["data"][0]["customdata"][1][2:] == [1, "done | why | what | of | ", 124]
        assert fig["data"][0]["customdata"][2][2:] == [2, "is | it | the | gordon | chastity", 32]

    def test_barchart(self, client, dummy_model, mocker):
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualization/barchart",
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
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500
        assert len(fig["data"]) == 2
        assert fig["data"][0]["type"] == "bar"
        assert fig["data"][0]["y"] == ("what  ", "why  ", "done  ")
        assert fig["data"][1]["type"] == "bar"
        assert fig["data"][1]["y"] == ("the  ", "it  ", "is  ")

    def test_hierarchy(self, client, dummy_model, mocker):
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualization/hierarchy",
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
        assert fig["layout"]["height"] == 500
        assert fig["layout"]["yaxis"]["ticktext"] == ("1_done_why_what", "2_is_it_the")

    def test_heatmap(self, client, dummy_model, mocker):
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualization/heatmap",
            json={
                "model": {"model_id": str(uuid.uuid4()), "version": 1},
                "top_n_topics": 2,
                "width": 500,
                "height": 500,
            },
        )
        assert response.status_code == 200
        fig = from_json(response.json())
        assert fig["data"][0]["type"] == "heatmap"
        assert fig["data"][0]["x"] == ("1_done_why_what", "2_is_it_the")
        assert fig["data"][0]["y"] == ("1_done_why_what", "2_is_it_the")
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500

    def test_distribution(self, client, dummy_model, mocker):
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualization/distribution",
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
            "<b>Topic 0</b>: the_to_of_and_in",
            "<b>Topic 1</b>: done_why_what_of_",
        )
        assert fig["layout"]["width"] == 500
        assert fig["layout"]["height"] == 500

    def test_term_rank(self, client, dummy_model, mocker):
        mocker.patch("service.api.endpoints.visualization.check_topics")
        mocker.patch("service.api.endpoints.visualization.load_model", return_value=dummy_model)
        response = client.post(
            "/visualization/term_rank",
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
        assert len(fig["data"]) == 2
        assert fig["data"][0]["type"] == "scatter"
        assert fig["data"][0]["hovertext"] == "<b>Topic 0</b>:the_to_of_and_in_is_that_it_for_you"
        assert fig["data"][1]["type"] == "scatter"
        assert fig["data"][1]["hovertext"] == "<b>Topic 2</b>:is_it_the_gordon_chastity_n3jxp_int"
