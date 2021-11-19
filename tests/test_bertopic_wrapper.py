import pytest
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from service.schemas.bertopic_wrapper import (
    BERTopicWrapper,
    HDBSCANParams,
    UMAPParams,
    VectorizerParams,
)

pytestmark = pytest.mark.unit


class TestBERTopicWrapper:
    def test_init(self):
        wrapper = BERTopicWrapper()
        assert type(wrapper.model) == BERTopic

    def test_init_vectorizer(self):
        vectorizer_params = {
            "lowercase": False,
            "stop_words": ["foo"],
            "ngram_range": [1, 2],
            "max_df": 5,
            "min_df": 2,
        }
        wrapper = BERTopicWrapper(vectorizer_params=VectorizerParams.parse_obj(vectorizer_params))
        assert type(wrapper.vectorizer_model) == CountVectorizer
        for param, value in vectorizer_params.items():
            assert getattr(wrapper.vectorizer_model, param) == value

    def test_init_umap(self):
        umap_params = {
            "n_neighbors": 50,
            "n_components": 5,
            "metric": "cosine",
            "learning_rate": 0.1,
            "n_epochs": 5,
        }
        wrapper = BERTopicWrapper(umap_params=UMAPParams.parse_obj(umap_params))
        assert type(wrapper.umap_model) == UMAP
        for param, value in umap_params.items():
            assert getattr(wrapper.umap_model, param) == value

    def test_init_hdbscan(self):
        hdbscan_params = {
            "min_cluster_size": 7,
            "min_samples": 100,
            "metric": "cosine",
            "alpha": 0.1,
            "p": 1,
        }
        wrapper = BERTopicWrapper(hdbscan_params=HDBSCANParams.parse_obj(hdbscan_params))
        assert type(wrapper.hdbscan_model) == HDBSCAN
        for param, value in hdbscan_params.items():
            assert getattr(wrapper.hdbscan_model, param) == value
