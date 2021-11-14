from typing import Any, Dict, Optional, Union

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class BERTopicWrapper(object):
    def __init__(
        self,
        language: str = "english",
        top_n_words: int = 10,
        nr_topics: Optional[Union[int, str]] = None,
        calculate_probabilities: bool = False,
        seed_topic_list: Optional[Dict[str, Any]] = None,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        umap_params: Optional[Dict[str, Any]] = None,
        hdbscan_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        self.language = language
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.calculate_probabilities = calculate_probabilities
        self.seed_topic_list = seed_topic_list
        self.verbose = verbose
        self.vectorizer_params = vectorizer_params
        self.umap_params = umap_params
        self.hdbscan_params = hdbscan_params

        # Vectorizer
        self.vectorizer_model = (
            CountVectorizer(**self.vectorizer_params) if self.vectorizer_params else None
        )

        # UMAP
        self.umap_model = UMAP(**self.umap_params) if self.umap_params else None

        # UMAP
        self.hdbscan_model = HDBSCAN(**self.hdbscan_params) if self.hdbscan_params else None

        self.model = BERTopic(
            language=self.language,
            top_n_words=self.top_n_words,
            nr_topics=self.nr_topics,
            calculate_probabilities=self.calculate_probabilities,
            seed_topic_list=self.seed_topic_list,
            vectorizer_model=self.vectorizer_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            verbose=self.verbose,
        )
