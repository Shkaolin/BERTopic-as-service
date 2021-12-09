from typing import Any, Dict, Iterable, List, Optional, Union

from bertopic import BERTopic
from hdbscan import HDBSCAN
from pydantic import BaseModel
from pydantic.fields import Field
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class VectorizerParams(BaseModel):
    encoding: Optional[str] = "utf-8"
    decode_error: Optional[str] = "strict"
    strip_accents: Optional[str] = None
    lowercase: Optional[bool] = True
    stop_words: Optional[Union[str, List[str]]] = None
    token_pattern: Optional[str] = r"(?u)\b\w\w+\b"
    ngram_range: Optional[List[int]] = Field([1, 1], max_items=2, min_items=2)
    analyzer: Optional[str] = "word"
    max_df: Optional[Union[float, int]] = 1
    min_df: Optional[Union[float, int]] = 1
    max_features: Optional[int] = None
    vocabulary: Optional[Iterable[str]] = None
    binary: Optional[bool] = False


class UMAPParams(BaseModel):
    n_neighbors: Optional[float] = 15
    n_components: Optional[int] = 2
    metric: Optional[str] = "euclidean"
    output_metric: Optional[str] = "euclidean"
    n_epochs: Optional[int] = None
    learning_rate: Optional[float] = 1.0
    init: Optional[str] = "spectral"
    min_dist: Optional[float] = 0.1
    spread: Optional[float] = 1.0
    low_memory: Optional[bool] = True
    set_op_mix_ratio: Optional[float] = 1.0
    local_connectivity: Optional[int] = 1
    repulsion_strength: Optional[float] = 1.0
    negative_sample_rate: Optional[int] = 5
    transform_queue_size: Optional[float] = 4.0
    a: Optional[float] = None
    b: Optional[float] = None
    random_state: Optional[int] = None
    angular_rp_forest: Optional[bool] = False
    n_jobs: Optional[int] = -1
    target_n_neighbors: Optional[int] = -1
    target_metric: Optional[str] = "categorical"
    target_weight: Optional[float] = 0.5
    transform_seed: Optional[int] = 42
    verbose: Optional[bool] = False
    unique: Optional[bool] = False
    densmap: Optional[bool] = False
    dens_lambda: Optional[float] = 2.0
    dens_frac: Optional[float] = 0.3
    dens_var_shift: Optional[float] = 0.1
    output_dens: Optional[bool] = False
    disconnection_distance: Optional[float] = None


class HDBSCANParams(BaseModel):
    min_cluster_size: Optional[int] = 5
    min_samples: Optional[int] = None
    metric: Optional[str] = "euclidean"
    p: Optional[int] = None
    alpha: Optional[float] = 1.0
    cluster_selection_epsilon: Optional[float] = 0.0
    algorithm: Optional[str] = "best"
    leaf_size: Optional[int] = 40
    approx_min_span_tree: Optional[bool] = True
    gen_min_span_tree: Optional[bool] = False
    core_dist_n_jobs: Optional[int] = 4
    cluster_selection_method: Optional[str] = "eom"
    allow_single_cluster: Optional[bool] = False
    prediction_data: Optional[bool] = False
    match_reference_implementation: Optional[bool] = False


class BERTopicWrapper(object):
    def __init__(
        self,
        language: str = "english",
        top_n_words: int = 10,
        nr_topics: Optional[Union[int, str]] = None,
        calculate_probabilities: bool = False,
        seed_topic_list: Optional[Dict[str, Any]] = None,
        vectorizer_params: Optional[VectorizerParams] = None,
        umap_params: Optional[UMAPParams] = None,
        hdbscan_params: Optional[HDBSCANParams] = None,
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
            CountVectorizer(**self.vectorizer_params.dict()) if self.vectorizer_params else None
        )

        # UMAP
        self.umap_model = UMAP(**self.umap_params.dict()) if self.umap_params else None

        # UMAP
        self.hdbscan_model = HDBSCAN(**self.hdbscan_params.dict()) if self.hdbscan_params else None

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
