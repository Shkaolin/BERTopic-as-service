from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

from ..schemas.bertopic_wrapper import HDBSCANParams, UMAPParams, VectorizerParams


class Input(BaseModel):
    texts: List[str] = []
    language: str = "english"
    top_n_words: int = 10
    nr_topics: Optional[Union[int, str]] = None
    calculate_probabilities: bool = True
    seed_topic_list: Optional[Dict[str, Any]] = None
    vectorizer_params: Optional[VectorizerParams] = None
    umap_params: Optional[UMAPParams] = None
    hdbscan_params: Optional[HDBSCANParams] = None
    verbose: bool = False

    class Config:
        schema_extra = {
            "example": {
                "language": "english",
                "nr_topics": 3,
                "vectorizer_params": {"stop_words": ["foo", "bar"], "ngram_range": (1, 2)},
                "verbose": True,
            }
        }


class NotEmptyInput(BaseModel):
    texts: List[str] = Field(min_length=1)


class ModelId(BaseModel):
    model_id: UUID4
    version: int = 1


class ModelPrediction(BaseModel):
    topics: List[int]
    probabilities: Optional[List[List[float]]]


class FitResult(ModelId):
    predictions: ModelPrediction


class DocsWithPredictions(Input, ModelPrediction):
    ...


class BaseVisualization(BaseModel):
    model: ModelId
    topics: Optional[List[int]] = None
    top_n_topics: Optional[int] = None
    width: int = 650
    height: int = 650

    @classmethod
    @validator("top_n_topics", always=True)
    def check_topics_and_top_n_topics(
        cls, top_n_topics: Optional[List[int]], values: Dict[str, Optional[str]]
    ) -> Optional[List[int]]:
        topics = values.get("topics")
        if not top_n_topics and not topics:
            raise ValueError("Topics must be defined or top_n_topics grather then zero")
        return top_n_topics


class VisTopicsInput(BaseVisualization):
    ...


class VisBarchartInput(BaseVisualization):
    n_words: int = 5


class VisHierarchyInput(BaseVisualization):
    orientation: str = "left"
    width: int = 1000
    height: int = 600


class VisHeatmapInput(BaseVisualization):
    n_clusters: Optional[int] = None
    width: int = 800
    height: int = 800


class VisDistributionInput(BaseModel):
    model: ModelId
    probabilities: List[float]
    min_probability: float = 0.015
    width: int = 800
    height: int = 600


class VisTermRankInput(BaseModel):
    model: ModelId
    topics: List[int]
    log_scale: bool = False
    width: int = 800
    height: int = 500
