from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic.types import UUID4

from service.schemas.bertopic_wrapper import HDBSCANParams, UMAPParams, VectorizerParams


class Input(BaseModel):
    texts: List[str] = []


class ModelParams(Input):
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
                "language": "multilingual",
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
