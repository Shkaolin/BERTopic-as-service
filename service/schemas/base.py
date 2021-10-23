from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic.types import UUID4


class Input(BaseModel):
    texts: List[str] = []


class NotEmptyInput(BaseModel):
    texts: List[str] = Field(min_length=1)


class Topic(BaseModel):
    name: str
    count: int
    topic: int


class ModelId(BaseModel):
    model_id: UUID4


class ModelPrediction(BaseModel):
    topics: List[int]
    probabilities: Optional[List[List[float]]]


class Word(BaseModel):
    name: str
    score: float


class TopicTopWords(BaseModel):
    topic_id: int
    name: str
    top_words: List[Word]
