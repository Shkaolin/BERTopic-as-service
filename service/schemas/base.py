from typing import List, Optional
from pydantic import BaseModel
from pydantic.types import UUID4

class Input(BaseModel):
    texts: List[str] = []

class Topic(BaseModel):
    name: str
    count: int
    topic: int

class ModelId(BaseModel):
    model_id: UUID4


class ModelPrediction(BaseModel):
    topics: List[int]
    probabilities: Optional[List[List[float]]]