from typing import List

import uuid

from bertopic import BERTopic
from fastapi.exceptions import HTTPException
from fastapi.params import Query
from fastapi.routing import APIRouter
from pydantic.types import UUID4
from sklearn.datasets import fetch_20newsgroups

from service.core.config import settings
from service.schemas.base import Input, ModelId, ModelPrediction, Topic, TopicTopWords, Word

router = APIRouter()


def load_model(model_id: UUID4) -> BERTopic:
    model_path = settings.DATA_DIR / str(model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    return BERTopic.load(model_path)


@router.post("/modeling", summary="Running topic modeling", response_model=ModelId)
async def fit(data: Input) -> ModelId:
    topic_model = BERTopic()
    if data.texts:
        topics, probs = topic_model.fit_transform(data.texts)
    else:
        docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"][
            :100
        ]
        topics, probs = topic_model.fit_transform(docs)
    model_id = uuid.uuid4()
    settings.DATA_DIR.mkdir(
        exist_ok=True, parents=True
    )  # we need to put this in the settings so that the folder is created at the start of the project
    topic_model.save(settings.DATA_DIR / str(model_id))
    return ModelId(model_id=model_id)


@router.post("/predict", summary="Predict with existing model", response_model=ModelPrediction)
async def predict(
    data: Input, model_id: UUID4 = Query(...), calculate_probabilities: bool = Query(default=False)
) -> ModelPrediction:
    topic_model = load_model(model_id)
    topic_model.calculate_probabilities = calculate_probabilities
    topics, probabilities = topic_model.transform(data.texts)
    if probabilities is not None:
        probabilities = probabilities.tolist()
    return ModelPrediction(topics=topics, probabilities=probabilities)


@router.get("/models", summary="Get all existing model ids", response_model=List[str])
async def list_models():
    return [p.name for p in settings.DATA_DIR.iterdir()]


@router.get("/get_topics", summary="Get topics", response_model=List[TopicTopWords])
async def get_topics(model_id: UUID4 = Query(...)) -> List[TopicTopWords]:
    topic_model = load_model(model_id)
    topic_info = topic_model.get_topics()
    topics = []
    for topic_id, top_words in topic_info.items():
        topic = TopicTopWords(
            name=topic_model.topic_names[topic_id],
            topic_id=topic_id,
            top_words=[Word(name=w[0], score=w[1]) for w in top_words],
        )
        topics.append(topic)
    return topics


@router.get("/topics_info", summary="Get topics info", response_model=List[Topic])
async def get_topic_info(model_id: UUID4 = Query(...)) -> List[dict]:
    topic_model = load_model(model_id)
    topic_info = topic_model.get_topic_info()
    topic_info.columns = topic_info.columns.str.lower()
    return topic_info.to_dict("records")


@router.get("/remove_model", summary="Remove topic model")
async def remove_model(model_id: UUID4 = Query(...)) -> str:
    model_path = settings.DATA_DIR / str(model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    model_path.unlink()
    return "ok"
