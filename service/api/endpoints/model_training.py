from typing import List

import io
import uuid

import joblib
from aiobotocore.session import ClientCreatorContext
from bertopic import BERTopic
from fastapi.exceptions import HTTPException
from fastapi.params import Depends, Query
from fastapi.routing import APIRouter
from pydantic.types import UUID4
from sklearn.datasets import fetch_20newsgroups
from sqlmodel import Session, select

from service.api import deps
from service.core.config import settings
from service.models import models
from service.schemas.base import (
    Input,
    ModelId,
    ModelPrediction,
    NotEmptyInput,
    TopicTopWords,
    Word,
)

router = APIRouter()


async def load_model(s3: ClientCreatorContext, model_id: UUID4) -> BERTopic:
    try:
        response = await s3.get_object(Bucket=settings.MINIO_BUCKET_NAME, Key=str(model_id))

        with io.BytesIO() as f:  # double memory usage
            async with response["Body"] as stream:
                data = await stream.read()
                f.write(data)
                f.seek(0)
            return joblib.load(f)

    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model not found")


async def save_model(s3: ClientCreatorContext, topic_model: BERTopic) -> uuid.UUID:
    model_id = str(uuid.uuid4())

    with io.BytesIO() as f:
        joblib.dump(topic_model, f)
        f.seek(0)
        await s3.put_object(Bucket=settings.MINIO_BUCKET_NAME, Key=model_id, Body=f.read())
    return model_id


@router.post("/fit", summary="Run topic modeling", response_model=ModelId)
async def fit(
    data: Input,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: Session = Depends(deps.get_db),
) -> ModelId:
    model_id = str(uuid.uuid4())
    topic_model = BERTopic()
    if data.texts:
        topics, probs = topic_model.fit_transform(data.texts)
    else:
        docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"][
            :100
        ]
        topics, probs = topic_model.fit_transform(docs)

    model_id = await save_model(s3, topic_model)

    topic_info = topic_model.get_topic_info()
    topic_info.columns = topic_info.columns.str.lower()
    for topic_info in topic_info.to_dict("records"):
        session.add(models.Topic(model_id=model_id, **topic_info))
    session.commit()
    return ModelId(model_id=model_id)


@router.post("/predict", summary="Predict with existing model", response_model=ModelPrediction)
async def predict(
    data: NotEmptyInput,
    model_id: UUID4 = Query(...),
    calculate_probabilities: bool = Query(default=False),
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> ModelPrediction:
    topic_model = await load_model(s3, model_id)
    topic_model.calculate_probabilities = calculate_probabilities
    topics, probabilities = topic_model.transform(data.texts)
    if probabilities is not None:
        probabilities = probabilities.tolist()
    return ModelPrediction(topics=topics, probabilities=probabilities)


@router.get("/models", summary="Get all existing model ids", response_model=List[str])
async def list_models(s3: ClientCreatorContext = Depends(deps.get_s3)) -> List[str]:
    paginator = s3.get_paginator("list_objects")
    async for result in paginator.paginate(Bucket=settings.MINIO_BUCKET_NAME):
        return [x["Key"] for x in result.get("Contents", [])]


@router.get("/topics", summary="Get topics", response_model=List[TopicTopWords])
async def get_topics(
    model_id: UUID4 = Query(...), s3: ClientCreatorContext = Depends(deps.get_s3)
) -> List[TopicTopWords]:
    topic_model = await load_model(s3, model_id)
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


@router.get("/topics_info", summary="Get topics info", response_model=List[models.TopicBase])
async def get_topics_info(
    model_id: UUID4 = Query(...), session: Session = Depends(deps.get_db)
) -> List[models.Topic]:
    return session.exec(
        select(models.Topic)
        .filter(models.Topic.model_id == model_id)
        .order_by(-models.Topic.count)
    ).all()


@router.get("/remove_model", summary="Remove topic model")
async def remove_model(
    model_id: UUID4 = Query(...), s3: ClientCreatorContext = Depends(deps.get_s3)
) -> str:
    try:
        await s3.delete_object(Bucket=settings.MINIO_BUCKET_NAME, Key=str(model_id))
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model not found")
    return "ok"
