from typing import Any, Dict, List, Optional, Sequence

import io
import uuid

import joblib
import numpy as np
from aiobotocore.session import ClientCreatorContext
from bertopic import BERTopic
from fastapi import Depends, Query
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter
from pydantic.types import UUID4
from sklearn.datasets import fetch_20newsgroups
from sqlalchemy import func
from sqlalchemy.exc import NoResultFound
from sqlmodel.ext.asyncio.session import AsyncSession

from service import crud
from service.api import deps
from service.core.config import settings
from service.models import models
from service.schemas.base import (
    DocsWithPredictions,
    FitResult,
    Input,
    ModelPrediction,
    NotEmptyInput,
)
from service.schemas.bertopic_wrapper import BERTopicWrapper

router = APIRouter()


def get_model_filename(model_id: UUID4, version: int = 1) -> str:
    return f"{model_id}_{version}"


async def load_model(s3: ClientCreatorContext, model_id: uuid.UUID, version: int = 1) -> BERTopic:
    try:
        model_name = get_model_filename(model_id, version)
        response = await s3.get_object(Bucket=settings.MINIO_BUCKET_NAME, Key=model_name)

        with io.BytesIO() as f:  # double memory usage
            async with response["Body"] as stream:
                data = await stream.read()
                f.write(data)
                f.seek(0)
            return joblib.load(f)

    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Model not found")


async def save_model(
    s3: ClientCreatorContext,
    topic_model: BERTopic,
    model_id: Optional[uuid.UUID] = None,
    version: int = 1,
) -> uuid.UUID:
    if model_id is None:
        model_id = uuid.uuid4()
    model_name = get_model_filename(model_id, version)

    with io.BytesIO() as f:
        joblib.dump(topic_model, f)
        f.seek(0)
        await s3.put_object(Bucket=settings.MINIO_BUCKET_NAME, Key=model_name, Body=f.read())
    return model_id


async def gather_topics(topic_model: BERTopic) -> List[Dict[str, Any]]:
    topic_info = topic_model.get_topics()
    topics = []
    for topic_index, top_words in topic_info.items():
        topics.append(
            {
                "name": topic_model.topic_names[topic_index],
                "count": topic_model.topic_sizes[topic_index],
                "topic_index": topic_index,
                "top_words": [{"name": w[0], "score": w[1]} for w in top_words],
            }
        )
    return topics


def get_sample_dataset():
    return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"][:100]


@router.post("/fit", summary="Run topic modeling", response_model=FitResult)
async def fit(
    data: Input,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> FitResult:
    topic_model = BERTopicWrapper(
        language=data.language,
        top_n_words=data.top_n_words,
        nr_topics=data.nr_topics,
        calculate_probabilities=data.calculate_probabilities,
        seed_topic_list=data.seed_topic_list,
        vectorizer_params=data.vectorizer_params,
        umap_params=data.umap_params,
        hdbscan_params=data.hdbscan_params,
        verbose=data.verbose,
    ).model
    if data.texts:
        predicted_topics, probs = topic_model.fit_transform(data.texts)
    else:
        docs = get_sample_dataset()
        predicted_topics, probs = topic_model.fit_transform(docs)

    model_id = await save_model(s3, topic_model)
    topics = await gather_topics(topic_model)
    model = await crud.topic_model.create(session, obj_in=models.TopicModelBase(model_id=model_id))
    await crud.topic.save_topics(session, topics=topics, model=model)

    return FitResult(
        model_id=model_id,
        predictions=ModelPrediction(topics=predicted_topics, probabilities=probs.tolist()),
    )


@router.post("/predict", summary="Predict with existing model", response_model=ModelPrediction)
async def predict(
    data: NotEmptyInput,
    model_id: UUID4 = Query(...),
    version: int = 1,
    calculate_probabilities: bool = Query(default=False),
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> ModelPrediction:
    topic_model = await load_model(s3, model_id, version)
    topic_model.calculate_probabilities = calculate_probabilities
    topics, probabilities = topic_model.transform(data.texts)
    if probabilities is not None:
        probabilities = probabilities.tolist()
    return ModelPrediction(topics=topics, probabilities=probabilities)


@router.post(
    "/reduce_topics",
    summary="Reduce number of topics in existing model",
    response_model=FitResult,
)
async def reduce_topics(
    data: DocsWithPredictions,
    model_id: UUID4 = Query(...),
    version: int = Query(default=1),
    num_topics: int = Query(...),
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> FitResult:
    topic_model = await load_model(s3, model_id, version)
    if len(topic_model.get_topics()) < num_topics:
        raise HTTPException(
            status_code=400, detail=f"num_topics must be less than {len(topic_model.get_topics())}"
        )

    if len(data.texts) == 0:
        data.texts = get_sample_dataset()
    predicted_topics, probs = topic_model.reduce_topics(
        docs=data.texts,
        topics=data.topics,
        probabilities=np.array(data.probabilities),
        nr_topics=num_topics,
    )
    current_max_version = await crud.topic_model.get_max_version(session, model_id=model_id)

    model_id = await save_model(s3, topic_model, model_id, current_max_version + 1)
    topics = await gather_topics(topic_model)
    model = await crud.topic_model.create(
        session, obj_in=models.TopicModelBase(model_id=model_id, version=current_max_version + 1)
    )
    await crud.topic.save_topics(session, topics=topics, model=model)

    return FitResult(
        model_id=model_id,
        version=current_max_version + 1,
        predictions=ModelPrediction(topics=predicted_topics, probabilities=probs.tolist()),
    )


@router.get(
    "/models", summary="Get all existing model ids", response_model=List[models.TopicModelBase]
)
async def list_models(
    skip: int = Query(default=0),
    limit: int = Query(default=100),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Sequence[models.TopicModel]:
    result: Sequence[models.TopicModel] = await crud.topic_model.get_multi(
        session, skip=skip, limit=limit
    )
    return result


@router.get("/topics", summary="Get topics", response_model=List[models.TopicWithWords])
async def get_topics(
    model_id: UUID4 = Query(...),
    version: int = 1,
    session: AsyncSession = Depends(deps.get_db_async),
) -> Sequence[models.TopicWithWords]:
    result: Sequence[models.TopicWithWords] = await crud.topic.get_model_topics(
        session, model_id=model_id, version=version, with_words=True
    )
    return result


@router.get("/topics_info", summary="Get topics info", response_model=List[models.TopicBase])
async def get_topics_info(
    model_id: UUID4 = Query(...),
    version: int = 1,
    session: AsyncSession = Depends(deps.get_db_async),
) -> Sequence[models.Topic]:
    result: Sequence[models.Topic] = await crud.topic.get_model_topics(
        session, model_id=model_id, version=version
    )
    return result


@router.get("/remove_model", summary="Remove topic model")
async def remove_model(
    model_id: UUID4 = Query(...),
    version: int = 1,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> str:
    try:
        await crud.topic_model.remove_by_id_version(session, model_id=model_id, version=version)
        await s3.delete_object(Bucket=settings.MINIO_BUCKET_NAME, Key=str(model_id))
    except (NoResultFound, s3.exceptions.NoSuchKey):
        raise HTTPException(status_code=404, detail="Model not found")
    return "ok"
