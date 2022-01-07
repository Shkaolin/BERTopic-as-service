from typing import Optional

import io
import uuid

import joblib
from aiobotocore.session import ClientCreatorContext
from bertopic import BERTopic
from fastapi.exceptions import HTTPException
from pydantic.types import UUID4
from sklearn.datasets import fetch_20newsgroups
from sqlmodel.ext.asyncio.session import AsyncSession

from service.core.config import settings
from service.models import models


def get_sample_dataset():
    return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"][:100]


def get_model_filename(model_id: UUID4, version: int = 1) -> str:
    return f"{model_id}_{version}"


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


async def save_topics(
    topic_model: BERTopic, session: AsyncSession, model: models.TopicModel
) -> None:
    topic_info = topic_model.get_topics()
    for topic_index, top_words in topic_info.items():
        topic = models.Topic(
            name=topic_model.topic_names[topic_index],
            count=topic_model.topic_sizes[topic_index],
            topic_index=topic_index,
            topic_model=model,
            top_words=[models.Word(name=w[0], score=w[1]) for w in top_words],
        )
        session.add(topic)
