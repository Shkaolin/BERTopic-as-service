from typing import AsyncGenerator, Generator

import io
import uuid

import joblib
from aiobotocore.client import AioBaseClient
from aiobotocore.session import ClientCreatorContext, get_session
from bertopic import BERTopic
from fastapi.exceptions import HTTPException
from pydantic.types import UUID4
from sqlmodel import Session
from sqlmodel.ext.asyncio.session import AsyncSession

from service.core.config import settings
from service.db.db import engine, engine_async


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


async def get_s3() -> AsyncGenerator[AioBaseClient, None]:
    session = get_session()
    async with session.create_client(
        "s3",
        region_name=settings.MINIO_REGION_NAME,
        endpoint_url=settings.MINIO_URL,
        use_ssl=False,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
    ) as client:
        yield client


async def get_db_async() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine_async) as session:
        yield session


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
