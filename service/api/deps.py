from typing import AsyncGenerator, Generator

from aiobotocore.client import AioBaseClient
from aiobotocore.session import get_session
from sqlmodel import Session

from service.core.config import settings
from service.db.db import engine


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


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
