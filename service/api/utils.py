import io
import uuid

import joblib
from aiobotocore.session import ClientCreatorContext
from bertopic import BERTopic
from fastapi.exceptions import HTTPException
from pydantic.types import UUID4

from service.core.config import settings


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
