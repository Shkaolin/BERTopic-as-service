from typing import List, Sequence, Union

from aiobotocore.session import ClientCreatorContext
from fastapi import Depends, Path
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from fastapi_pagination import LimitOffsetPage
from fastapi_pagination.bases import AbstractPage
from pydantic.types import UUID4
from sqlalchemy.exc import NoResultFound
from sqlmodel.ext.asyncio.session import AsyncSession

from ... import crud
from ...api import deps
from ...core.config import settings
from ...models import models
from ...schemas.base import Message

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/", summary="Get existing models", response_model=LimitOffsetPage[models.TopicModelBase]
)
async def list_models(
    session: AsyncSession = Depends(deps.get_db_async),
) -> AbstractPage[models.TopicModel]:
    return await crud.topic_model.paginate(session)


@router.get("/{model_id}/", summary="Get topics", response_model=List[models.TopicBase])
async def get_topics(
    model_id: UUID4 = Path(...),
    version: int = 1,
    session: AsyncSession = Depends(deps.get_db_async),
) -> Sequence[models.Topic]:
    result: Sequence[models.Topic] = await crud.topic.get_model_topics(
        session, model_id=model_id, version=version
    )
    return result


@router.get(
    "/{model_id}/topics",
    summary="Get topics with words",
    response_model=List[models.TopicWithWords],
)
async def get_topics_info(
    model_id: UUID4 = Path(...),
    version: int = 1,
    session: AsyncSession = Depends(deps.get_db_async),
) -> Sequence[models.TopicWithWords]:
    result: Sequence[models.TopicWithWords] = await crud.topic.get_model_topics(
        session, model_id=model_id, version=version, with_words=True
    )
    return result


@router.delete(
    "/{model_id}",
    summary="Remove topic model",
    responses={404: {"model": Message}},
    response_model=Message,
)
async def remove_model(
    model_id: UUID4 = Path(...),
    version: int = 1,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Union[Message, JSONResponse]:
    try:
        await crud.topic_model.remove_by_id_version(session, model_id=model_id, version=version)
        await s3.delete_object(Bucket=settings.MINIO_BUCKET_NAME, Key=str(model_id))
    except (NoResultFound, s3.exceptions.NoSuchKey):
        return JSONResponse(status_code=404, content=dict(Message(message="Model not found")))
    return Message(message="ok")
