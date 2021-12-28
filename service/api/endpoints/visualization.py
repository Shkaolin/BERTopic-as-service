from typing import Any, List

import numpy as np
from aiobotocore.session import ClientCreatorContext
from fastapi import Depends
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from service.api import deps
from service.api.utils import load_model
from service.models import models
from service.schemas.base import (
    ModelId,
    VisBarchartInput,
    VisDistributionInput,
    VisHeatmapInput,
    VisHierarchyInput,
    VisTermRankInput,
    VisTopicsInput,
)

router = APIRouter(
    prefix="/visualization",
    tags=["visualization"],
    responses={404: {"description": "Not found"}},
)


async def check_topics(
    model: ModelId,
    topics: List[int],
    session: AsyncSession,
) -> None:
    model_topics = (
        (
            await session.execute(
                select(models.Topic)
                .join(models.TopicModel)
                .filter(
                    models.TopicModel.model_id == model.model_id,
                    models.TopicModel.version == model.version,
                )
                .with_only_columns(models.Topic.topic_index)
            )
        )
        .scalars()
        .all()
    )
    topics_oor = set(topics) - set(model_topics)
    if topics_oor:
        raise HTTPException(
            status_code=400, detail=f"Indexes in topics list are out of range - {topics_oor}"
        )


@router.post("/topics", summary="Visualize topics, their sizes, and their corresponding words")
async def topics(
    data: VisTopicsInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Any:
    params = dict(data)
    model = params.pop("model")
    await check_topics(model, params["topics"], session)
    topic_model = await load_model(s3, model.model_id, model.version)
    return topic_model.visualize_topics(**params).to_json()


@router.post("/barchart", summary="Visualize a barchart of selected topics")
async def barchart(
    data: VisBarchartInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Any:
    params = dict(data)
    model = params.pop("model")
    await check_topics(model, params["topics"], session)
    topic_model = await load_model(s3, model.model_id, model.version)
    return topic_model.visualize_barchart(**params).to_json()


@router.post("/hierarchy", summary="Visualize a hierarchical structure of the topics")
async def hierarchy(
    data: VisHierarchyInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Any:
    params = dict(data)
    model = params.pop("model")
    await check_topics(model, params["topics"], session)
    topic_model = await load_model(s3, model.model_id, model.version)
    return topic_model.visualize_hierarchy(**params).to_json()


@router.post("/heatmap", summary="Visualize a heatmap of the topic's similarity matrix")
async def heatmap(
    data: VisHeatmapInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Any:
    params = dict(data)
    model = params.pop("model")
    await check_topics(model, params["topics"], session)
    topic_model = await load_model(s3, model.model_id, model.version)
    return topic_model.visualize_heatmap(**params).to_json()


@router.post("/distribution", summary="Visualize the distribution of topic probabilities")
async def distribution(
    data: VisDistributionInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> Any:
    params = dict(data)
    params["probabilities"] = np.array(data.probabilities)
    model = params.pop("model")
    topic_model = await load_model(s3, model.model_id, model.version)
    return topic_model.visualize_distribution(**params).to_json()


@router.post("/term_rank", summary="Visualize the ranks of all terms across all topics")
async def term_rank(
    data: VisTermRankInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
    session: AsyncSession = Depends(deps.get_db_async),
) -> Any:
    params = dict(data)
    model = params.pop("model")
    await check_topics(model, params["topics"], session)
    topic_model = await load_model(s3, model.model_id, model.version)
    return topic_model.visualize_term_rank(**params).to_json()
