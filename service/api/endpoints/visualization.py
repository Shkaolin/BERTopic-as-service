from typing import Any

from aiobotocore.session import ClientCreatorContext
from fastapi import Depends
from fastapi.routing import APIRouter

from service.api import deps
from service.schemas.base import BarchartInput, HeatmapInput, HierarchyInput, IntertopicInput

router = APIRouter(
    prefix="/visualization",
    tags=["visualization"],
    responses={404: {"description": "Not found"}},
)


@router.post("/intertopic", summary="Intertopic Distance Map")
async def intertopic(
    data: IntertopicInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> Any:
    topic_model = await deps.load_model(s3, data.model.model_id, data.model.version)
    return topic_model.visualize_topics(
        topics=data.topics,
        top_n_topics=data.top_n_topics,
        width=data.width,
        height=data.height,
    ).to_json()


@router.post("/barchart", summary="Topic Word Scores")
async def barchart(
    data: BarchartInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> Any:
    topic_model = await deps.load_model(s3, data.model.model_id, data.model.version)
    return topic_model.visualize_barchart(
        topics=data.topics,
        top_n_topics=data.top_n_topics,
        n_words=data.n_words,
        width=data.width,
        height=data.height,
    ).to_json()


@router.post("/hierarchy", summary="Hierarchical Clustering")
async def hierarchy(
    data: HierarchyInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> Any:
    topic_model = await deps.load_model(s3, data.model.model_id, data.model.version)
    return topic_model.visualize_hierarchy(
        orientation=data.orientation,
        topics=data.topics,
        top_n_topics=data.top_n_topics,
        width=data.width,
        height=data.height,
    ).to_json()


@router.post("/heatmap", summary="Similarity Matrix")
async def heatmap(
    data: HeatmapInput,
    s3: ClientCreatorContext = Depends(deps.get_s3),
) -> Any:
    topic_model = await deps.load_model(s3, data.model.model_id, data.model.version)
    return topic_model.visualize_heatmap(
        topics=data.topics,
        top_n_topics=data.top_n_topics,
        n_clusters=data.n_clusters,
        width=data.width,
        height=data.height,
    ).to_json()
