from fastapi import APIRouter

from .endpoints import model_training, visualization

tags_metadata = [
    {
        "name": "model_training",
        "description": "Model training/observing",
    },
    {
        "name": "visualization",
        "description": "Topics visualizations",
    },
]

api_router = APIRouter()
api_router.include_router(model_training.router)
api_router.include_router(visualization.router)
