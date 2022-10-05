from fastapi import APIRouter

from .endpoints import base, modeling, models, visualization

tags_metadata = [
    {
        "name": "base",
        "description": "Service routes, not connected with BERTopic",
    },
    {
        "name": "modeling",
        "description": "Model training/predicting",
    },
    {
        "name": "models",
        "description": "Information about trained models",
    },
    {
        "name": "visualization",
        "description": "Topics visualizations",
    },
]

api_router = APIRouter()
api_router.include_router(base.router)
api_router.include_router(models.router)
api_router.include_router(modeling.router)
api_router.include_router(visualization.router)
