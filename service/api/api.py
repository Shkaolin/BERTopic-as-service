from fastapi import APIRouter

from service.api.endpoints import model_training

api_router = APIRouter()
api_router.include_router(model_training.router, tags=["model_training"])
