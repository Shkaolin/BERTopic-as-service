from fastapi import FastAPI
from fastapi_pagination import add_pagination

from .api.api import api_router

app = FastAPI()
app.include_router(api_router)
add_pagination(app)
