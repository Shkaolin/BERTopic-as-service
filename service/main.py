from fastapi import FastAPI

from service.api.api import api_router
from service.db.db import init_db

app = FastAPI()

app.include_router(api_router)


@app.on_event("startup")
def on_startup():
    init_db()
