from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, create_engine

from service.core.config import settings
from service.models import models  # NOQA

# connect_args = {"check_same_thread": False}
db_url = (
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

engine_async = create_async_engine(db_url)
engine = create_engine(db_url.replace("+asyncpg", ""))


async def init_db():
    async with engine_async.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
