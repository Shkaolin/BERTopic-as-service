from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import create_engine

from ..core.config import settings
from ..models import models  # NOQA

# connect_args = {"check_same_thread": False}
db_url = (
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

engine_async = create_async_engine(db_url)
engine = create_engine(db_url.replace("+asyncpg", ""))
