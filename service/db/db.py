from sqlmodel import SQLModel, create_engine

from service import models  # NOQA
from service.core.config import settings

# connect_args = {"check_same_thread": False}
db_url = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

engine = create_engine(db_url)


def init_db():
    SQLModel.metadata.create_all(engine)
