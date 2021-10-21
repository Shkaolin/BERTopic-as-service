from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SERVER_NAME: str = "bertopic_as_service"
    DATA_DIR: Path = Path().resolve() / "data"

    class Config:
        case_sensitive = True


settings = Settings()
