from pydantic import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SERVER_NAME: str = "bertopic_as_service"

    MINIO_URL: str
    MINIO_REGION_NAME: str
    MINIO_BUCKET_NAME: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str

    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
