FROM python:3.8.12-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONHASHSEED=random \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  POETRY_VERSION=1.1.11

WORKDIR /app

RUN apt-get update \
  && apt-get -y install gcc make curl \
  && pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
  && poetry export --without-hashes -f requirements.txt --dev \
  |  poetry run pip install -r /dev/stdin \
  && poetry debug
