test:
	poetry run pytest -c pyproject.toml

run-dev:
	docker-compose up -d --build && env $(cat .env) poetry run python -m uvicorn service.main:app --reload