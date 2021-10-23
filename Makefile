.PHONY: flake black test isort check-code build up down test
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

flake: ## Check formatting with flake8
	poetry run flakehell lint service tests

black: ## Format code with black
	poetry run black --config pyproject.toml service tests

isort: ## Check sorting with black
	poetry run isort --settings-path pyproject.toml service tests

check-code: flake black isort ## Run all code checks

build: ## Build compose
	docker-compose build

up: ## Start compose
	docker-compose up -d

down: ## Down compose
	docker-compose down

test: ## Run integration tests
	docker-compose run --rm bertopic pytest