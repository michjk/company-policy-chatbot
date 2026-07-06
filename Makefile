.PHONY: install test lint format typecheck check dev up down logs eval-data eval-opt eval-export eval

install:
	uv sync

test:
	uv run pytest

test/%:
	uv run pytest $*

lint:
	uv run ruff check --fix

format:
	uv run ruff format

typecheck:
	uv run pyrefly check

check:
	uv run pre-commit run --all-files

dev:
	uv run uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f app

eval-data:
	uv run python -m eval.generate_dataset

eval-opt:
	uv run python -m eval.optimize

eval-export:
	uv run python -m eval.export_prompts

eval:
	$(MAKE) eval-data && $(MAKE) eval-opt && $(MAKE) eval-export
