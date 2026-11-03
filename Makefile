.PHONY: install test lint build build-quick api web dev migrate docker-build legacy load-test

install:            ## Python + frontend dependencies
	poetry install --with dev
	cd frontend && npm ci

test:               ## all tests (legacy + v4)
	poetry run python -m pytest -q -W ignore

lint:
	poetry run ruff check src/ntg/core src/ntg/datasets src/ntg/recsys src/ntg/tastetwin src/ntg/intent src/ntg/discovery src/ntg/journey src/ntg/enterprise src/ntg/platform src/ntg/api src/ntg/streaming src/ntg/pipelines/v4.py src/ntg/cli.py tests/v4 migrations
	cd frontend && npm run typecheck

build:              ## synthetic data -> tuned models -> reports/v4/synthetic -> artifacts/synthetic (~2-3 min)
	poetry run python -m ntg.cli build --dataset synthetic

build-quick:
	poetry run python -m ntg.cli build --dataset synthetic --quick

build-ml32m:        ## real MovieLens 32M placed at data/external/movielens/ml-32m (10% user sample)
	poetry run python -m ntg.cli build --dataset ml-32m --source data/external/movielens/ml-32m --user-sample 0.1 --min-item-support 20

migrate:
	poetry run alembic upgrade head

api:                ## http://127.0.0.1:8000/docs
	poetry run python -m ntg.cli serve

web:                ## http://localhost:3000 (expects the API on :8000)
	cd frontend && npm run dev

load-test:
	poetry run python scripts/api_load_test.py

docker-build:
	docker build -t ntg-api:4 . && docker build -t ntg-web:4 frontend

legacy:             ## original v1 DuckDB pipeline (needs ML-32M)
	poetry run python -m ntg.cli legacy
