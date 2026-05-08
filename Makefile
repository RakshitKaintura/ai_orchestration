.PHONY: build up down restart logs test lint eval clean shell-api shell-db help

# ─── Docker ──────────────────────────────────────────────────────────────────

build:          ## Build all Docker images
	docker compose build

up:             ## Start all services (detached)
	docker compose up -d

up-logs:        ## Start all services and follow logs
	docker compose up

down:           ## Stop all services
	docker compose down

restart:        ## Restart all services
	docker compose down && docker compose up -d

rebuild:        ## Rebuild and restart (useful after code changes)
	docker compose down && docker compose build && docker compose up -d

# ─── Logs ─────────────────────────────────────────────────────────────────────

logs:           ## Tail logs for all services
	docker compose logs -f

logs-api:       ## Tail API server logs
	docker compose logs -f api

logs-worker:    ## Tail Celery worker logs
	docker compose logs -f worker

logs-db:        ## Tail database logs
	docker compose logs -f db

# ─── Development ──────────────────────────────────────────────────────────────

shell-api:      ## Open shell inside API container
	docker compose exec api bash

shell-db:       ## Open psql inside DB container
	docker compose exec db psql -U $${POSTGRES_USER} -d $${POSTGRES_DB}

shell-worker:   ## Open shell inside worker container
	docker compose exec worker bash

# ─── Testing & Quality ────────────────────────────────────────────────────────

test:           ## Run unit tests
	docker compose exec api pytest tests/ -v --tb=short

lint:           ## Run linters (ruff + mypy)
	docker compose exec api ruff check .
	docker compose exec api mypy . --ignore-missing-imports

format:         ## Auto-format code
	docker compose exec api ruff format .

# ─── Evaluation ───────────────────────────────────────────────────────────────

eval:           ## Run the full evaluation harness (15 test cases)
	docker compose exec api python -m eval.runner --all

eval-baseline:  ## Run only baseline eval cases
	docker compose exec api python -m eval.runner --category baseline

eval-adversarial: ## Run only adversarial eval cases
	docker compose exec api python -m eval.runner --category adversarial

eval-diff:      ## Show diff between last two eval runs
	docker compose exec api python -m eval.runner --diff

# ─── Database ─────────────────────────────────────────────────────────────────

db-migrate:     ## Re-apply schema (destructive in dev)
	docker compose exec db psql -U $${POSTGRES_USER} -d $${POSTGRES_DB} -f /docker-entrypoint-initdb.d/01_schema.sql

db-seed:        ## Re-seed sample data
	docker compose exec db psql -U $${POSTGRES_USER} -d $${POSTGRES_DB} -f /docker-entrypoint-initdb.d/02_seed.sql

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:          ## Remove containers, volumes, and images
	docker compose down -v --rmi local

clean-all:      ## Nuclear clean (removes ALL local images and volumes)
	docker compose down -v --rmi all

# ─── Help ─────────────────────────────────────────────────────────────────────

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
