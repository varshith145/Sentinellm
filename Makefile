# SentinelLM — Makefile
# Run `make help` to see all available commands.

.PHONY: help up down logs train mock-model test lint

## ── General ──────────────────────────────────────────────────
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

## ── Stack ────────────────────────────────────────────────────
up: ## Start the full stack (gateway + db + admin dashboard)
	docker compose up -d
	@echo ""
	@echo "  Gateway:   http://localhost:8000"
	@echo "  Dashboard: http://localhost:8501"
	@echo ""

down: ## Stop the full stack
	docker compose down

logs: ## Stream logs from all services
	docker compose logs -f

restart-gateway: ## Restart only the gateway (e.g. after model update)
	docker compose restart gateway

## ── Model Training ───────────────────────────────────────────
train: ## Run the full training pipeline (prepare → train → evaluate) via Docker
	@bash train.sh

mock-model: ## Create a mock model for development (random weights, fast to load)
	python3 model/create_mock_model.py

## ── Development ──────────────────────────────────────────────
test: ## Run the test suite
	cd gateway && python3 -m pytest ../tests/ -v --tb=short

lint: ## Run ruff linter
	cd gateway && python3 -m ruff check . --fix

typecheck: ## Run mypy type checker
	cd gateway && python3 -m mypy app/

## ── Database ─────────────────────────────────────────────────
db-shell: ## Open a psql shell in the running db container
	docker compose exec db psql -U ppg -d ppg

db-reset: ## Drop and recreate the audit_log table (WARNING: deletes all audit data)
	docker compose exec db psql -U ppg -d ppg -c "DROP TABLE IF EXISTS audit_log;"
	docker compose restart gateway
