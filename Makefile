RACE ?= 2025_monza
SPEED ?= 60

.PHONY: setup test lint data train backtest replay dev deploy

setup:
	cd backend && uv sync
	cd frontend && npm install

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

data:
	cd backend && uv run python -m ingest.fastf1_pull

train:
	cd backend && uv run python -m models.train

backtest:
	cd backend && uv run python -m backtest.replay_eval

replay:
	cd backend && DATA_SOURCE=replay REPLAY_RACE=$(RACE) REPLAY_SPEED=$(SPEED) \
		uv run python -m ingest.replay

dev:
	./scripts/dev.sh

deploy:
	@echo "Phase 8: fly deploy (backend) + Cloudflare Pages (frontend)" && exit 1
