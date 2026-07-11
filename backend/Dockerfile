# Pitwall backend — persistent process (MQTT/replay + FastAPI)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# libgomp for LightGBM wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

# model artifacts + replay data are mounted or baked in at deploy time:
#   models/  -> /models   (MODEL_DIR)
#   data/    -> /data     (PITWALL_DATA_DIR, replay mode only)
ENV MODEL_DIR=/models \
    PITWALL_DATA_DIR=/data \
    DATA_SOURCE=live \
    PORT=8000

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
