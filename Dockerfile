# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.18 AS uv
FROM python:3.11.16-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra dashboard --extra alpaca --extra qlib
COPY src/ src/
COPY scripts/ scripts/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev --extra dashboard --extra alpaca --extra qlib

FROM python:3.11.16-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/outputs /app/logs \
    && chown -R appuser:appuser /app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser lean/ lean/
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH"
USER appuser
EXPOSE 8501
CMD ["python", "scripts/run_pipeline.py", "run", "--mode", "paper"]
