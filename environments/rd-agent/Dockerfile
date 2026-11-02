FROM ghcr.io/astral-sh/uv:0.12.18 AS uv
FROM python:3.11.16-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/native
COPY environments/rd-agent/pyproject.toml environments/rd-agent/uv.lock ./
RUN uv sync --locked --no-dev
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
COPY src /opt/quantagent/src
COPY environments/rd-agent/offline_smoke.py /opt/quantagent/offline_smoke.py
ENV PATH="/opt/native/.venv/bin:$PATH" PYTHONPATH="/opt/quantagent" PYTHONDONTWRITEBYTECODE=1 QUANTAGENT_NATIVE_SANDBOX=1
RUN useradd --uid 10001 --create-home native
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["python", "-m", "src.research.native_runtime.entrypoint"]
