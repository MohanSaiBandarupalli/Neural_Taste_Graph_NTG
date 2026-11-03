# NTG API + pipeline image (Python 3.11). Multi-stage: deps are cached separately from source.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1
# libgomp1: OpenMP runtime required by LightGBM wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir "poetry==2.0.0"
COPY pyproject.toml poetry.lock /app/
RUN poetry install --only main --with postgres --no-root --no-ansi

FROM base AS runtime
COPY README.md LICENSE /app/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
RUN pip install --no-cache-dir --no-deps -e . \
    && useradd --create-home --uid 10001 ntg && mkdir -p /app/data /app/artifacts /app/reports && chown -R ntg /app
USER ntg
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
# Default: serve the API. The bundle must exist (run `ntg build` once; see docker-compose.yml `build-models`).
CMD ["uvicorn", "ntg.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
