FROM python:3.12-slim
ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ENV FORECASTOPS_GIT_COMMIT=$GIT_COMMIT FORECASTOPS_GIT_DIRTY=$GIT_DIRTY
LABEL org.opencontainers.image.revision=$GIT_COMMIT
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
RUN useradd -m -u 10001 app && mkdir -p artifacts data && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "forecastops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
