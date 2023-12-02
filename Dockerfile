FROM python:3.9.12-slim-buster as base

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
WORKDIR /app



FROM base as builder
RUN apt-get update  && \
    apt-get install -y apt-utils build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean


RUN pip install --upgrade pip
RUN pip install poetry==1.5.0



COPY pyproject.toml poetry.lock ./


RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root



FROM base as production
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN pip install --upgrade pip
RUN pip install tensorflow==2.15.0


COPY src ./src
COPY raw_data ./raw_data




CMD  ["/bin/sh", "-ec", "sleep infinity"]
# CMD ["python", "-m", "src.model_comparison"]