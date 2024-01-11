# https://github.com/tiangolo/uvicorn-gunicorn-machine-learning-docker
FROM python:3.10

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    POETRY_VERSION=1.7.1

# install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 - && \
    cd /usr/local/bin && \
    ln -s /etc/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# install dependencies
COPY ../pyproject.toml poetry.lock /
RUN poetry install --no-interaction --no-cache --no-ansi

# Copy source files
WORKDIR /app
COPY .. .

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8050", "-t", "1200", "app:server"]
