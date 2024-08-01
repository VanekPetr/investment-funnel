# https://github.com/tiangolo/uvicorn-gunicorn-machine-learning-docker
FROM python:3.11

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    POETRY_VERSION=1.7.1

# install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 - && \
    cd /usr/local/bin && \
    ln -s /etc/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /app

# Copy files necessary for installing dependencies
COPY pyproject.toml poetry.lock /

# Install dependencies using Poetry
RUN poetry install --no-interaction --no-ansi --no-cache

# Copy funnel application code
COPY funnel ./funnel

# If your funnel app.py is exposing the application as `server`, change as necessary
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8050", "-t", "1200", "funnel.app:server"]

# TODO: To deploy on AWS App Runner, build the image with the following command
# docker buildx build --platform linux/amd64 -t investment-funnel --load -f docker/Dockerfile .
