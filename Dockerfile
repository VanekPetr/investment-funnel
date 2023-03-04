# https://github.com/tiangolo/uvicorn-gunicorn-machine-learning-docker
FROM python:3.8

ARG PROD_ENV=production
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* \

# Create a working directory.
RUN mkdir wd
WORKDIR wd

# Install requirements
COPY requirements.txt .
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt

# Copy source files
COPY app /app
WORKDIR /app

ENTRYPOINT ["/bin/bash", "-c"]