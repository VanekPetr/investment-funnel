# https://github.com/tiangolo/uvicorn-gunicorn-machine-learning-docker
FROM python:3.10

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install requirements
COPY requirements.txt .
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt

# Copy source files
WORKDIR /app
COPY . .

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8050", "-t", "1200", "app:server"]
