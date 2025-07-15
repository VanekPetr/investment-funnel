# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy UV installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files only
COPY uv.lock pyproject.toml README.md ./

# Install dependencies into a virtual environment
RUN uv venv /app/.venv
RUN uv sync --frozen

# Runtime stage
FROM python:3.11-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PORT=8050 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Install gunicorn
RUN /app/.venv/bin/pip install gunicorn

# Copy application code only in runtime stage
COPY . .

# Create non-root user
RUN useradd -m myuser && chown -R myuser:myuser /app
USER myuser

EXPOSE ${PORT}

CMD ["gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--timeout", "1200", \
     "--workers", "4", \
     "--worker-class", "sync", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "funnel.app:server"]

# TODO: To deploy on AWS App Runner, build the image with the following command
# docker buildx build --platform linux/amd64 -t investment-funnel --load -f docker/Dockerfile .
