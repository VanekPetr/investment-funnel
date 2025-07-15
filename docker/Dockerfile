################################################################################
# BUILDER STAGE
################################################################################
# This stage is responsible for installing all dependencies and building any
# necessary artifacts. Using a multi-stage build reduces the final image size
# by not including build tools in the runtime image.
################################################################################
FROM python:3.12-slim as builder

# We don't need build dependencies because:
# 1. Modern Python package repositories provide pre-built wheels for most packages
# 2. UV (used below) can install these pre-built wheels without compilation
# 3. If a package doesn't have a pre-built wheel, UV will try to find an alternative
# Clean up apt cache to reduce image size
RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy UV installer from the official UV image
# UV is a fast Python package installer and resolver, used as an alternative to pip
# We copy both 'uv' (the main tool) and 'uvx' (the extended version with additional features)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory for the build stage
WORKDIR /app

# Copy only the dependency files needed for installation
# This optimizes build caching - if these files don't change, Docker can use cached layers
# uv.lock: Contains exact versions of all dependencies (like package-lock.json in Node.js)
# pyproject.toml: Defines project metadata and dependencies
# README.md: Included as it's referenced in pyproject.toml
COPY uv.lock pyproject.toml README.md ./

# Create a virtual environment and install dependencies
# First command creates a virtual environment at /app/.venv
# Second command installs all dependencies defined in uv.lock (--frozen ensures exact versions)
RUN uv venv /app/.venv
RUN uv sync --frozen

################################################################################
# RUNTIME STAGE
################################################################################
# This stage contains only what's necessary to run the application,
# resulting in a smaller and more secure final image
################################################################################
FROM python:3.12-slim

# Set environment variables
# PYTHONFAULTHANDLER=1: Helps debug segfaults by displaying tracebacks
# PYTHONUNBUFFERED=1: Ensures Python output is sent straight to terminal (helps with logging)
# PYTHONHASHSEED=random: Randomizes hash seed for security (prevents hash collision attacks)
# PORT=8050: Default port for the application (used by Dash)
# PATH: Adds the virtual environment's bin directory to PATH so we can use its executables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PORT=8050 \
    PATH="/app/.venv/bin:$PATH"

# Set the working directory for the runtime stage
WORKDIR /app

# Copy the virtual environment from the builder stage
# This contains all our dependencies without the build tools
COPY --from=builder /app/.venv /app/.venv

# Install gunicorn for production serving
# We install it separately as it's a production-only dependency
# Using the virtual environment's pip ensures it's installed in the right place
RUN /app/.venv/bin/pip install gunicorn

# Copy the application code
# This is done after installing dependencies to optimize rebuild times
# If your code changes but dependencies don't, Docker will reuse cached layers
COPY . .

# Create a non-root user for security
# Running as non-root is a security best practice to limit potential damage if container is compromised
# We also change ownership of the /app directory to this user
RUN useradd -m myuser && chown -R myuser:myuser /app
USER myuser

# Expose the port the application will run on
# This is a documentation feature and doesn't actually publish the port
EXPOSE ${PORT}

# Start the application with gunicorn
# --bind: Binds to all interfaces on the specified port
# --timeout: Sets worker timeout to 1200 seconds (20 minutes) for long-running operations
# --workers: Uses 4 worker processes (typically 2-4 × num_cores)
# --worker-class: Uses sync worker type (alternatives: gevent, eventlet for async)
# --access-logfile/-: Logs access to stdout
# --error-logfile/-: Logs errors to stdout
# funnel.app:server: Points to the Flask server instance in the funnel.app module
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--timeout", "1200", \
     "--workers", "4", \
     "--worker-class", "sync", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "funnel.app:server"]

################################################################################
# DEPLOYMENT NOTES
################################################################################
# To deploy on AWS App Runner, build the image with the following command:
# docker buildx build --platform linux/amd64 -t investment-funnel --load -f docker/Dockerfile .
#
# For local development, you can build and run with:
# docker build -t investment-funnel -f docker/Dockerfile .
# docker run -p 8050:8050 investment-funnel
################################################################################
