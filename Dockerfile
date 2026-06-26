# =============================================================================
# Multi-stage Python Dockerfile for llm-bench (CLI)
# =============================================================================

# =============================================================================
# Stage 1: Build dependencies and install package
# =============================================================================
FROM python:3.13-slim AS builder

WORKDIR /app

# Copy uv from official image (faster and smaller than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Inject build version into the package
ARG APP_VERSION=dev

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Write the build version before installing
RUN echo "__version__ = \"${APP_VERSION}\"" > src/llm_bench/version.py

# Install package and dependencies (without dev packages)
RUN uv sync --frozen --no-dev --no-editable

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user for security (UID 10001)
RUN groupadd --gid 10001 appgroup && \
    useradd \
        --uid 10001 \
        --gid appgroup \
        --shell /bin/false \
        --no-create-home \
        appuser

# Copy virtual environment from builder (includes installed package)
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER appuser:appgroup

# CLI entry point
ENTRYPOINT ["llm-bench"]
