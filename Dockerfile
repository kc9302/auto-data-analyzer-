# Multi-stage ultra-fast container with uv
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using uv (10-100x faster than standard pip)
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application source
COPY configs/ ./configs/
COPY src/ ./src/
COPY run.py .

# Output volume mount points
VOLUME ["/app/dist", "/app/data"]

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
