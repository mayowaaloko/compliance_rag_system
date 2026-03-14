# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Compliance RAG API
# ─────────────────────────────────────────────────────────────────────────────
#
# What is a Dockerfile?
# It is a recipe that tells Docker how to package our application into a
# self-contained image. Anyone with Docker installed can run the same image
# on any machine and get identical behaviour — no "works on my machine" issues.
#
# Build stages:
#   Stage 1 (builder): Install all Python dependencies into a virtual env.
#                      This stage produces a heavy image we don't ship.
#   Stage 2 (runtime): Copy only the virtual env + our code into a slim image.
#                      This stage produces the final ~600MB image we deploy.
#
# Why multi-stage?
# The builder installs compilers and headers needed to build packages like
# torch and pymupdf. Those tools are not needed at runtime — including them
# would make the final image 2x larger for no benefit.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Why python:3.11-slim?
# - 3.11 matches the notebook's kernel version (kernelspec shows 3.11.14)
# - slim removes unnecessary system packages, reducing the base image size

# Set working directory inside the container
WORKDIR /build

# Install system-level build dependencies
# These are needed to compile packages like pymupdf, torch, and cryptography
# but are NOT needed at runtime — they stay in the builder stage only
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment inside the container
# This makes it easy to copy to the runtime stage in one go
RUN python -m venv /opt/venv

# Activate the virtual env for all subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first (before the code)
# Docker caches each layer. If requirements.txt hasn't changed,
# Docker skips the pip install step on the next build — much faster.
COPY requirements.txt .

# Install all Python dependencies into the virtual env
# --no-cache-dir: don't cache downloads inside the image (saves ~200MB)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install only the runtime system libraries that pymupdf needs
# (libmupdf provides the underlying PDF rendering engine)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
# Create a non-root user for security
# Running as root inside a container is a security anti-pattern:
# if the container is compromised, the attacker has root access.
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy the virtual env from the builder stage (all installed packages)
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY app/ ./app/

# Copy the frontend UI so FastAPI can serve it at /
COPY ui/ ./ui/

# Create the documents and cache directories and set ownership
# These directories are mounted as volumes in docker-compose.yml,
# but we create them here as fallbacks for standalone container runs
RUN mkdir -p ./documents ./.embedding_cache && \
    chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Tell Docker that the container listens on port 8000
EXPOSE 8000

# Add the virtual env to PATH so 'uvicorn' resolves correctly
ENV PATH="/opt/venv/bin:$PATH"

# Set Python environment variables
# PYTHONUNBUFFERED=1: print() output appears in Docker logs immediately (no buffering)
# PYTHONDONTWRITEBYTECODE=1: don't create .pyc files (saves a tiny amount of disk space)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check: Docker will poll this every 30 seconds.
# If it fails 3 times in a row, the container is marked "unhealthy"
# and docker-compose will restart it (or a k8s probe will trigger a restart).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# The command to start the server when the container runs
# --host 0.0.0.0: listen on all network interfaces (not just localhost)
#                 Required inside Docker so the host machine can reach the API
# --port 8000:    The port the API listens on (must match EXPOSE above)
# --workers 1:    One worker process. For multi-core, increase this or use gunicorn.
# --reload is NOT used here — only in development (see docker-compose.yml override)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
