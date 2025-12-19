# Multi-stage build for Interactive RAG Application
FROM python:3.11-slim as builder

# Build arguments
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

# Install runtime dependencies only
# libgl1 and libglib2.0-0 are required for OpenCV (used by docling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}

# Copy Python packages from builder
COPY --from=builder /root/.local /home/${APP_USER}/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} rag/ ./rag/

# Set PATH to include user's local bin
ENV PATH=/home/${APP_USER}/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the application
# Note: Flask app is in rag/app.py, so we need to set FLASK_APP
ENV FLASK_APP=rag/app.py
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5001"]

