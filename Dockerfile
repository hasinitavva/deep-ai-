# Production Dockerfile for SENTINEL AI Deepfake Detector
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU build first to keep image lightweight and fast to build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ backend/
COPY ml/ ml/
COPY frontend/ frontend/
COPY models/ models/
COPY data/ data/

# Ensure output and DB directories exist with write permissions
RUN mkdir -p outputs/heatmaps outputs/reports backend/database

EXPOSE 8000

# Health check to ensure API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI application
CMD ["python", "backend/main.py"]
