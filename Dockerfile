# ==============================================================================
# NutriScanAI Production Multi-Service Container
# Compatible with Hugging Face Spaces (Free 16GB RAM), Render, Railway, Cloud Run
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files, buffer stdout/stderr, and cap memory fragmentation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# 1. Install lightweight CPU-only PyTorch & TorchVision (keeps image ~400MB instead of 4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code, models, and frontend
COPY . .

# Make startup script executable and configure non-root user permissions
RUN chmod +x /app/start.sh && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["/app/start.sh"]
