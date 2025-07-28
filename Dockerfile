# Use Python 3.10 slim image for smaller size
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/.cache

# Copy application code
COPY app/ /app/app/
COPY main.py /app/
COPY README.md /app/

# Set permissions
RUN chmod +x /app/main.py

# Pre-download model for faster startup (optional, comment if size is concern)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('distiluse-base-multilingual-cased-v1')" || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "main.py"]
