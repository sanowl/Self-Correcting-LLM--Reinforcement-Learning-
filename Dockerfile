# Multi-stage build for optimized production image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS builder

# Set working directory
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Production stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS production

# Create non-root user for security
RUN groupadd -r scoreuser && useradd -r -g scoreuser scoreuser

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories and set permissions
RUN mkdir -p data outputs logs \
    && chown -R scoreuser:scoreuser /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV SCORE_ENV=production
ENV SCORE_LOG_DIR=/app/logs
ENV SCORE_OUTPUT_DIR=/app/outputs
ENV SCORE_DATA_PATH=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Switch to non-root user
USER scoreuser

# Expose port for monitoring (if needed)
EXPOSE 8080

# Default command
ENTRYPOINT ["python", "main.py"] 