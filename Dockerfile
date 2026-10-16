# Use standard Python 3.11 slim image for minimal footprint
FROM python:3.11-slim

# Avoid writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501 \
    TRAFFIC_SIGN_WEIGHTS="weights/yolo11n_baseline_best.pt"

# Install system dependencies required for OpenCV and media handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deployment dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY app.py .

# Copy production recommended checkpoint
COPY weights/yolo11n_baseline_best.pt ./weights/yolo11n_baseline_best.pt

# Expose Streamlit port
EXPOSE 8501

# Streamlit healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit server bound to 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
