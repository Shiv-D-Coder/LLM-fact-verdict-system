# Use an official lightweight Python image
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies needed by faiss-cpu and others
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libopenblas-dev \
        && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install spaCy model directly from GitHub
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy application code
# Note: .dockerignore ensures .env is not copied
COPY . ./

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Start the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false"]