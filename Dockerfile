FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# openenv from_docker_image() maps {random_host_port}:8000 (hardcoded)
# HF Spaces is told to use port 8000 via app_port in README metadata
ENV PORT=8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os, httpx; httpx.get(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health').raise_for_status()" || exit 1

# Run server
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT}
