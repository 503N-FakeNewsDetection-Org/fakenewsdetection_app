FROM python:3.11-slim

# Optional: change if you have a private PyPI mirror
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy source code first (for layer caching)
COPY . ./text_model/

# Install minimal system deps
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --upgrade pip && pip install -r text_model/requirements.txt

# Expose nothing – internal container
ENV PORT=8001
CMD ["uvicorn", "text_model.app:app", "--host", "0.0.0.0", "--port", "8001"]
