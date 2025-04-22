FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

# Install deps just once
RUN apt-get update && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 80
EXPOSE 9101

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
