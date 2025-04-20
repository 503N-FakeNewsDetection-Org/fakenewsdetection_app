FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . ./image_model/

RUN apt-get update && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install -r image_model/requirements.txt

ENV PORT=8002
CMD ["uvicorn", "image_model.app:app", "--host", "0.0.0.0", "--port", "8002"]
