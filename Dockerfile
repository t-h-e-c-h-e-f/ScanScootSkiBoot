FROM python:3.12-slim

WORKDIR /app

# Minimal runtime deps for FastAPI uploads + uvicorn.
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

COPY . /app

EXPOSE 16444

ENV HPDB_PATH=/data/hpdb_default.sqlite \
    UPLOAD_DIR=/data/uploads \
    API_KEYS_PATH=/data/keys.ini \
    ZIP_CSV_PATH=/data/uszips.csv

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "16444"]

