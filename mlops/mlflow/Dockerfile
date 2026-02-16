FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.12.2 \
    boto3>=1.34 \
    psycopg2-binary>=2.9

ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000

EXPOSE 5000

CMD ["sh", "-c", "mlflow server \
  --host ${MLFLOW_HOST} \
  --port ${MLFLOW_PORT} \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"]
