FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow + common extras
RUN pip install --no-cache-dir \
    mlflow==2.12.2 \
    boto3>=1.34 \
    psycopg2-binary>=2.9

# Defaults (can be overridden from docker-compose)
ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# # Non-root user (optional but recommended)
# RUN useradd -m -u 10001 appuser
# USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:5000/ || exit 1

CMD ["sh", "-c", "mlflow server \
  --host ${MLFLOW_HOST} \
  --port ${MLFLOW_PORT} \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --artifacts-destination ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"]
