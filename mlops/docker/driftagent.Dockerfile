FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY mlops/docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/agentic_ai/drift_retrain.py /app/drift_retrain.py

# Healthcheck: check if drift_retrain process is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD pgrep -f drift_retrain.py || exit 1

CMD ["python", "/app/drift_retrain.py"]
