# ARG AIRFLOW_VERSION=2.8.4
# ARG PYTHON_VERSION=3.10

# FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

# USER root
# RUN apt-get update && apt-get install -y --no-install-recommends gcc \
#  && apt-get clean && rm -rf /var/lib/apt/lists/*

# ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# COPY airflow/requirements.txt /requirements.txt

# # Install everything (except torch) under Airflow constraints
# RUN pip install --no-cache-dir -r /requirements.txt --constraint "${CONSTRAINT_URL}"

# # Install CPU-only torch separately (doesn't need to obey Airflow constraints)
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.2


FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY mlops/docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.2

COPY src /app/src
COPY configs /app/configs

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
