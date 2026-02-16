# airflow/Dockerfile
ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.10
FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

USER root

# (Optional) OS libs if your DAG needs them; keep minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# Install python deps using official constraints
ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.10
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt --constraint "${CONSTRAINT_URL}"
