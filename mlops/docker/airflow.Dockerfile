ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.10

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

USER root

# OS deps as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 🔻 switch BEFORE pip installs
USER airflow

# Install your extra python deps as airflow user
RUN pip install --no-cache-dir \
  "numpy==1.26.4" \
  "fastapi>=0.110" \
  "uvicorn[standard]>=0.27" \
  "prometheus-client>=0.20" \
  "pydantic>=2.6" \
  "jinja2>=3.1" \
  "pandas>=2.1" \
  "pyarrow>=14" \
  "mlflow>=2.10" \
  "boto3>=1.34" \
  "python-dotenv>=1.0" \
  "pyyaml" \
  "apache-airflow-providers-docker" \
  "docker"

# CPU-only torch (also as airflow user)
RUN pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.1.2

# Airflow constraints install (keep it last)
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt --constraint "${CONSTRAINT_URL}"
