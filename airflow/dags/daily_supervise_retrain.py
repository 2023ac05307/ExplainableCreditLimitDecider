from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.models import Variable

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=4),
}

with DAG(
    dag_id="daily_train_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["train", "pipeline"],
) as dag:

    s3_bucket = Variable.get("TRAIN_DATA_BUCKET", default_var="my-bucket")
    s3_prefix = Variable.get("TRAIN_DATA_PREFIX", default_var="rl_dataset/curated/")
    mlflow_uri = Variable.get("MLFLOW_TRACKING_URI", default_var="http://localhost:5000")

    run_training = DockerOperator(
        task_id="run_train_pipeline",
        image="myorg/credit-limit-trainer:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mount_tmp_dir=False,
        environment={
            "S3_BUCKET": s3_bucket,
            "S3_PREFIX": s3_prefix,
            "MLFLOW_TRACKING_URI": mlflow_uri,
            "RUN_DATE": "{{ ds }}",
        },
        # IMPORTANT: adjust to your actual module path:
        # If your file is src/pipelines/train_pipeline.py -> use python -m src.pipelines.train_pipeline
        command="python -m src.pipelines.train_pipeline --run-date {{ ds }}",
    )
