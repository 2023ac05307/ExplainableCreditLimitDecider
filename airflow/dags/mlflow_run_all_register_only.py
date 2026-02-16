from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="mlflow_run_all_register_only_bash",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow", "registry", "demo"],
) as dag:

    run_all_only_register = BashOperator(
        task_id="run_all_only_register",
        bash_command=(
            "set -euo pipefail; "
            "cd /opt/airflow/app; "
            "python -m mlops.mlflow.run_all "
            "--config configs/paths.yaml "
            "--experiment CreditLimit-RL-OptionA "
            "--run-name airflow_only_register "
            "--tracking-uri http://mlflow:5000 "
        ),
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "CKPT_ROOT": "/host_checkpoints",
            "PYTHONPATH": "/opt/airflow/app",
        },
    )
