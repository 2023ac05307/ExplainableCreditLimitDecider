# src/mlops/mlflow/tracking.py
from __future__ import annotations
import os
import mlflow

def setup_mlflow(experiment: str = "CreditLimit-RL"):
    # If user sets MLFLOW_TRACKING_URI env, this respects it.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return tracking_uri
