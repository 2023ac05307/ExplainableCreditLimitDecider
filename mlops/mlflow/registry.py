# src/mlops/mlflow/registry.py
from __future__ import annotations
from typing import Optional, Dict
import mlflow
from mlflow.tracking import MlflowClient

def set_alias(model_name: str, version: str, alias: str):
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)

def get_alias_version(model_name: str, alias: str) -> Optional[str]:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return str(mv.version)
    except Exception:
        return None

def set_version_tags(model_name: str, version: str, tags: Dict[str, str]):
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(model_name, version, k, str(v))
