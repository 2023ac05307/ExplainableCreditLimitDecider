from __future__ import annotations
from pathlib import Path
import json
import time
import argparse

import mlflow
from mlflow.tracking import MlflowClient


def register_artifact_only(model_name: str, artifact_local_path: str) -> str:
    """
    Logs a file as an artifact and registers that artifact path as a model version source.
    Works even when mlflow.register_model() / logged-models endpoints are unavailable.
    """
    artifact_local_path = str(artifact_local_path)
    fname = Path(artifact_local_path).name

    mlflow.log_artifact(artifact_local_path, artifact_path="checkpoints")

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active run. Did you start_run()?")

    run_id = run.info.run_id
    source = f"runs:/{run_id}/checkpoints/{fname}"

    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    mv = client.create_model_version(name=model_name, source=source, run_id=run_id)
    return str(mv.version)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    ap.add_argument("--experiment", default="smoke_tests")
    ap.add_argument("--model-name", default="cl_policy_gate")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # create a tiny dummy "checkpoint"
    tmp = Path("tmp")
    tmp.mkdir(exist_ok=True)
    ckpt = tmp / "dummy_checkpoint.txt"
    ckpt.write_text(f"dummy ckpt created at {time.time()}\n")

    with mlflow.start_run(run_name="mlflow_smoke_test") as run:
        mlflow.log_param("smoke", "true")
        mlflow.log_metrics({"acc": 0.91, "f1": 0.77})
        mlflow.log_dict({"note": "hello"}, "notes.json")

        version = register_artifact_only(args.model_name, str(ckpt))
        mlflow.set_tag("registered_version", version)

        print("✅ Run:", run.info.run_id)
        print("✅ Registered model:", args.model_name, "version:", version)
        print("Open:", f"{args.tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
