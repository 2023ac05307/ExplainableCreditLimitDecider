import os
from pathlib import Path
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # inside mlflow container, localhost is correct

CKPT_ROOT = Path("/host_checkpoints")

files = [
    CKPT_ROOT / "classification" / "gate_awac.pt",
    CKPT_ROOT / "classification" / "dir_awac.pt",
    CKPT_ROOT / "regression" / "mag_cli_beta.pt",
    CKPT_ROOT / "regression" / "mag_cld_beta.pt",
]

for f in files:
    if not f.exists():
        raise SystemExit(f"Missing: {f}")
    print("FOUND", f, "size=", f.stat().st_size)

mlflow.set_experiment("proof_artifact_run")

with mlflow.start_run(run_name="proof_ckpt_artifacts") as run:
    run_id = run.info.run_id
    print("RUN", run_id)

    # upload into artifacts/bundle/... (same structure serving expects)
    mlflow.log_artifacts(str(CKPT_ROOT / "classification"), artifact_path="bundle/classification")
    mlflow.log_artifacts(str(CKPT_ROOT / "regression"), artifact_path="bundle/regression")

    # also upload one file explicitly as a second proof
    mlflow.log_artifact(str(files[0]), artifact_path="bundle/single_file_proof")

print("DONE")
