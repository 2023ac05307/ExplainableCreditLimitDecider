# mlops/mlflow/run_all.py
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


from pathlib import Path
import mlflow

from mlops.mlflow.tracking import setup_mlflow
from mlflow.tracking import MlflowClient
from mlops.mlflow.log_artifacts import save_confusion_png_binary
from mlops.mlflow.registry import set_alias, set_version_tags
from mlops.mlflow.promote import promote_if_better
from src.pipelines.eval_pipeline import EvalPipelineConfig, run_eval
from src.pipelines.explain_pipeline import ExplainPipelineConfig, run_explain_pipeline

# Pipeline (single source of truth for training)
from src.pipelines.train_pipeline import _read_yaml, build_config, run_training


# -----------------------------------------------------------------------------
# Repo root
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]  # .../mlops/mlflow/run_all.py -> repo root


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    """
    Always run from repo root by default; ensures module imports work.
    """
    use_cwd = str(cwd or REPO)
    use_env = os.environ.copy()
    if env:
        use_env.update(env)

    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=use_cwd, env=use_env)


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _register_ckpt_as_artifact_only(model_name: str, ckpt_path: str) -> str:
    ckpt_path = str(ckpt_path)

    # If ckpt_path is checkpoints/classification/gate_awac.pt
    # then subdir = classification and ckpt_name = gate_awac.pt
    p = Path(ckpt_path)
    ckpt_name = p.name
    subdir = p.parent.name  # classification or regression

    artifact_dir = f"checkpoints/{subdir}"
    mlflow.log_artifact(ckpt_path, artifact_path=artifact_dir)

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run.")
    run_id = run.info.run_id

    source = f"runs:/{run_id}/{artifact_dir}/{ckpt_name}"

    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    mv = client.create_model_version(name=model_name, source=source, run_id=run_id)
    return str(mv.version)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline-driven MLflow runner: train -> eval -> register -> promote")
    p.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/paths.yaml)")

    p.add_argument("--experiment", default="CreditLimit-RL-OptionA", help="MLflow experiment name")
    p.add_argument("--run-name", default="train_eval_register_optionA", help="MLflow run name")
    p.add_argument("--tracking-uri", default=None, help="Override MLFLOW_TRACKING_URI (e.g. http://mlflow:5000)")

    # Toggles (same spirit as train_pipeline)
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--no-dir", action="store_true")
    p.add_argument("--no-mag-cli", action="store_true")
    p.add_argument("--no-mag-cld", action="store_true")
    p.add_argument("--only-register", action="store_true", help="Skip train/eval; only register existing checkpoints")
    p.add_argument("--gate-ckpt", default=None, help="Path to existing gate checkpoint")
    p.add_argument("--dir-ckpt", default=None, help="Path to existing dir checkpoint")
    p.add_argument("--mag-cli-ckpt", default=None, help="Path to existing mag CLI checkpoint")
    p.add_argument("--mag-cld-ckpt", default=None, help="Path to existing mag CLD checkpoint")


    # Optional overrides
    p.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    p.add_argument("--seed", type=int, default=None, help="Override seed")

    # Gate threshold for evaluation
    p.add_argument("--gate-thr", default="0.75", help="Gate eval threshold")
    p.add_argument("--test-3cls", default=None, help="3-class trajectories parquet for evaluation (action_id 0/1/2)")
    p.add_argument("--features-parquet", default=None, help="Features parquet with cust_id + s_ columns for explain pipeline")
    p.add_argument("--explain-stage", default="final", choices=["gate", "dir", "final"])
    p.add_argument("--explain-topk", type=int, default=8)
    p.add_argument("--explain-limit", type=int, default=200, help="Explain only first N rows (speed).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    # Setup MLflow experiment
    setup_mlflow(experiment=args.experiment)

    # Build training config from YAML (single source of truth)
    cfg = _read_yaml(args.config)
    conf = build_config(cfg)

    if args.device is not None:
        conf.device = args.device
    if args.seed is not None:
        conf.seed = args.seed

    conf.do_gate = not args.no_gate
    conf.do_dir = not args.no_dir
    conf.do_mag_cli = not args.no_mag_cli
    conf.do_mag_cld = not args.no_mag_cld

    # Ensure output dirs
    reports_dir = REPO / "reports" / "eval"
    _ensure_dir(reports_dir)
    _ensure_dir(REPO / "checkpoints")

    # Log important config bits


    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print("RUN", run_id)

        mlflow.log_param("config_path", args.config)
        mlflow.log_param("seed", conf.seed)
        mlflow.log_param("device", conf.device)
        mlflow.log_param("do_gate", int(conf.do_gate))
        mlflow.log_param("do_dir", int(conf.do_dir))
        mlflow.log_param("do_mag_cli", int(conf.do_mag_cli))
        mlflow.log_param("do_mag_cld", int(conf.do_mag_cld))
        mlflow.log_param("gate_eval_thr", args.gate_thr)

        # -----------------------------
        # Registry-only mode
        # -----------------------------
        if args.only_register:
            # Resolve checkpoint paths: CLI overrides > config outputs > defaults
            CKPT_ROOT = Path(os.getenv("CKPT_ROOT", "checkpoints"))
            gate_ckpt = Path(args.gate_ckpt) if args.gate_ckpt else (CKPT_ROOT / "classification/gate_awac.pt")
            dir_ckpt  = Path(args.dir_ckpt)  if args.dir_ckpt  else (CKPT_ROOT / "classification/dir_awac.pt")
            mag_cli_ckpt = Path(args.mag_cli_ckpt) if args.mag_cli_ckpt else (CKPT_ROOT / "regression/mag_cli_beta.pt")
            mag_cld_ckpt = Path(args.mag_cld_ckpt) if args.mag_cld_ckpt else (CKPT_ROOT / "regression/mag_cld_beta.pt")

            # Verify files exist (fail fast)
            for name, pth in [
                ("gate_ckpt", gate_ckpt),
                ("dir_ckpt", dir_ckpt),
                ("mag_cli_ckpt", mag_cli_ckpt),
                ("mag_cld_ckpt", mag_cld_ckpt),
            ]:
                if pth and not Path(pth).exists():
                    raise FileNotFoundError(f"{name} not found: {pth}")

            # Register
            if conf.do_gate:
                gate_version = _register_ckpt_as_artifact_only("cl_policy_gate", gate_ckpt)
                set_alias("cl_policy_gate", gate_version, "challenger")
                mlflow.set_tag("gate_version", gate_version)

            if conf.do_dir:
                dir_version = _register_ckpt_as_artifact_only("cl_policy_dir", dir_ckpt)
                set_alias("cl_policy_dir", dir_version, "challenger")
                mlflow.set_tag("dir_version", dir_version)

            if conf.do_mag_cli:
                cli_version = _register_ckpt_as_artifact_only("cl_mag_cli", mag_cli_ckpt)
                set_alias("cl_mag_cli", cli_version, "challenger")
                mlflow.set_tag("mag_cli_version", cli_version)

            if conf.do_mag_cld:
                cld_version = _register_ckpt_as_artifact_only("cl_mag_cld", mag_cld_ckpt)
                set_alias("cl_mag_cld", cld_version, "challenger")
                mlflow.set_tag("mag_cld_version", cld_version)

            print("Registry-only completed.")
            return

        # ---------------------------------------------------------------------
        # 1) TRAIN (pipeline-driven)
        # ---------------------------------------------------------------------
        outputs = run_training(conf)
        # Save outputs snapshot for reproducibility
        out_json = reports_dir / "train_outputs.json"
        out_json.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out_json), artifact_path="train")

        print("Training pipeline outputs:")
        for k, v in outputs.items():
            print("  -", k, ":", v)

        # Helper paths
        gate_ckpt = outputs.get("gate_ckpt")
        dir_ckpt = outputs.get("dir_ckpt")
        mag_cli_ckpt = outputs.get("mag_cli_ckpt")
        mag_cld_ckpt = outputs.get("mag_cld_ckpt")

        # ---------------------------------------------------------------------
        # 2) EVAL (end-to-end using serving bundle) + log artifacts/metrics
        # ---------------------------------------------------------------------
        # Choose the 3-class test parquet:
        test_3cls = args.test_3cls or conf.traj_val_3cls  # default fallback

        ev = run_eval(EvalPipelineConfig(
            test_parquet_3cls=test_3cls,
            out_dir=str(reports_dir),
            gate_ckpt=str(gate_ckpt) if gate_ckpt else "checkpoints/classification/gate_awac.pt",
            dir_ckpt=str(dir_ckpt) if dir_ckpt else "checkpoints/classification/dir_awac.pt",
            mag_cli_ckpt=str(mag_cli_ckpt) if mag_cli_ckpt else "checkpoints/regression/mag_cli_beta.pt",
            mag_cld_ckpt=str(mag_cld_ckpt) if mag_cld_ckpt else "checkpoints/regression/mag_cld_beta.pt",
            gate_thr=float(args.gate_thr),
            # optionally expose a dir_thr arg too; otherwise keep default in EvalPipelineConfig
        ))

        # Log artifacts produced by eval pipeline
        mlflow.log_artifact(ev["metrics_json"], artifact_path="eval/e2e")
        mlflow.log_artifact(ev["confusion_csv"], artifact_path="eval/e2e")
        mlflow.log_artifact(ev["preds_parquet"], artifact_path="eval/e2e")

        # Log headline metrics into MLflow
        m = ev["metrics"]
        gm = m["gate"]
        mlflow.log_metrics({
            "gate_precision": float(gm["precision"]),
            "gate_recall": float(gm["recall"]),
            "gate_f1": float(gm["f1"]),
            "dir_acc_on_true_nonhold": float(m["dir"]["acc_on_true_nonhold"]),
        })

        # If magnitude metrics exist, log them too
        mag = m.get("magnitude", {})
        for key in ["CLI", "CLD"]:
            if key in mag:
                mlflow.log_metrics({
                    f"mag_{key.lower()}_mae": float(mag[key]["mae"]),
                    f"mag_{key.lower()}_p90_ae": float(mag[key]["p90"]),
                    f"mag_{key.lower()}_p95_ae": float(mag[key]["p95"]),
                })

        # ---------------------------------------------------------------------
        # 3) REGISTER + CHAMPION/CHALLENGER PROMOTION (Model Registry)
        # ---------------------------------------------------------------------
        # NOTE: These register the *checkpoint artifacts* (not MLflow flavor models).
        # Works well for a champion/challenger pattern + traceability.

        # Gate
        if conf.do_gate and gate_ckpt:
            gate_version = _register_ckpt_as_artifact_only("cl_policy_gate", str(gate_ckpt))
            set_alias("cl_policy_gate", gate_version, "challenger")
            set_version_tags(
                "cl_policy_gate",
                gate_version,
                {
                    "task": "gate",
                    "gate_thr_used": str(args.gate_thr),
                    "precision": str(gm["precision"]),
                    "recall": str(gm["recall"]),
                    "f1": str(gm["f1"]),
                    "seed": str(conf.seed),
                    "device": str(conf.device),
                    "ckpt_path": str(gate_ckpt),
                },
            )
            # promote_if_better returns (bool_promoted, message)
            _, msg = promote_if_better("cl_policy_gate", gate_version, kind="clf")
            mlflow.set_tag("gate_promotion", msg)

        # Dir
        if conf.do_dir and dir_ckpt:
            dir_version = _register_ckpt_as_artifact_only("cl_policy_dir", str(dir_ckpt))
            set_alias("cl_policy_dir", dir_version, "challenger")
            set_version_tags(
                "cl_policy_dir",
                dir_version,
                {
                    "task": "dir",
                    "acc_on_true_nonhold": str(m["dir"]["acc_on_true_nonhold"]),
                    "seed": str(conf.seed),
                    "device": str(conf.device),
                    "ckpt_path": str(dir_ckpt),
                },
            )
            _, msg = promote_if_better("cl_policy_dir", dir_version, kind="clf")
            mlflow.set_tag("dir_promotion", msg)

        # Magnitude CLI
        if conf.do_mag_cli and mag_cli_ckpt:
            cli_metrics = m.get("magnitude", {}).get("CLI", {})
            cli_version = _register_ckpt_as_artifact_only("cl_mag_cli", str(mag_cli_ckpt))
            set_alias("cl_mag_cli", cli_version, "challenger")
            set_version_tags(
                "cl_mag_cli",
                cli_version,
                {
                    "task": "mag_cli",
                    "mae": str(cli_metrics.get("mae", "")),
                    "p90": str(cli_metrics.get("p90", "")),
                    "p95": str(cli_metrics.get("p95", "")),
                    "seed": str(conf.seed),
                    "device": str(conf.device),
                    "ckpt_path": str(mag_cli_ckpt),
                },
            )
            _, msg = promote_if_better("cl_mag_cli", cli_version, kind="reg")
            mlflow.set_tag("mag_cli_promotion", msg)

        # Magnitude CLD
        if conf.do_mag_cld and mag_cld_ckpt:
            cld_metrics = m.get("magnitude", {}).get("CLD", {})
            cld_version = _register_ckpt_as_artifact_only("cl_mag_cld", str(mag_cld_ckpt))
            set_alias("cl_mag_cld", cld_version, "challenger")
            set_version_tags(
                "cl_mag_cld",
                cld_version,
                {
                    "task": "mag_cld",
                    "mae": str(cld_metrics.get("mae", "")),
                    "p90": str(cld_metrics.get("p90", "")),
                    "p95": str(cld_metrics.get("p95", "")),
                    "seed": str(conf.seed),
                    "device": str(conf.device),
                    "ckpt_path": str(mag_cld_ckpt),
                },
            )
            _, msg = promote_if_better("cl_mag_cld", cld_version, kind="reg")
            mlflow.set_tag("mag_cld_promotion", msg)

        print("✅ Registered & promoted (champion/challenger) via MLflow Model Registry.")



        print("✅ Completed pipeline-driven train/eval/register/promote.")


if __name__ == "__main__":
    main()
