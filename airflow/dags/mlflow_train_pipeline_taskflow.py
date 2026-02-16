from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml
import mlflow
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from mlflow.tracking import MlflowClient
from airflow.operators.python import get_current_context
from airflow.operators.bash import BashOperator
import sys
sys.path.insert(0, "/opt/airflow/app")

IMPORT_ERROR = None
try:
    from mlops.mlflow.tracking import setup_mlflow
    from mlops.mlflow.registry import set_alias, set_version_tags
    from mlops.mlflow.promote import promote_if_better
    from src.pipelines.eval_pipeline import EvalPipelineConfig, run_eval
    from src.pipelines.train_pipeline import _read_yaml, build_config, run_training
except Exception as e:
    IMPORT_ERROR = repr(e)


# Reuse the helper from run_all (copy-pasted pattern)
# IMPORTANT: this must run inside an active mlflow run
def _register_ckpt_as_artifact_only(model_name: str, ckpt_path: str) -> str:
    ckpt_path = str(ckpt_path)
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


def _alias_exists(model_name: str, alias: str) -> bool:
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(model_name, alias)
        return True
    except Exception:
        return False


with DAG(
    dag_id="mlflow_train_taskflow_optionA",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow", "train", "taskflow", "optionA", "champion-challenger"],
) as dag:
    
    if IMPORT_ERROR:
        @task
        def show_import_error():
            raise Exception(f"DAG import failed: {IMPORT_ERROR}")
        show_import_error()
    else:

        @task
        def make_runtime_config(
            run_id: str,
            base_work_root: str = "/opt/airflow/app/work/data_engineering",
            template_config: str = "/opt/airflow/app/configs/paths.yaml",  # <-- use absolute
            view_root_name: str = "training_view",
            make_view_unique_per_dag_run: bool = True,  # avoids collisions on reruns
        ) -> Dict[str, Any]:
            base = Path(base_work_root) / run_id
            curated = base / "curated"
            staging = base / "staging"

            # inputs produced by data pipeline
            gate_dir_src = curated / "gate_dir"
            aug_src = curated / "trajectories_train_aug"
            splits_src = staging / "splits"

            # sanity checks
            for p in [gate_dir_src, aug_src, splits_src]:
                if not p.exists():
                    raise AirflowFailException(f"Missing required folder: {p}")

            # make view path (optionally unique per current DAG run)
            ctx = get_current_context()
            dag_run_id = ctx["dag_run"].run_id if (ctx.get("dag_run") and make_view_unique_per_dag_run) else None

            view = base / view_root_name / (dag_run_id if dag_run_id else "")
            view.mkdir(parents=True, exist_ok=True)

            view_gate = view / "gate_dir"          # should be a symlink (do NOT mkdir this)
            view_splits = view / "splits"          # should be a real dir (we place symlinks inside)
            view_splits.mkdir(parents=True, exist_ok=True)

            # robust link helper: replaces existing dst whether file/dir/symlink
            def link_any(src: Path, dst: Path, is_dir: bool = True):
                dst.parent.mkdir(parents=True, exist_ok=True)

                # remove existing dst safely
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                elif dst.exists():  # directory
                    shutil.rmtree(dst)

                dst.symlink_to(src, target_is_directory=is_dir)

            # link curated folders into view
            link_any(gate_dir_src, view_gate, is_dir=True)
            link_any(aug_src, view_splits / "trajectories_aug", is_dir=True)  # matches train_pipeline expectation

            # link val/test parquet files into view_splits
            for fn in ["trajectories_val.parquet", "trajectories_test.parquet"]:
                srcf = splits_src / fn
                if not srcf.exists():
                    raise AirflowFailException(f"Missing required file: {srcf}")

                dstf = view_splits / fn
                if dstf.is_symlink() or dstf.is_file():
                    dstf.unlink()
                elif dstf.exists():
                    shutil.rmtree(dstf)

                dstf.symlink_to(srcf)

            # build runtime yaml that overrides gate_dir_dir and splits_dir
            cfg = _read_yaml(str(template_config))
            cfg["gate_dir_dir"] = str(view_gate)
            cfg["splits_dir"] = str(view_splits)

            # checkpoints root (keep existing if present, else set default)
            cfg.setdefault("checkpoints", {})
            cfg["checkpoints"]["root"] = "/opt/airflow/app/checkpoints"

            runtime_yaml = view / "paths_runtime.yaml"  # store runtime yaml inside view (unique per run if enabled)
            runtime_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            return {
                "config_path": str(runtime_yaml),
                "run_id": run_id,
                "training_view": str(view),
                "dag_run_id": dag_run_id,
            }
        @task
        def load_training_conf(config_path: str = "configs/paths.yaml") -> Dict[str, Any]:
            """
            Build the same conf object as run_all.py:
            cfg = _read_yaml(args.config)
            conf = build_config(cfg)
            """
            cfg = _read_yaml(config_path)
            conf = build_config(cfg)

            # allow env overrides (like run_all does)
            if os.getenv("TRAIN_DEVICE"):
                conf.device = os.getenv("TRAIN_DEVICE")
            if os.getenv("TRAIN_SEED"):
                conf.seed = int(os.getenv("TRAIN_SEED"))

            # Serialize only what we need downstream
            return {
                "config_path": config_path,
                "seed": int(conf.seed),
                "device": str(conf.device),
            }

        @task
        def parse_json_line(s: str) -> Dict[str, Any]:
            import json
            return json.loads(s)        
        @task
        def eval_e2e(
            conf_meta: Dict[str, Any],
            gate_out: Dict[str, str],
            dir_out: Dict[str, str],
            mag_out: Dict[str, str],
            gate_thr: float = 0.75,
        ) -> Dict[str, Any]:
            """
            Same style as run_all.py: run_eval(EvalPipelineConfig(...)) and return metrics + file paths【:contentReference[oaicite:7]{index=7}】
            """
            cfg = _read_yaml(conf_meta["config_path"])
            ckpt_root = cfg["checkpoints"]["root"]
            conf = build_config(cfg)

            reports_dir = Path("/opt/airflow/app/reports/eval")
            reports_dir.mkdir(parents=True, exist_ok=True)

            test_3cls = conf.traj_val_3cls  # same default fallback used in run_all【:contentReference[oaicite:8]{index=8}】

            ev = run_eval(
                EvalPipelineConfig(
                    test_parquet_3cls=test_3cls,
                    out_dir=str(reports_dir),
                    gate_ckpt=f"{ckpt_root}/classification/gate_awac.pt",
                    dir_ckpt=f"{ckpt_root}/classification/dir_awac.pt",
                    mag_cli_ckpt=f"{ckpt_root}/regression/mag_cli_beta.pt",
                    mag_cld_ckpt=f"{ckpt_root}/regression/mag_cld_beta.pt",
                    gate_thr=float(gate_thr),
                    baseline_stats_json="/opt/airflow/app/reports/baseline/baseline_stats.json"
                )
            )

            # ev expected keys are used by run_all: metrics_json, confusion_csv, preds_parquet, metrics【:contentReference[oaicite:9]{index=9}】
            return ev

        @task
        def get_source_run_id(**context) -> str:
            conf = (context["dag_run"].conf or {})
            rid = conf.get("source_run_id")
            if not rid:
                raise AirflowFailException(
                    "Missing source_run_id. Trigger with JSON like "
                    "{'source_run_id':'manual__2026-02-05T15:22:54.281486+00:00'}"
                )
            return rid

        @task
        def mlflow_register_and_promote(
            conf_meta: Dict[str, Any],
            gate_out: Dict[str, str],
            dir_out: Dict[str, str],
            mag_out: Dict[str, str],
            ev: Dict[str, Any],
            experiment: str = "CreditLimit-RL-OptionA",
            run_name: str = "airflow_train_taskflow",
            tracking_uri: str = "http://mlflow:5000",
            gate_thr_used: float = 0.75,
        ) -> Dict[str, Any]:
            """
            One MLflow run:
            - log eval artifacts + headline metrics (same as run_all)【】
            - register ckpt artifacts as model versions (gate/dir/mag_cli/mag_cld)【:contentReference[oaicite:11]{index=11}】
            - set challenger alias
            - promote_if_better
            - first run: if champion alias absent, set champion = challenger
            - remove old champion tag when champion changes
            """
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            setup_mlflow(experiment=experiment)  # same setup as run_all【:contentReference[oaicite:12]{index=12}】

            m = ev["metrics"]
            gm = m["gate"]

            # Optional: include more production metrics if your eval pipeline exports them.
            # For now we log the ones run_all already logs: gate_precision/recall/f1, dir_acc_on_true_nonhold, mag mae/p90/p95【】

            client = MlflowClient(tracking_uri=tracking_uri)

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("config_path", conf_meta["config_path"])
                mlflow.log_param("seed", conf_meta["seed"])
                mlflow.log_param("device", conf_meta["device"])
                mlflow.log_param("gate_eval_thr", gate_thr_used)

                # Log eval artifacts
                mlflow.log_artifact(ev["metrics_json"], artifact_path="eval/e2e")
                mlflow.log_artifact(ev["confusion_csv"], artifact_path="eval/e2e")
                mlflow.log_artifact(ev["preds_parquet"], artifact_path="eval/e2e")
                mlflow.log_artifact(ev["drift_json"], artifact_path="eval/e2e")
                # Headline metrics (same pattern as run_all)
                mlflow.log_metrics(
                    {
                        "gate_precision": float(gm["precision"]),
                        "gate_recall": float(gm["recall"]),
                        "gate_f1": float(gm["f1"]),
                        "gate_balanced_acc": float(gm["balanced_acc"]),   
                        "dir_acc_on_true_nonhold": float(m["dir"]["acc_on_true_nonhold"]),
                        "dir_balanced_acc": float(m["dir"]["balanced_acc"]),  
                    }
                )
                gate_extra = {}
                for k in ["logloss", "brier", "ece_10bins", "roc_auc", "pr_auc"]:
                    if k in gm and gm[k] == gm[k]:
                        gate_extra[f"gate_{k}"] = float(gm[k])
                if gate_extra:
                    mlflow.log_metrics(gate_extra)

                dir_extra = {}
                dm = m["dir"]
                for k in ["logloss", "brier", "ece_10bins", "roc_auc", "pr_auc"]:
                    if k in dm and dm[k] == dm[k]:
                        dir_extra[f"dir_{k}"] = float(dm[k])
                if dir_extra:
                    mlflow.log_metrics(dir_extra)                
                if "drift" in ev:
                    js = ev["drift"].get("action_js_divergence")
                    if js is not None and js == js:
                        mlflow.log_metric("action_js_divergence", float(js))

                def _pick(d: dict, names: list[str]):
                    """Return (name,value) for the first present metric key (case-insensitive), else (None,None)."""
                    if not isinstance(d, dict):
                        return (None, None)
                    # exact
                    for n in names:
                        if n in d and d[n] is not None:
                            return (n, d[n])
                    # case-insensitive
                    lower = {str(k).lower(): v for k, v in d.items()}
                    for n in names:
                        if n.lower() in lower and lower[n.lower()] is not None:
                            return (n, lower[n.lower()])
                    return (None, None)

                mag = m.get("magnitude", {}) or {}
                for key in ["CLI", "CLD"]:
                    if key in mag:
                        mm = mag[key] or {}
                        to_log = {}

                        # These keys exist in eval_pipeline.py
                        if "mae_pp" in mm and mm["mae_pp"] == mm["mae_pp"]:  # not NaN
                            to_log[f"mag_{key.lower()}_mae_pp"] = float(mm["mae_pp"])
                        if "rmse_pp" in mm and mm["rmse_pp"] == mm["rmse_pp"]:
                            to_log[f"mag_{key.lower()}_rmse_pp"] = float(mm["rmse_pp"])
                        if "p90" in mm and mm["p90"] == mm["p90"]:
                            to_log[f"mag_{key.lower()}_p90_ae"] = float(mm["p90"])
                        if "p95" in mm and mm["p95"] == mm["p95"]:
                            to_log[f"mag_{key.lower()}_p95_ae"] = float(mm["p95"])

                        if to_log:
                            mlflow.log_metrics(to_log)

                # --------------------------
                # REGISTER + PROMOTE
                # --------------------------
                # Gate
                gate_version = _register_ckpt_as_artifact_only("cl_policy_gate", gate_out["gate_ckpt"])
                set_alias("cl_policy_gate", gate_version, "challenger")
                set_version_tags(
                    "cl_policy_gate",
                    gate_version,
                    {
                        "task": "gate",
                        "gate_thr_used": str(gate_thr_used),
                        "precision": str(gm["precision"]),
                        "recall": str(gm["recall"]),
                        "f1": str(gm["f1"]),
                        "balanced_acc": str(gm["balanced_acc"]),
                        "seed": str(conf_meta["seed"]),
                        "device": str(conf_meta["device"]),
                        "ckpt_path": gate_out["gate_ckpt"],
                    },
                )
                promoted_gate, msg_gate = promote_if_better("cl_policy_gate", gate_version, kind="clf")
                mlflow.set_tag("gate_promotion", msg_gate)

                # Dir
                dir_version = _register_ckpt_as_artifact_only("cl_policy_dir", dir_out["dir_ckpt"])
                set_alias("cl_policy_dir", dir_version, "challenger")

                set_version_tags(
                    "cl_policy_dir",
                    dir_version,
                    {
                        "task": "dir",
                        "acc_on_true_nonhold": str(m["dir"]["acc_on_true_nonhold"]),
                        "f1": str(m["dir"]["f1"]),                    
                        "balanced_acc": str(m["dir"]["balanced_acc"]),
                        "n_true_nonhold": str(m["dir"].get("n_true_nonhold", "")),
                        "seed": str(conf_meta["seed"]),
                        "device": str(conf_meta["device"]),
                        "ckpt_path": dir_out["dir_ckpt"],
                    },
                )
                promoted_dir, msg_dir = promote_if_better("cl_policy_dir", dir_version, kind="clf")
                mlflow.set_tag("dir_promotion", msg_dir)

                # Mag CLI
                cli_metrics = m.get("magnitude", {}).get("CLI", {}) or {}
                _, cli_mae = _pick(cli_metrics, ["mae", "mean_absolute_error", "mean_abs_error"])
                _, cli_p90 = _pick(cli_metrics, ["p90", "p90_ae", "p90_abs_error", "ae_p90"])
                _, cli_p95 = _pick(cli_metrics, ["p95", "p95_ae", "p95_abs_error", "ae_p95"])
                cli_version = _register_ckpt_as_artifact_only("cl_mag_cli", mag_out["mag_cli_ckpt"])
                set_alias("cl_mag_cli", cli_version, "challenger")
                set_version_tags(
                    "cl_mag_cli",
                    cli_version,
                    {
                        "task": "mag_cli",
                        "mae_pp": str(cli_metrics.get("mae_pp", "")),     
                        "rmse_pp": str(cli_metrics.get("rmse_pp", "")),  
                        "p90": str(cli_metrics.get("p90", "")),
                        "p95": str(cli_metrics.get("p95", "")),
                        "seed": str(conf_meta["seed"]),
                        "device": str(conf_meta["device"]),
                        "ckpt_path": mag_out["mag_cli_ckpt"],
                    },
                )                
                promoted_cli, msg_cli = promote_if_better("cl_mag_cli", cli_version, kind="reg")
                mlflow.set_tag("mag_cli_promotion", msg_cli)

                # Mag CLD
                cld_metrics = m.get("magnitude", {}).get("CLD", {})
                _, cld_mae = _pick(cld_metrics, ["mae", "mean_absolute_error", "mean_abs_error"])
                _, cld_p90 = _pick(cld_metrics, ["p90", "p90_ae", "p90_abs_error", "ae_p90"])
                _, cld_p95 = _pick(cld_metrics, ["p95", "p95_ae", "p95_abs_error", "ae_p95"])                
                cld_version = _register_ckpt_as_artifact_only("cl_mag_cld", mag_out["mag_cld_ckpt"])
                set_alias("cl_mag_cld", cld_version, "challenger")
                set_version_tags(
                    "cl_mag_cld",
                    cld_version,
                    {
                        "task": "mag_cld",
                        "mae_pp": str(cld_metrics.get("mae_pp", "")),     
                        "rmse_pp": str(cld_metrics.get("rmse_pp", "")),   
                        "p90": str(cld_metrics.get("p90", "")),
                        "p95": str(cld_metrics.get("p95", "")),
                        "seed": str(conf_meta["seed"]),
                        "device": str(conf_meta["device"]),
                        "ckpt_path": mag_out["mag_cld_ckpt"],
                    },
                )                
                promoted_cld, msg_cld = promote_if_better("cl_mag_cld", cld_version, kind="reg")
                mlflow.set_tag("mag_cld_promotion", msg_cld)

                # --------------------------
                # FIRST RUN RULE: challenger -> champion
                # --------------------------
                for model_name, version in [
                    ("cl_policy_gate", gate_version),
                    ("cl_policy_dir", dir_version),
                    ("cl_mag_cli", cli_version),
                    ("cl_mag_cld", cld_version),
                ]:
                    if not _alias_exists(model_name, "champion"):
                        set_alias(model_name, version, "champion")
                        # optional “role” tag for easy filtering
                        client.set_model_version_tag(model_name, version, "role", "champion")

                # --------------------------
                # CLEANUP RULE: if promotion happened, remove old champion role tag
                # --------------------------
                # promote_if_better() likely moves the alias; we also remove the tag from previous champion.
                # We do this by comparing previous champion before promotion would require reading earlier;
                # so we do best-effort cleanup: ensure ONLY alias-holder has role=champion.
                for model_name in ["cl_policy_gate", "cl_policy_dir", "cl_mag_cli", "cl_mag_cld"]:
                    try:
                        champ = client.get_model_version_by_alias(model_name, "champion")
                        # tag champion
                        client.set_model_version_tag(model_name, champ.version, "role", "champion")
                        # remove role tag from all other versions that have role=champion
                        for mv in client.search_model_versions(f"name='{model_name}'"):
                            if str(mv.version) != str(champ.version):
                                try:
                                    client.delete_model_version_tag(model_name, mv.version, "role")
                                except Exception:
                                    pass
                    except Exception:
                        pass

                return {
                    "run_id": run.info.run_id,
                    "versions": {
                        "cl_policy_gate": gate_version,
                        "cl_policy_dir": dir_version,
                        "cl_mag_cli": cli_version,
                        "cl_mag_cld": cld_version,
                    },
                    "promotions": {
                        "gate": msg_gate,
                        "dir": msg_dir,
                        "mag_cli": msg_cli,
                        "mag_cld": msg_cld,
                    },
                }

        # ---------------- DAG wiring (Gate -> Dir -> Magnitude -> Eval -> Register/Promote) ----------------
        source_run_id = get_source_run_id()
        rt = make_runtime_config(run_id=source_run_id)

        conf_meta = load_training_conf(config_path=rt["config_path"])
        train_gate = BashOperator(
            task_id="train_gate",
            do_xcom_push=True,   # IMPORTANT: captures last stdout line into XCom
            bash_command=r"""set -euo pipefail
export PYTHONPATH=/opt/airflow/app
CFG="{{ ti.xcom_pull(task_ids='make_runtime_config')['config_path'] }}"
cd /opt/airflow/app

python - <<'PY'
import json, os, sys
sys.path.insert(0, "/opt/airflow/app")

from src.pipelines.train_pipeline import _read_yaml, build_config, run_training

cfg_path = os.environ["CFG"]
cfg = _read_yaml(cfg_path)
conf = build_config(cfg)
conf.device = "cuda"

# Gate only
conf.do_gate = True
conf.do_dir = False
conf.do_mag_cli = False
conf.do_mag_cld = False

outputs = run_training(conf)

# Print JSON as LAST line for XCom
print(json.dumps({"gate_ckpt": outputs["gate_ckpt"]}))
PY
""",env={"CFG": "{{ ti.xcom_pull(task_ids='make_runtime_config')['config_path'] }}"})

        train_direction = BashOperator(
            task_id="train_direction",
            do_xcom_push=True,
            bash_command=r"""set -euo pipefail
export PYTHONPATH=/opt/airflow/app
CFG="{{ ti.xcom_pull(task_ids='make_runtime_config')['config_path'] }}"
cd /opt/airflow/app

python - <<'PY'
import json, os, sys
sys.path.insert(0, "/opt/airflow/app")

from src.pipelines.train_pipeline import _read_yaml, build_config, run_training

cfg_path = os.environ["CFG"]
cfg = _read_yaml(cfg_path)
conf = build_config(cfg)
conf.device = "cuda"

# Direction only
conf.do_gate = False
conf.do_dir = True
conf.do_mag_cli = False
conf.do_mag_cld = False

outputs = run_training(conf)
print(json.dumps({"dir_ckpt": outputs["dir_ckpt"]}))
PY
""",env={"CFG": "{{ ti.xcom_pull(task_ids='make_runtime_config')['config_path'] }}"})

        train_magnitude = BashOperator(
            task_id="train_magnitude",
            do_xcom_push=True,
            bash_command=r"""set -euo pipefail
export PYTHONPATH=/opt/airflow/app
CFG="{{ ti.xcom_pull(task_ids='make_runtime_config')['config_path'] }}"
cd /opt/airflow/app

python - <<'PY'
import json, os, sys
sys.path.insert(0, "/opt/airflow/app")

from src.pipelines.train_pipeline import _read_yaml, build_config, run_training

cfg_path = os.environ["CFG"]
cfg = _read_yaml(cfg_path)
conf = build_config(cfg)
conf.device = "cuda"

# Magnitude only
conf.do_gate = False
conf.do_dir = False
conf.do_mag_cli = True
conf.do_mag_cld = True

outputs = run_training(conf)
print(json.dumps({
    "mag_cli_ckpt": outputs["mag_cli_ckpt"],
    "mag_cld_ckpt": outputs["mag_cld_ckpt"],
}))
PY
""",env={"CFG": "{{ ti.xcom_pull(task_ids='make_runtime_config')['config_path'] }}"})
        # gate_out = train_gate(conf_meta)
        # dir_out = train_direction(conf_meta)
        # mag_out = train_magnitude(conf_meta)
        gate_out = parse_json_line(train_gate.output)
        dir_out = parse_json_line(train_direction.output)
        mag_out = parse_json_line(train_magnitude.output)        
        ev = eval_e2e(conf_meta, gate_out, dir_out, mag_out)
        rt >> train_gate >> train_direction >> train_magnitude
        mlflow_register_and_promote(conf_meta, gate_out, dir_out, mag_out, ev)
