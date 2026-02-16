from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class PipeCfg:
    # S3/MinIO
    aws_conn_id: str
    bucket: str
    raw_prefix: str
    staging_prefix: str
    curated_prefix: str

    # Local working dir inside airflow container
    work_root: str

    # Scripts (inside airflow container)
    build_monthly_snapshots_py: str
    feature_engineering_py: str
    build_trajectories_py: str
    splitting_py: str
    augmentation_py: str
    prepare_gate_dir_py: str

    # Options
    include_derived_features: bool
    compression: str
    seed: int


def _cfg() -> PipeCfg:
    def v(key: str, default: str) -> str:
        return Variable.get(key, default_var=os.getenv(key, default))

    return PipeCfg(
        aws_conn_id=v("MINIO_AWS_CONN_ID", "minio_s3"),
        bucket=v("ECLD_BUCKET", "explainablecreditlimitdecider"),
        raw_prefix=v("RAW_PREFIX", "raw"),
        staging_prefix=v("STAGING_PREFIX", "staging"),
        curated_prefix=v("CURATED_PREFIX", "curated"),

        work_root=v("ECLD_WORK_ROOT", "/opt/airflow/app/work"),

        # You must place these scripts into the airflow container (recommended under /opt/airflow/app/scripts)
        build_monthly_snapshots_py=v("BUILD_MONTHLY_SNAPSHOTS_PY", "/opt/airflow/app/src/data_engineering/build_monthly_snapshots.py"),
        feature_engineering_py=v("FEATURE_ENGINEERING_PY", "/opt/airflow/app/src/data_engineering/feature_engineering.py"),
        build_trajectories_py=v("BUILD_TRAJECTORIES_PY", "/opt/airflow/app/src/data_engineering/build_trajectories.py"),
        splitting_py=v("SPLITTING_PY", "/opt/airflow/app/src/data_engineering/splitting.py"),
        augmentation_py=v("AUGMENTATION_PY", "/opt/airflow/app/src/data_engineering/augment_trajectories.py"),
        prepare_gate_dir_py=v("PREPARE_GATE_DIR_PY", "/opt/airflow/app/src/data_engineering/prepare_gate_dir_datasets.py"),

        include_derived_features=v("INCLUDE_DERIVED_FEATURES", "true").lower() == "true",
        compression=v("PARQUET_COMPRESSION", "snappy"),
        seed=int(v("PIPELINE_SEED", "42")),
    )


# -----------------------------
# Utilities
# -----------------------------
def _run(cmd: list[str], cwd: str | None = None):
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Always print output into Airflow logs
    if p.stdout:
        print(p.stdout)

    if p.returncode != 0:
        raise AirflowFailException(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _assert_exists_nonempty(path: Path, min_bytes: int = 128) -> None:
    if not path.exists():
        raise AirflowFailException(f"Missing expected output: {path}")
    if path.is_file() and path.stat().st_size < min_bytes:
        raise AirflowFailException(f"Output too small/empty: {path} ({path.stat().st_size} bytes)")


def _list_keys(s3: S3Hook, bucket: str, prefix: str) -> List[str]:
    keys = s3.list_keys(bucket_name=bucket, prefix=prefix)
    return keys or []



def _download_prefix_to_local(*, s3, bucket: str, prefix: str, local_dir: Path) -> list[str]:
    """
    Download all objects under s3://bucket/<prefix>/... into:
      <local_dir>/<relative key path>.

    IMPORTANT:
    Airflow S3Hook.download_file expects local_path to be a DIRECTORY (it creates a temp file inside it),
    so we pass dst.parent and then rename the returned temp file to the final dst filename.
    """
    # Ensure prefix ends with "/" so relative slicing is consistent
    prefix = prefix.strip("/")
    if prefix:
        prefix = prefix + "/"

    keys = s3.list_keys(bucket_name=bucket, prefix=prefix) or []
    downloaded = []

    for k in keys:
        if k.endswith("/"):
            continue

        # key relative to prefix, e.g. credit_events/credit_events_2006-01.csv
        rel = k[len(prefix):]
        dst = Path(local_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # ✅ download_file wants a DIRECTORY, not the full filename
        tmp_path = s3.download_file(
            key=k,
            bucket_name=bucket,
            local_path=str(dst.parent),
        )

        # Move temp file to the exact target filename
        Path(tmp_path).replace(dst)
        downloaded.append(str(dst))

    return downloaded


def _upload_file(s3: S3Hook, bucket: str, key: str, local_path: Path) -> None:
    s3.load_file(filename=str(local_path), key=key, bucket_name=bucket, replace=True)


def _upload_dir_as_prefix(s3: S3Hook, bucket: str, s3_prefix: str, local_dir: Path) -> int:
    """
    Uploads every file under local_dir to s3://bucket/s3_prefix/<relative>.
    Returns count uploaded.
    """
    s3_prefix = s3_prefix.strip("/")
    uploaded = 0
    for fp in local_dir.rglob("*"):
        if fp.is_file():
            rel = fp.relative_to(local_dir).as_posix()
            key = f"{s3_prefix}/{rel}"
            _upload_file(s3, bucket, key, fp)
            uploaded += 1
    return uploaded


# -----------------------------
# DAG
# -----------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="data_engineering_pipeline_minio_s3_taskflow",
    description="RAW->STAGING->CURATED data engineering pipeline (snapshots, traj, splits, augmentation, gate/dir datasets) on MinIO S3",
    default_args=default_args,
    start_date=datetime(2026, 2, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["ecld", "data", "minio", "raw", "staging", "curated"],
) as dag:

    @task
    def preflight_minio() -> Dict[str, str]:
        cfg = _cfg()
        s3 = S3Hook(aws_conn_id=cfg.aws_conn_id)

        # Fail fast if RAW prefix missing
        raw_root = f"{cfg.raw_prefix.strip('/')}/"
        keys = _list_keys(s3, cfg.bucket, raw_root)
        if not keys:
            raise AirflowFailException(f"RAW is empty: s3://{cfg.bucket}/{raw_root}")

        return {"bucket": cfg.bucket, "raw_prefix": cfg.raw_prefix}

    @task
    def stage_raw_locally() -> Dict[str, str]:
        """
        Downloads RAW objects from MinIO to local working dir:
          work/<run_id>/raw/{statements,transactions,income_history,credit_events,derived_features}/...
        """
        cfg = _cfg()
        s3 = S3Hook(aws_conn_id=cfg.aws_conn_id)

        run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID", "manual_run")
        base = Path(cfg.work_root) / "data_engineering" / run_id

        raw_local = base / "raw"
        if raw_local.exists():
            shutil.rmtree(raw_local, ignore_errors=True)
        _ensure_dir(raw_local)

        downloaded = _download_prefix_to_local(
            s3=s3,
            bucket=cfg.bucket,
            prefix=cfg.raw_prefix.strip("/"),
            local_dir=raw_local,
        )
        # Basic presence checks (folders)
        # Your snapshot script expects statements/ derived_features/ income_history/
        required_folders = ["statements", "income_history", "derived_features"]
        for f in required_folders:
            if not (raw_local / f).exists():
                raise AirflowFailException(f"Missing RAW folder in local stage: {raw_local/f}")

        return {
            "run_id": run_id,
            "base": str(base),
            "raw_local": str(raw_local),
            "downloaded": str(downloaded),
        }

    @task
    def build_monthly_snapshots(meta: Dict[str, str]) -> Dict[str, str]:
        """
        Generates monthly base snapshots (STAGING base):
          staging/snapshots_base_YYYY-MM.parquet
        """
        cfg = _cfg()
        base = Path(meta["base"])
        raw_local = Path(meta["raw_local"])

        staging_local = base / "staging"
        _ensure_dir(staging_local)

        _run(["python", cfg.build_monthly_snapshots_py, "--sim-dir", str(raw_local), "--out-dir", str(staging_local)])

        # ✅ monthly-only outputs
        monthly = sorted(staging_local.glob("snapshots_base_*.parquet"))
        if not monthly:
            raise AirflowFailException(f"No monthly base snapshots produced in {staging_local}")

        # return directory (not a single file)
        return {**meta, "staging_local": str(staging_local), "snapshots_base_dir": str(staging_local)}


    @task
    def feature_engineer_snapshots(meta: Dict[str, str]) -> Dict[str, str]:
        """
        Takes monthly base snapshots -> monthly engineered snapshots (STAGING FE):
          staging/feature_engineered/snapshots_YYYY-MM.parquet
        """
        cfg = _cfg()
        base = Path(meta["base"])
        staging_local = Path(meta["staging_local"])

        fe_dir = base / "staging" / "feature_engineered"
        _ensure_dir(fe_dir)

        # Your monthly-only feature_engineering.py reads snapshots_base_YYYY-MM.parquet from --in_dir
        _run(["python", cfg.feature_engineering_py, "--in_dir", str(staging_local), "--out_dir", str(fe_dir), "--mode", "batch"])

        # ✅ monthly-only verify
        monthly_fe = sorted(fe_dir.glob("snapshots_*.parquet"))
        if not monthly_fe:
            raise AirflowFailException(f"No monthly engineered snapshots produced in {fe_dir}")

        return {**meta, "snapshots_fe_dir": str(fe_dir)}


    @task
    def build_trajectories(meta: Dict[str, str]) -> Dict[str, str]:
        """
        Monthly engineered snapshots -> monthly trajectories + combined strict (STAGING):
          staging/trajectories/trajectories_strict_YYYY-MM.parquet
          staging/trajectories/trajectories_strict.parquet
        """
        cfg = _cfg()
        base = Path(meta["base"])
        snap_fe_dir = Path(meta["snapshots_fe_dir"])

        traj_dir = base / "staging" / "trajectories"
        _ensure_dir(traj_dir)

        # ✅ ensure combined strict is produced for downstream split step
        _run([
            "python", cfg.build_trajectories_py,
            "--snap_dir", str(snap_fe_dir),
            "--out_dir", str(traj_dir),
            "--combine_strict",
        ])
        print("[DEBUG] snap_fe_dir:", snap_fe_dir)
        print("[DEBUG] traj_dir:", traj_dir)
        print("[DEBUG] snapshots count:", len(list(Path(snap_fe_dir).glob("snapshots_*.parquet"))))
        print("[DEBUG] traj_dir contents:", list(Path(traj_dir).glob("*")))

        # verify monthly strict exists
        monthly_strict = sorted(traj_dir.glob("trajectories_strict_*.parquet"))
        if not monthly_strict:
            raise AirflowFailException(f"No monthly strict trajectories produced in {traj_dir}")

        # verify combined strict exists
        traj_strict = traj_dir / "trajectories_strict.parquet"
        _assert_exists_nonempty(traj_strict, min_bytes=1024)

        return {**meta, "traj_dir": str(traj_dir), "traj_strict": str(traj_strict)}


    @task
    def split_trajectories(meta: Dict[str, str]) -> Dict[str, str]:
        """
        trajectories_strict.parquet -> trajectories_train/val/test.parquet (STAGING)
        """
        cfg = _cfg()
        base = Path(meta["base"])
        traj_strict = Path(meta["traj_strict"])

        split_dir = base / "staging" / "splits"
        _ensure_dir(split_dir)

        _run([
            "python", cfg.splitting_py,
            "--in_parquet", str(traj_strict),
            "--out_dir", str(split_dir),
            "--group_col", "cust_id",
            "--label_col", "action_id",
            "--seed", str(cfg.seed),
            "--compression", cfg.compression,
        ])

        tr = split_dir / "trajectories_train.parquet"
        va = split_dir / "trajectories_val.parquet"
        te = split_dir / "trajectories_test.parquet"
        _assert_exists_nonempty(tr, min_bytes=1024)
        _assert_exists_nonempty(va, min_bytes=1024)
        _assert_exists_nonempty(te, min_bytes=1024)

        return {**meta, "split_dir": str(split_dir), "traj_train": str(tr), "traj_val": str(va), "traj_test": str(te)}

    @task
    def augment_train(meta: Dict[str, str]) -> Dict[str, str]:
        """
        Augment ONLY train trajectories (counterfactual CLI/CLD from HOLD) -> parquet dataset dir (CURATED)
        """
        cfg = _cfg()
        base = Path(meta["base"])
        traj_train = Path(meta["traj_train"])

        curated_local = base / "curated"
        _ensure_dir(curated_local)

        aug_dir = curated_local / "trajectories_train_aug"
        if aug_dir.exists():
            shutil.rmtree(aug_dir, ignore_errors=True)
        _ensure_dir(aug_dir)

        # augment_trajectories.py (your augmenter) typically wants --in-traj and --out-dir
        _run([
            "python", cfg.augmentation_py,
            "--in-traj", str(traj_train),
            "--out-dir", str(aug_dir),
            "--seed", str(cfg.seed),
            "--compression", cfg.compression,
        ])

        # Expect dataset parts + _summary.json
        summary = aug_dir / "_summary.json"
        _assert_exists_nonempty(summary, min_bytes=64)

        return {**meta, "curated_local": str(curated_local), "aug_dir": str(aug_dir)}

    @task
    def prepare_gate_dir(meta: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare gate + dir datasets:
          gated_train/val.parquet and dir_train/val.parquet (CURATED)
        Inputs:
          - merged_train = augmented train dir (3-class still present)
          - merged_test  = original val parquet
        """
        cfg = _cfg()
        curated_local = Path(meta["curated_local"])
        aug_dir = Path(meta["aug_dir"])
        traj_val = Path(meta["traj_val"])

        out_dir = curated_local / "gate_dir"
        _ensure_dir(out_dir)

        _run([
            "python", cfg.prepare_gate_dir_py,
            "--merged_train", str(aug_dir),
            "--merged_test", str(traj_val),
            "--out_dir", str(out_dir),
            "--batch_rows", "200000",
            "--compression", cfg.compression,
        ])

        expected = [
            out_dir / "gated_train.parquet",
            out_dir / "gated_val.parquet",
            out_dir / "dir_train.parquet",
            out_dir / "dir_val.parquet",
        ]
        for p in expected:
            _assert_exists_nonempty(p, min_bytes=1024)

        return {
            **meta,
            "gate_dir_out": str(out_dir),
            "gated_train": str(out_dir / "gated_train.parquet"),
            "gated_val": str(out_dir / "gated_val.parquet"),
            "dir_train": str(out_dir / "dir_train.parquet"),
            "dir_val": str(out_dir / "dir_val.parquet"),
        }

    @task
    def upload_staging_and_curated(meta: Dict[str, str]) -> Dict[str, str]:
        """
        Upload:
          STAGING:
            - snapshots_base_all_years.parquet
            - feature_engineered/snapshots_all_years.parquet
            - trajectories/*.parquet
            - splits/trajectories_{train,val,test}.parquet
          CURATED:
            - trajectories_train_aug/ (dataset dir)
            - gate_dir/*.parquet
        """
        cfg = _cfg()
        s3 = S3Hook(aws_conn_id=cfg.aws_conn_id)

        run_id = meta["run_id"]
        base = Path(meta["base"])

        staging_local = base / "staging"
        curated_local = base / "curated"

        # Write under s3://bucket/staging/<run_id>/... and curated/<run_id>/...
        staging_s3 = f"{cfg.staging_prefix.strip('/')}/{run_id}"
        curated_s3 = f"{cfg.curated_prefix.strip('/')}/{run_id}"

        st_up = _upload_dir_as_prefix(s3, cfg.bucket, staging_s3, staging_local)
        cu_up = _upload_dir_as_prefix(s3, cfg.bucket, curated_s3, curated_local)

        return {
            "staging_prefix_uploaded": f"s3://{cfg.bucket}/{staging_s3}",
            "curated_prefix_uploaded": f"s3://{cfg.bucket}/{curated_s3}",
            "staging_files_uploaded": str(st_up),
            "curated_files_uploaded": str(cu_up),
        }

    @task
    def verify_s3_written(upload_stats: Dict[str, str]) -> None:
        """
        Lightweight verification: confirm some key files exist in MinIO.
        """
        cfg = _cfg()
        s3 = S3Hook(aws_conn_id=cfg.aws_conn_id)

        # Just check prefixes have at least N objects
        # (strong verification can list specific objects if you want)
        staging_prefix = upload_stats["staging_prefix_uploaded"].split(f"s3://{cfg.bucket}/", 1)[1]
        curated_prefix = upload_stats["curated_prefix_uploaded"].split(f"s3://{cfg.bucket}/", 1)[1]

        st = _list_keys(s3, cfg.bucket, staging_prefix)
        cu = _list_keys(s3, cfg.bucket, curated_prefix)

        if not st or len(st) < 3:
            raise AirflowFailException(f"STAGING upload seems empty: {upload_stats['staging_prefix_uploaded']}")
        if not cu or len(cu) < 3:
            raise AirflowFailException(f"CURATED upload seems empty: {upload_stats['curated_prefix_uploaded']}")

        print("Upload verified:")
        print(upload_stats)

    # DAG wiring
    pf = preflight_minio()
    local = stage_raw_locally()
    snap = build_monthly_snapshots(local)
    fe = feature_engineer_snapshots(snap)
    traj = build_trajectories(fe)
    split = split_trajectories(traj)
    aug = augment_train(split)
    gd = prepare_gate_dir(aug)
    up = upload_staging_and_curated(gd)
    verify_s3_written(up)
