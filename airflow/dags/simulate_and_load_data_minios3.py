"""
simulate_and_load_raw_minio.py

Airflow DAG to:
1) Run 20-year credit portfolio simulator
2) Verify outputs (expected yearly CSV partitions)
3) Upload to MinIO (S3-compatible) into:
   s3://explainablecreditlimitdecider/raw/{statements|transactions|income_history|credit_events}/...

Assumptions:
- Your simulator writes yearly partitioned files as:
    statements_YYYY.csv
    transactions_YYYY.csv
    income_history_YYYY.csv
    credit_events_YYYY.csv
    derived_features_YYYY.csv (optional)
  plus customers.csv and action_map.csv (optional; we can upload too if you want)
- MinIO is reachable from Airflow container/network.
"""

from __future__ import annotations

import os
import re
import glob
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable

# Airflow AWS provider works for MinIO as long as you set endpoint_url in the connection extra.
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class SimConfig:
    # Simulator execution
    python_bin: str
    simulator_path: str
    customers: int
    years: int
    start_date: str
    seed: int
    out_dir: str

    # Storage
    bucket: str
    raw_prefix: str  # usually "raw"
    include_derived_features: bool
    include_customers_and_action_map: bool

    # Operational
    clean_out_dir_before_run: bool


def _load_config() -> SimConfig:
    """
    Centralized config loader: Variables > env defaults.
    You can set these in Airflow UI -> Admin -> Variables.
    """

    def v(key: str, default: str) -> str:
        return Variable.get(key, default_var=os.getenv(key, default))

    return SimConfig(
        python_bin=v("SIM_PYTHON_BIN", "python"),
        simulator_path=v("SIMULATOR_SCRIPT_PATH", "/opt/airflow/app/src/data_engineering/simulate_finance_stream.py"),
        customers=int(v("SIM_CUSTOMERS", "20000")),
        years=int(v("SIM_YEARS", "20")),
        start_date=v("SIM_START_DATE", "2006-01-01"),
        seed=int(v("SIM_SEED", "42")),
        out_dir=v("SIM_OUT_DIR", "/opt/airflow/app/data/sim_output_v2"),

        bucket=v("RAW_BUCKET", "explainablecreditlimitdecider"),
        raw_prefix=v("RAW_PREFIX", "raw"),
        include_derived_features=v("INCLUDE_DERIVED_FEATURES", "true").lower() == "true",
        include_customers_and_action_map=v("INCLUDE_CUSTOMERS_AND_ACTION_MAP", "true").lower() == "true",

        clean_out_dir_before_run=v("CLEAN_OUT_DIR_BEFORE_RUN", "true").lower() == "true",
    )

import csv

def _assert_csv_has_data_rows(path: str, min_data_rows: int = 1) -> None:
    """
    Ensures CSV exists and has at least `min_data_rows` rows excluding header.
    Works for tiny CSVs (like action maps) without relying on file size.
    """
    if not os.path.exists(path):
        raise AirflowFailException(f"Missing file: {path}")

    # Must not be empty file
    if os.path.getsize(path) <= 0:
        raise AirflowFailException(f"Empty file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise AirflowFailException(f"CSV has no rows: {path}")

    # assume first row is header (common), require at least N data rows
    data_rows = max(0, len(rows) - 1)
    if data_rows < min_data_rows:
        raise AirflowFailException(
            f"CSV has insufficient data rows: {path} (data_rows={data_rows}, required={min_data_rows})"
        )


# Map simulator output file prefix -> S3 raw folder
OUTPUT_TO_RAW_FOLDER = {
    "statements": "statements",
    "transactions": "transactions",
    "income_history": "income_history",
    "credit_events": "credit_events",
    # optional:
    "derived_features": "derived_features",
}


# -----------------------------
# Monthly partition helpers (Option A)
# -----------------------------
_DEFAULT_DATE_COL_CANDIDATES = {
    "statements": ["statement_date", "stmt_date", "statement_month", "month", "date"],
    "transactions": ["txn_date", "transaction_date", "posted_date", "date"],
    "income_history": ["income_date", "as_of_date", "date"],
    "credit_events": ["event_date", "as_of_date", "date"],
    "derived_features": ["snapshot_month", "as_of_month", "month", "date"],
}

def _extract_yyyy_mm(raw: str) -> str | None:
    """Best-effort extraction of YYYY-MM from common date formats."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # ISO-like: YYYY-MM or YYYY-MM-DD
    if len(s) >= 7 and s[4] == "-" and s[6] == "-" and s[:4].isdigit() and s[5:7].isdigit():
        return s[:7]
    if len(s) >= 7 and s[4] == "-" and s[:4].isdigit() and s[5:7].isdigit():
        return s[:7]

    # Compact: YYYYMMDD or YYYYMM
    if len(s) >= 6 and s[:6].isdigit():
        yyyy = s[:4]
        mm = s[4:6]
        if 1 <= int(mm) <= 12:
            return f"{yyyy}-{mm}"

    # Slash: MM/DD/YYYY or DD/MM/YYYY (ambiguous). We'll assume last part is year.
    if "/" in s:
        parts = s.split("/")
        if len(parts) >= 3 and parts[-1].strip()[:4].isdigit():
            yyyy = parts[-1].strip()[:4]
            p1 = parts[0].strip()
            p2 = parts[1].strip()
            # choose the one that can be a month (1..12)
            mm = None
            for cand in (p1, p2):
                if cand.isdigit() and 1 <= int(cand) <= 12:
                    mm = int(cand)
                    break
            if mm is not None:
                return f"{yyyy}-{mm:02d}"

    return None

def _split_yearly_csv_to_monthly(
    in_path: str,
    dataset: str,
    monthly_root: str,
    date_cols: List[str],
) -> List[str]:
    """
    Splits a yearly CSV into monthly CSVs:
      <monthly_root>/<dataset>/<dataset>_YYYY-MM.csv
    Returns list of created file paths.
    """
    created: List[str] = []
    _safe_mkdir(os.path.join(monthly_root, dataset))

    with open(in_path, "r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader, None)
        if not header:
            return created

        # Find a usable date column
        header_lc = [h.strip().lower() for h in header]
        date_idx = None
        for c in date_cols:
            c_lc = c.lower()
            if c_lc in header_lc:
                date_idx = header_lc.index(c_lc)
                break
        if date_idx is None:
            # fallback: try any header containing 'date' or 'month'
            for i, h in enumerate(header_lc):
                if "date" in h or "month" in h:
                    date_idx = i
                    break

        if date_idx is None:
            raise AirflowFailException(
                f"Could not find a date/month column in {in_path}. " 
                f"Tried: {date_cols} and fallback ('date'/'month' contains)."
            )

        writers: Dict[str, tuple] = {}  # yyyy_mm -> (file_handle, csv_writer)
        try:
            for row in reader:
                if not row:
                    continue
                if date_idx >= len(row):
                    continue
                yyyy_mm = _extract_yyyy_mm(row[date_idx])
                if not yyyy_mm:
                    continue

                out_path = os.path.join(monthly_root, dataset, f"{dataset}_{yyyy_mm}.csv")
                if yyyy_mm not in writers:
                    # open new file and write header once
                    fh = open(out_path, "w", newline="", encoding="utf-8")
                    w = csv.writer(fh)
                    w.writerow(header)
                    writers[yyyy_mm] = (fh, w)
                    created.append(out_path)

                writers[yyyy_mm][1].writerow(row)
        finally:
            for fh, _ in writers.values():
                try:
                    fh.close()
                except Exception:
                    pass

    return created


def _expected_years(start_date: str, years: int) -> List[int]:
    """
    For start_date=2006-01-01 and years=20 => [2006..2025]
    """
    start_year = datetime.fromisoformat(start_date).year
    # 20 years => 2006..2025 inclusive
    return list(range(start_year, start_year + years))


def _expected_months(start_date: str, years: int) -> List[str]:
    """For start_date=2006-01-01 and years=20 => ['2006-01'..'2025-12']"""
    start = datetime.fromisoformat(start_date)
    months = []
    y = start.year
    m = start.month
    total = years * 12
    for _ in range(total):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_cmd(cmd: List[str], cwd: str | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AirflowFailException(
            "Simulator command failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )


def _list_yearly_files(out_dir: str) -> Dict[str, Dict[int, str]]:
    """
    Returns: {dataset: {year: filepath}}
    For example:
      {
        "statements": {2006: ".../statements_2006.csv", ...},
        "transactions": {2006: ".../transactions_2006.csv", ...},
        ...
      }
    """
    datasets = list(OUTPUT_TO_RAW_FOLDER.keys())
    found: Dict[str, Dict[int, str]] = {d: {} for d in datasets}

    # match e.g. statements_2006.csv
    pattern = re.compile(r"^(?P<ds>[a-z_]+)_(?P<year>\d{4})\.csv$")

    for p in glob.glob(os.path.join(out_dir, "*.csv")):
        name = os.path.basename(p)
        m = pattern.match(name)
        if not m:
            continue
        ds = m.group("ds")
        year = int(m.group("year"))
        if ds in found:
            found[ds][year] = p

    return found


def _assert_non_empty_file(path: str, min_bytes: int = 128) -> None:
    if not os.path.exists(path):
        raise AirflowFailException(f"Missing file: {path}")
    size = os.path.getsize(path)
    if size < min_bytes:
        raise AirflowFailException(f"File too small / likely empty: {path} (size={size} bytes)")


def _make_s3_key(prefix: str, folder: str, filename: str) -> str:
    prefix = prefix.strip("/")

    # raw/statements/statements_2006.csv
    return f"{prefix}/{folder}/{filename}"


def _ensure_bucket(s3: S3Hook, bucket: str) -> None:
    # In MinIO, head_bucket works. If missing, create it.
    if not s3.check_for_bucket(bucket_name=bucket):
        s3.create_bucket(bucket_name=bucket)


# -----------------------------
# DAG
# -----------------------------

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


with DAG(
    dag_id="simulate_20yr_credit_data_to_minio_raw",
    description="Simulate 20-year credit data and upload partitions to MinIO raw zone",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 1),
    schedule=None,  # manual trigger for demo; change to cron if needed
    catchup=False,
    max_active_runs=1,
    tags=["dissertation", "simulator", "minio", "raw"],
) as dag:

    @task
    def simulate() -> Dict[str, str]:
        """
        Executes simulator script and returns metadata.
        """
        conf = _load_config()

        if conf.clean_out_dir_before_run and os.path.exists(conf.out_dir):
            shutil.rmtree(conf.out_dir, ignore_errors=True)

        _safe_mkdir(conf.out_dir)

        cmd = [
            conf.python_bin,
            conf.simulator_path,
            "--customers", str(conf.customers),
            "--years", str(conf.years),
            "--start", conf.start_date,
            "--seed", str(conf.seed),
            "--out", conf.out_dir,
            # by default your script partitions by year; keep it ON
        ]
        _run_cmd(cmd)

        return {
            "out_dir": conf.out_dir,
            "start_date": conf.start_date,
            "years": str(conf.years),
        }

    @task
    def verify_outputs(meta: Dict[str, str]) -> Dict[str, Dict[int, str]]:
        """
        Verify yearly files exist for each dataset and are non-empty.
        Returns discovered file map for downstream upload.
        """
        conf = _load_config()
        out_dir = meta["out_dir"]

        expected_years = _expected_years(conf.start_date, conf.years)
        found = _list_yearly_files(out_dir)

        required_datasets = ["statements", "transactions", "income_history", "credit_events"]
        if conf.include_derived_features:
            required_datasets.append("derived_features")

        errors: List[str] = []

        for ds in required_datasets:
            years_map = found.get(ds, {})
            for y in expected_years:
                p = years_map.get(y)
                if not p:
                    errors.append(f"[MISSING] {ds}_{y}.csv")
                else:
                    try:
                        _assert_non_empty_file(p)
                    except Exception as e:
                        errors.append(f"[BAD] {ds}_{y}.csv -> {e}")

        if conf.include_customers_and_action_map:
            # customers should be "real" data
            customers_path = os.path.join(out_dir, "customers.csv")
            try:
                _assert_csv_has_data_rows(customers_path, min_data_rows=1)
            except Exception as e:
                errors.append(f"[BAD] customers.csv -> {e}")

            # action_map is often tiny, but should still have at least 1 row after header
            action_map_path = os.path.join(out_dir, "action_map.csv")
            try:
                _assert_csv_has_data_rows(action_map_path, min_data_rows=1)
            except Exception as e:
                errors.append(f"[BAD] action_map.csv -> {e}")


        if errors:
            raise AirflowFailException("Verification failed:\n" + "\n".join(errors))

        # Return only what we intend to upload
        upload_map: Dict[str, Dict[int, str]] = {}
        for ds in required_datasets:
            upload_map[ds] = {y: found[ds][y] for y in expected_years}

        if conf.include_customers_and_action_map:
            upload_map["__meta__"] = {
                0: os.path.join(out_dir, "customers.csv"),
                1: os.path.join(out_dir, "action_map.csv"),
            }

        return upload_map

   
    @task
    def yearly_to_monthly_partitions(upload_map: Dict[str, Dict[int, str]]) -> Dict[str, List[str] | Dict[str, str]]:
        """
        Option A:
        - Keep simulator output as yearly CSVs
        - Create monthly CSVs under: <out_dir>/monthly/<dataset>/<dataset>_YYYY-MM.csv

        Returns a map:
          {
            "statements": ["/.../monthly/statements/statements_2006-01.csv", ...],
            "transactions": [...],
            ...
            "__meta__": {"customers": "/.../customers.csv", "action_map": "/.../action_map.csv"}
          }
        """
        conf = _load_config()

        # Monthly output root
        monthly_root = os.path.join(conf.out_dir, "monthly")
        _safe_mkdir(monthly_root)

        expected_months = set(_expected_months(conf.start_date, conf.years))

        monthly_map: Dict[str, List[str] | Dict[str, str]] = {}

        for ds, years_map in upload_map.items():
            if ds == "__meta__":
                continue

            date_cols = _DEFAULT_DATE_COL_CANDIDATES.get(ds, ["date", "month"])
            created_all: List[str] = []
            for _, yearly_path in years_map.items():
                created = _split_yearly_csv_to_monthly(
                    in_path=yearly_path,
                    dataset=ds,
                    monthly_root=monthly_root,
                    date_cols=date_cols,
                )
                created_all.extend(created)

            # Strong completeness check for statements (acts like your "month snapshot" base)
            if ds == "statements":
                present = set()
                for p in created_all:
                    # statements_YYYY-MM.csv
                    base = os.path.basename(p)
                    mm = base.replace(f"{ds}_", "").replace(".csv", "")
                    if len(mm) == 7:
                        present.add(mm)
                missing = sorted(list(expected_months - present))
                if missing:
                    raise AirflowFailException(
                        f"Monthly partitioning incomplete for statements. Missing months (sample up to 15): {missing[:15]}"
                    )

            monthly_map[ds] = sorted(list(set(created_all)))

        # pass-through meta files (customers/action_map)
        if conf.include_customers_and_action_map and "__meta__" in upload_map:
            meta = upload_map["__meta__"]
            monthly_map["__meta__"] = {
                "customers": meta.get(0),
                "action_map": meta.get(1),
            }

        return monthly_map

    @task
    def upload_to_minio(monthly_map: Dict[str, List[str] | Dict[str, str]]) -> Dict[str, int]:
        """
        Upload monthly partitioned files to MinIO (S3-compatible) as:

          raw/statements/statements_YYYY-MM.csv
          raw/transactions/transactions_YYYY-MM.csv
          raw/income_history/income_history_YYYY-MM.csv
          raw/credit_events/credit_events_YYYY-MM.csv
          raw/derived_features/derived_features_YYYY-MM.csv (optional)

          raw/customers.csv
          raw/action_map.csv
        """
        conf = _load_config()

        s3 = S3Hook(aws_conn_id="minio_s3")
        _ensure_bucket(s3, conf.bucket)

        uploaded = 0

        # Upload monthly datasets
        for ds, paths in monthly_map.items():
            if ds == "__meta__":
                continue
            if ds == "derived_features" and not conf.include_derived_features:
                continue

            folder = OUTPUT_TO_RAW_FOLDER[ds]
            for local_path in (paths or []):
                filename = os.path.basename(local_path)
                key = _make_s3_key(conf.raw_prefix, folder, filename)  # raw/<folder>/<filename>
                s3.load_file(
                    filename=local_path,
                    key=key,
                    bucket_name=conf.bucket,
                    replace=True,
                )
                uploaded += 1

        # Upload customers.csv and action_map.csv at raw/ root
        if conf.include_customers_and_action_map and "__meta__" in monthly_map:
            meta = monthly_map["__meta__"] or {}
            for name in ["customers", "action_map"]:
                local_path = meta.get(name)
                if not local_path:
                    continue
                filename = os.path.basename(local_path)
                key = f"{conf.raw_prefix.strip('/')}/{filename}"
                s3.load_file(
                    filename=local_path,
                    key=key,
                    bucket_name=conf.bucket,
                    replace=True,
                )
                uploaded += 1

        return {"uploaded_files": uploaded}


    @task
    def post_upload_smoke_check(stats: Dict[str, int]) -> None:
        """
        Final sanity log (useful in prod): confirms counts.
        """
        if stats.get("uploaded_files", 0) <= 0:
            raise AirflowFailException("Upload completed but uploaded_files==0 (unexpected).")
        print(json.dumps(stats, indent=2))

    # Wiring
    meta = simulate()
    verified = verify_outputs(meta)
    monthly = yearly_to_monthly_partitions(verified)
    up_stats = upload_to_minio(monthly)
    post_upload_smoke_check(up_stats)
