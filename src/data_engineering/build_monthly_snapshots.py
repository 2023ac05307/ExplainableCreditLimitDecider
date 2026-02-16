# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# build_base_monthly_snapshots_parquet.py
# --------------------------------------
# INPUT  : CSV files (simulator outputs)
# OUTPUT : Base Parquet snapshots (NO rolling feature engineering)

# Writes:
#   - <out_dir>/snapshots_base_all_years.parquet
#   - <out_dir>/snapshots_base_<YYYY>.parquet
#   - <sim_dir>/action_map.parquet (ensured)
# """

# import os, glob, logging
# import pandas as pd
# import numpy as np

# ACTION_MAP_DEFAULT = pd.DataFrame(
#     [
#         {"action_id": 0, "action_name": "HOLD"},
#         {"action_id": 1, "action_name": "CLI"},
#         {"action_id": 2, "action_name": "CLD"},
#     ]
# )

# STATUS_TO_ID = {"current":0,"dpd30":1,"dpd60":2,"dpd90":3,"default":4,"closed":5}
# REGIME_TO_ID = {"boom":0,"normal":1,"recession":2}

# PARQUET_ENGINE = "pyarrow"
# PARQUET_COMPRESSION = "snappy"


# def _require_pyarrow():
#     try:
#         import pyarrow  # noqa
#     except Exception as e:
#         raise RuntimeError("Install pyarrow: pip install pyarrow") from e


# def write_parquet(df: pd.DataFrame, path: str):
#     _require_pyarrow()
#     df.to_parquet(path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)


# def read_yearly(prefix, base_dir, subdir=None):
#     root = os.path.join(base_dir, subdir) if subdir else base_dir
#     files = sorted(glob.glob(os.path.join(root, f"{prefix}_*.csv")))
#     if not files:
#         return pd.DataFrame()
#     return pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)


# def ensure_action_map(sim_dir):
#     pqt = os.path.join(sim_dir, "action_map.parquet")
#     legacy_csv = os.path.join(sim_dir, "action_map.csv")

#     if os.path.exists(pqt):
#         return pd.read_parquet(pqt, engine=PARQUET_ENGINE)
#     if os.path.exists(legacy_csv):
#         df = pd.read_csv(legacy_csv, low_memory=False)
#         write_parquet(df, pqt)
#         return df

#     write_parquet(ACTION_MAP_DEFAULT, pqt)
#     return ACTION_MAP_DEFAULT


# def load_customers_static(sim_dir):
#     customers_path = os.path.join(sim_dir, "customers.csv")
#     if not os.path.exists(customers_path):
#         return None

#     customers = pd.read_csv(customers_path, low_memory=False)
#     static_cols = [
#         "cust_id",
#         "num_open_accounts",
#         "num_closed_accounts",
#         "total_open_credit",
#         "external_delinquency_12m",
#         "residential_stability_years",
#         "employment_stability_years",
#         "industry_risk_score",
#     ]
#     static_cols = [c for c in static_cols if c in customers.columns]
#     if "cust_id" not in static_cols or len(static_cols) == 1:
#         return None
#     return customers[static_cols].copy()


# def build_base_monthly_snapshots(sim_dir: str, out_dir: str) -> str:
#     os.makedirs(out_dir, exist_ok=True)

#     statements = read_yearly("statements", sim_dir, subdir="statements")
#     derived    = read_yearly("derived_features", sim_dir, subdir="derived_features")
#     income     = read_yearly("income_history", sim_dir, subdir="income_history")

#     customers_static = load_customers_static(sim_dir)

#     if statements.empty:
#         raise RuntimeError("No statements_YYYY.csv found in statements/.")
#     if derived.empty:
#         raise RuntimeError("No derived_features_YYYY.csv found in derived_features/.")

#     statements["statement_date"] = pd.to_datetime(statements["statement_date"])
#     derived["date"] = pd.to_datetime(derived["date"])
#     if not income.empty:
#         income["date"] = pd.to_datetime(income["date"])

#     snap = statements.merge(
#         derived,
#         left_on=["cust_id", "statement_date"],
#         right_on=["cust_id", "date"],
#         how="left",
#         suffixes=("", "_drv"),
#     )

#     # Income by YEAR-MONTH
#     if not income.empty:
#         snap = snap.copy()
#         income = income.copy()

#         snap["year_month"] = snap["statement_date"].dt.to_period("M").astype(str)
#         income["year_month"] = income["date"].dt.to_period("M").astype(str)

#         income = (
#             income.sort_values(["cust_id", "date"])
#                   .drop_duplicates(["cust_id", "year_month"], keep="last")
#         )

#         snap = snap.merge(
#             income[["cust_id", "year_month", "monthly_income"]],
#             on=["cust_id", "year_month"],
#             how="left",
#         )
#         snap = snap.drop(columns=["year_month"], errors="ignore")
#         snap["monthly_income"] = pd.to_numeric(snap.get("monthly_income", 0.0), errors="coerce").fillna(0.0)

#     if customers_static is not None:
#         snap = snap.merge(customers_static, on="cust_id", how="left")

#     # Cleanup
#     if "date" in snap.columns:
#         snap = snap.drop(columns=["date"])

#     if "balance" not in snap.columns and "new_balance" in snap.columns:
#         snap = snap.rename(columns={"new_balance": "balance"})

#     if "action_id" not in snap.columns:
#         snap["action_id"] = 0
#     if "magnitude_pct" not in snap.columns:
#         snap["magnitude_pct"] = 0.0

#     snap["action_id"] = pd.to_numeric(snap["action_id"], errors="coerce").fillna(0).astype(int)
#     snap["magnitude_pct"] = pd.to_numeric(snap["magnitude_pct"], errors="coerce").fillna(0.0).astype(float)

#     for c in ["revenue", "risk_penalty", "churn_penalty"]:
#         if c not in snap.columns:
#             snap[c] = 0.0

#     snap["reward"] = (
#         pd.to_numeric(snap["revenue"], errors="coerce").fillna(0.0)
#         - pd.to_numeric(snap["risk_penalty"], errors="coerce").fillna(0.0)
#         - pd.to_numeric(snap["churn_penalty"], errors="coerce").fillna(0.0)
#     )

#     # keep raw strings for later done-logic + encoding
#     snap["status_raw"] = snap.get("status", "current").astype(str)
#     snap["macro_regime_raw"] = snap.get("macro_regime", "normal").astype(str)

#     snap["status_id"] = snap["status_raw"].map(STATUS_TO_ID).fillna(0).astype(int)
#     snap["macro_regime_id"] = snap["macro_regime_raw"].map(REGIME_TO_ID).fillna(1).astype(int)

#     snap = snap.sort_values(["cust_id", "statement_date"]).reset_index(drop=True)

#     # Write BASE snapshots
#     snap["year"] = snap["statement_date"].dt.year
#     for y, dfy in snap.groupby("year", sort=True):
#         write_parquet(dfy, os.path.join(out_dir, f"snapshots_base_{y}.parquet"))

#     write_parquet(snap, os.path.join(out_dir, "snapshots_base_all_years.parquet"))
#     return os.path.join(out_dir, "snapshots_base_all_years.parquet")


# def main():
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--sim-dir", required=True)
#     p.add_argument("--out-dir", default="rl_dataset")
#     args = p.parse_args()

#     _require_pyarrow()
#     ensure_action_map(args.sim_dir)
#     build_base_monthly_snapshots(args.sim_dir, args.out_dir)

#     print("Done. Wrote BASE snapshots:")
#     print(f" - {args.out_dir}/snapshots_base_all_years.parquet + snapshots_base_YYYY.parquet")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_monthly_snapshots_monthly_only.py
--------------------------------------
INPUT  : CSV files (simulator outputs) in monthly format:
         statements/statements_YYYY-MM.csv
         derived_features/derived_features_YYYY-MM.csv
         income_history/income_history_YYYY-MM.csv (optional but supported)
OUTPUT : Base Parquet snapshots (NO rolling feature engineering)

Writes ONLY:
  - <out_dir>/snapshots_base_YYYY-MM.parquet

Also ensures:
  - <sim_dir>/action_map.parquet
"""

import os, glob
import pandas as pd

ACTION_MAP_DEFAULT = pd.DataFrame(
    [
        {"action_id": 0, "action_name": "HOLD"},
        {"action_id": 1, "action_name": "CLI"},
        {"action_id": 2, "action_name": "CLD"},
    ]
)

STATUS_TO_ID = {"current":0,"dpd30":1,"dpd60":2,"dpd90":3,"default":4,"closed":5}
REGIME_TO_ID = {"boom":0,"normal":1,"recession":2}

PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


def _require_pyarrow():
    try:
        import pyarrow  # noqa
    except Exception as e:
        raise RuntimeError("Install pyarrow: pip install pyarrow") from e


def write_parquet(df: pd.DataFrame, path: str):
    _require_pyarrow()
    df.to_parquet(path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)


def read_partitioned_csv(prefix: str, base_dir: str, subdir: str | None = None) -> pd.DataFrame:
    """
    Reads all CSV files matching <prefix>_*.csv from <base_dir>/<subdir>/ (or base_dir).
    Works for both yearly and monthly naming, e.g. statements_2006.csv or statements_2006-01.csv.
    """
    root = os.path.join(base_dir, subdir) if subdir else base_dir
    files = sorted(glob.glob(os.path.join(root, f"{prefix}_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)


def ensure_action_map(sim_dir: str) -> pd.DataFrame:
    pqt = os.path.join(sim_dir, "action_map.parquet")
    legacy_csv = os.path.join(sim_dir, "action_map.csv")

    if os.path.exists(pqt):
        return pd.read_parquet(pqt, engine=PARQUET_ENGINE)
    if os.path.exists(legacy_csv):
        df = pd.read_csv(legacy_csv, low_memory=False)
        write_parquet(df, pqt)
        return df

    write_parquet(ACTION_MAP_DEFAULT, pqt)
    return ACTION_MAP_DEFAULT


def load_customers_static(sim_dir: str) -> pd.DataFrame | None:
    customers_path = os.path.join(sim_dir, "customers.csv")
    if not os.path.exists(customers_path):
        return None

    customers = pd.read_csv(customers_path, low_memory=False)
    static_cols = [
        "cust_id",
        "num_open_accounts",
        "num_closed_accounts",
        "total_open_credit",
        "external_delinquency_12m",
        "residential_stability_years",
        "employment_stability_years",
        "industry_risk_score",
    ]
    static_cols = [c for c in static_cols if c in customers.columns]
    if "cust_id" not in static_cols or len(static_cols) == 1:
        return None
    return customers[static_cols].copy()


def build_base_monthly_snapshots(sim_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    statements = read_partitioned_csv("statements", sim_dir, subdir="statements")
    derived    = read_partitioned_csv("derived_features", sim_dir, subdir="derived_features")
    income     = read_partitioned_csv("income_history", sim_dir, subdir="income_history")

    customers_static = load_customers_static(sim_dir)

    if statements.empty:
        raise RuntimeError("No statements_YYYY-MM.csv found in statements/.")
    if derived.empty:
        raise RuntimeError("No derived_features_YYYY-MM.csv found in derived_features/.")

    # Parse dates
    statements["statement_date"] = pd.to_datetime(statements["statement_date"])
    derived["date"] = pd.to_datetime(derived["date"])
    if not income.empty:
        income["date"] = pd.to_datetime(income["date"])

    # Merge statements + derived on exact date (cust_id + date)
    snap = statements.merge(
        derived,
        left_on=["cust_id", "statement_date"],
        right_on=["cust_id", "date"],
        how="left",
        suffixes=("", "_drv"),
    )

    # Income by YEAR-MONTH (cust_id + year_month)
    snap["year_month"] = snap["statement_date"].dt.to_period("M").astype(str)

    if not income.empty:
        income = income.copy()
        income["year_month"] = income["date"].dt.to_period("M").astype(str)

        income = (
            income.sort_values(["cust_id", "date"])
                  .drop_duplicates(["cust_id", "year_month"], keep="last")
        )

        snap = snap.merge(
            income[["cust_id", "year_month", "monthly_income"]],
            on=["cust_id", "year_month"],
            how="left",
        )
        snap["monthly_income"] = pd.to_numeric(snap.get("monthly_income", 0.0), errors="coerce").fillna(0.0)
    else:
        snap["monthly_income"] = 0.0

    # Add customer static cols if present
    if customers_static is not None:
        snap = snap.merge(customers_static, on="cust_id", how="left")

    # Cleanup duplicate date column from derived
    if "date" in snap.columns:
        snap = snap.drop(columns=["date"], errors="ignore")

    # Normalize field names
    if "balance" not in snap.columns and "new_balance" in snap.columns:
        snap = snap.rename(columns={"new_balance": "balance"})

    # Defaults
    if "action_id" not in snap.columns:
        snap["action_id"] = 0
    if "magnitude_pct" not in snap.columns:
        snap["magnitude_pct"] = 0.0

    snap["action_id"] = pd.to_numeric(snap["action_id"], errors="coerce").fillna(0).astype(int)
    snap["magnitude_pct"] = pd.to_numeric(snap["magnitude_pct"], errors="coerce").fillna(0.0).astype(float)

    for c in ["revenue", "risk_penalty", "churn_penalty"]:
        if c not in snap.columns:
            snap[c] = 0.0

    snap["reward"] = (
        pd.to_numeric(snap["revenue"], errors="coerce").fillna(0.0)
        - pd.to_numeric(snap["risk_penalty"], errors="coerce").fillna(0.0)
        - pd.to_numeric(snap["churn_penalty"], errors="coerce").fillna(0.0)
    )

    # Encodings
    snap["status_raw"] = snap.get("status", "current").astype(str)
    snap["macro_regime_raw"] = snap.get("macro_regime", "normal").astype(str)

    snap["status_id"] = snap["status_raw"].map(STATUS_TO_ID).fillna(0).astype(int)
    snap["macro_regime_id"] = snap["macro_regime_raw"].map(REGIME_TO_ID).fillna(1).astype(int)

    # Deterministic order
    snap = snap.sort_values(["cust_id", "statement_date"]).reset_index(drop=True)

    # ✅ MONTHLY-ONLY WRITE (no yearly, no all_years)
    for ym, dfm in snap.groupby("year_month", sort=True):
        out_path = os.path.join(out_dir, f"snapshots_base_{ym}.parquet")
        write_parquet(dfm, out_path)

    print("Done. Wrote BASE snapshots per month:")
    print(f" - {out_dir}/snapshots_base_YYYY-MM.parquet")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--out-dir", default="rl_dataset")
    args = p.parse_args()

    _require_pyarrow()
    ensure_action_map(args.sim_dir)
    build_base_monthly_snapshots(args.sim_dir, args.out_dir)


if __name__ == "__main__":
    main()

