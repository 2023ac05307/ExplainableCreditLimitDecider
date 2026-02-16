#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_gate_and_dir_datasets_parquet.py (STREAMING SAFE, FIXED)
---------------------------------------------------------------
INPUT  : Parquet (single file OR dataset directory)
OUTPUT : Parquet files only (NO CSV)

Writes:
  - gated_train.parquet  (binary gate label: action_id_gate, but keeps action_id_3cls)
  - gated_val.parquet
  - dir_train.parquet    (keeps original action_id in {1,2} and adds dir_label)
  - dir_val.parquet

Conventions:
- Original 3-class action_id: 0=HOLD, 1=CLI, 2=CLD
- Gate label:
    action_id_gate: 0=HOLD, 1=NON_HOLD
- Dir label (binary, for training convenience):
    dir_label: 0=CLD, 1=CLI
  while keeping:
    action_id (unchanged): 1=CLI, 2=CLD
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

A_HOLD, A_CLI, A_CLD = 0, 1, 2

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("Parquet support requires pyarrow. Install: pip install pyarrow") from e


def iter_parquet(path: str, batch_rows: int = 200_000, columns=None):
    """
    Stream Parquet as pandas DataFrames.
    Supports:
      - single .parquet file
      - parquet dataset directory (e.g., part-*.parquet)
    """
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_rows)
    for rb in scanner.to_batches():
        df = rb.to_pandas()
        df = df.replace([np.inf, -np.inf], np.nan)
        yield df


def ensure_cols(df: pd.DataFrame):
    # ensure common cols exist
    if "reward" not in df: df["reward"] = 0.0
    if "done" not in df: df["done"] = 0.0
    if "sample_weight" not in df: df["sample_weight"] = 1.0
    if "magnitude_pct" not in df: df["magnitude_pct"] = 0.0
    if "cust_id" not in df: df["cust_id"] = -1


def keep_relevant(df: pd.DataFrame):
    base = ["cust_id", "action_id", "reward", "done", "sample_weight", "magnitude_pct"]
    feats = [c for c in df.columns if c.startswith("s_") or c.startswith("s1_")]
    cols = base + feats
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def validate_actions(a: pd.Series):
    bad = ~a.isin([A_HOLD, A_CLI, A_CLD])
    if bad.any():
        raise RuntimeError(f"Bad action_id values: {a[bad].unique()}")


def make_gate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gate dataset:
      - Preserve original 3-class action in action_id_3cls
      - Overwrite action_id to binary gate label (0=HOLD, 1=NON_HOLD)
      - magnitude_pct = 0.0 (unused for gate)
    """
    a = pd.to_numeric(df["action_id"], errors="coerce").fillna(0).astype(int)
    validate_actions(a)

    out = df.copy()
    out["action_id_3cls"] = a                 # keep original 0/1/2 for analysis
    out["action_id"] = (a != A_HOLD).astype(int)  # <-- IMPORTANT: trainer expects this
    out["magnitude_pct"] = 0.0
    return out


def make_dir_chunk(df: pd.DataFrame):
    """
    Dir dataset:
      - Keep only original CLI/CLD rows (action_id in {1,2})
      - Preserve original 3-class action in action_id_3cls (1/2)
      - Overwrite action_id to binary dir label:
          0 = CLD
          1 = CLI
    """
    a = pd.to_numeric(df["action_id"], errors="coerce").fillna(0).astype(int)
    validate_actions(a)

    mask = a.isin([A_CLI, A_CLD])  # keep only 1,2
    if not mask.any():
        return None

    out = df.loc[mask].copy()
    a2 = a.loc[mask]

    out["action_id_3cls"] = a2                   # 1=CLI, 2=CLD (preserved)
    out["action_id"] = (a2 == A_CLI).astype(int) # 1=CLI, 0=CLD

    out["magnitude_pct"] = (
        pd.to_numeric(out.get("magnitude_pct", 0.0), errors="coerce")
          .fillna(0.0)
          .clip(lower=0.0)
    )
    return out



class ParquetAppender:
    """Append-write parquet safely using a ParquetWriter."""
    def __init__(self, out_path: Path, compression: str = "snappy"):
        self.out_path = out_path
        self.compression = compression
        self.writer = None
        self.schema = None
        if self.out_path.exists():
            self.out_path.unlink()

    def append(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return

        table = pa.Table.from_pandas(df, preserve_index=False)

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                where=str(self.out_path),
                schema=self.schema,
                compression=self.compression,
            )
        else:
            # Align to schema if chunk columns vary
            if table.schema != self.schema:
                cols = []
                for field in self.schema:
                    if field.name in table.column_names:
                        col = table[field.name]
                    else:
                        col = pa.nulls(table.num_rows, type=field.type)
                    cols.append(col)
                table = pa.Table.from_arrays(cols, schema=self.schema)

        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def run_prepare_gate_dir(
    merged_train: str,
    merged_test: str,
    out_dir: str,
    batch_rows: int = 200_000,
    compression: str = "snappy",
):
    """
    Import-friendly entry point.

    INPUT:
      - merged_train: parquet file OR dataset directory (e.g., part-*.parquet)
      - merged_test : parquet file OR dataset directory
    OUTPUT (written under out_dir):
      - gated_train.parquet
      - gated_val.parquet
      - dir_train.parquet
      - dir_val.parquet

    Returns dict of output paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "gated_train": out / "gated_train.parquet",
        "gated_val":   out / "gated_val.parquet",
        "dir_train":   out / "dir_train.parquet",
        "dir_val":     out / "dir_val.parquet",
    }

    writers = {
        "g_tr": ParquetAppender(paths["gated_train"], compression=compression),
        "g_va": ParquetAppender(paths["gated_val"],   compression=compression),
        "d_tr": ParquetAppender(paths["dir_train"],   compression=compression),
        "d_va": ParquetAppender(paths["dir_val"],     compression=compression),
    }

    print("Processing TRAIN (parquet)...")
    for ch in iter_parquet(merged_train, batch_rows=batch_rows, columns=None):
        ensure_cols(ch)
        ch = keep_relevant(ch)
        writers["g_tr"].append(make_gate_chunk(ch))
        writers["d_tr"].append(make_dir_chunk(ch))

    print("Processing VAL (parquet)...")
    for ch in iter_parquet(merged_test, batch_rows=batch_rows, columns=None):
        ensure_cols(ch)
        ch = keep_relevant(ch)
        writers["g_va"].append(make_gate_chunk(ch))
        writers["d_va"].append(make_dir_chunk(ch))

    for w in writers.values():
        w.close()

    # Sanity: ensure files exist + non-empty
    for k, p in paths.items():
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Expected output not written or empty: {p}")

    return {k: str(v) for k, v in paths.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_train", required=True, help="3-class parquet train input (file or dataset dir)")
    ap.add_argument("--merged_test", required=True, help="3-class parquet val/test input (file or dataset dir)")
    ap.add_argument("--out_dir", required=True, help="Output directory (parquet outputs)")
    ap.add_argument("--batch_rows", type=int, default=200_000, help="Rows per Arrow batch")
    ap.add_argument("--compression", default="snappy", choices=["snappy", "zstd", "gzip", "brotli", "lz4_raw"])
    args = ap.parse_args()

    run_prepare_gate_dir(
        merged_train=args.merged_train,
        merged_test=args.merged_test,
        out_dir=args.out_dir,
        batch_rows=args.batch_rows,
        compression=args.compression,
    )




if __name__ == "__main__":
    main()
