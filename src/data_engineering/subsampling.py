#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loss_equivalent_subsample_train_dir.py
-------------------------------------
Loss-mass–preserving subsampling from an augmented TRAIN
Parquet DIRECTORY (original + CF).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


def load_parquet_dir(dir_path: Path) -> pd.DataFrame:
    files = sorted(dir_path.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {dir_path}")

    print(f"[LOAD] Reading {len(files)} parquet files...")
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    return pd.concat(dfs, ignore_index=True)


def subsample(df, n_keep, label):
    n = len(df)
    if n <= n_keep:
        print(f"[KEEP ALL] {label}: {n}")
        return df.copy(), 1.0

    idx = RNG.choice(n, size=n_keep, replace=False)
    scale = n / n_keep
    print(f"[SUBSAMPLE] {label}: {n} → {n_keep} (scale={scale:.2f})")
    return df.iloc[idx].copy(), scale


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INPUT DIR]  {in_dir}")
    print(f"[OUTPUT DIR] {out_dir}")

    df = load_parquet_dir(in_dir)

    # Required columns check
    for col in ("action_id", "is_cf", "sample_weight"):
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")

    # Partition
    orig = df[df.is_cf == 0]
    cf   = df[df.is_cf == 1]

    orig_hold = orig[orig.action_id == 0]
    orig_cli  = orig[orig.action_id == 1]
    orig_cld  = orig[orig.action_id == 2]

    cf_cli = cf[cf.action_id == 1]
    cf_cld = cf[cf.action_id == 2]

    print("\n[ORIGINAL COUNTS]")
    print(f"HOLD={len(orig_hold)} CLI={len(orig_cli)} CLD={len(orig_cld)}")
    print("[CF COUNTS]")
    print(f"CLI={len(cf_cli)} CLD={len(cf_cld)}")

    # Loss-equivalent subsampling
    orig_hold_s, _ = subsample(orig_hold, args.keep_orig_hold, "ORIG HOLD")
    orig_cli_s,  _ = subsample(orig_cli,  len(orig_cli),      "ORIG CLI")
    orig_cld_s,  _ = subsample(orig_cld,  len(orig_cld),      "ORIG CLD")

    cf_cli_s, cli_scale = subsample(cf_cli, args.keep_cf_cli, "CF CLI")
    cf_cld_s, cld_scale = subsample(cf_cld, args.keep_cf_cld, "CF CLD")

    # Weight rescaling (loss-mass preservation)
    cf_cli_s.loc[:, "sample_weight"] *= cli_scale
    cf_cld_s.loc[:, "sample_weight"] *= cld_scale

    # Safety cap
    cf_cli_s["sample_weight"] = cf_cli_s["sample_weight"].clip(upper=1.0)
    cf_cld_s["sample_weight"] = cf_cld_s["sample_weight"].clip(upper=1.0)

    out_df = pd.concat(
        [orig_hold_s, orig_cli_s, orig_cld_s, cf_cli_s, cf_cld_s],
        ignore_index=True
    )

    print("\n[FINAL COUNTS]")
    print(out_df.action_id.value_counts().sort_index())
    print(f"TOTAL ROWS = {len(out_df)}")

    # Write as directory (single part by default)
    out_file = out_dir / "part-0000.parquet"
    out_df.to_parquet(out_file, index=False)

    print(f"\n[WRITE] {out_file}")
    print("\nDONE .... Loss-equivalent subsampling complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Augmented TRAIN parquet directory")
    ap.add_argument("--output-dir", required=True,
                    help="Output fast-train parquet directory")

    ap.add_argument("--keep-orig-hold", type=int, default=700000)
    ap.add_argument("--keep-cf-cli", type=int, default=350000)
    ap.add_argument("--keep-cf-cld", type=int, default=750000)

    args = ap.parse_args()
    main(args)
