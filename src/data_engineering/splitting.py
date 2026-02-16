#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pyarrow  # noqa: F401
except Exception as e:
    raise RuntimeError("Parquet support requires pyarrow. Install: pip install pyarrow") from e


def compute_label_counts(df: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    """
    Returns a dataframe indexed by group with columns = labels and values = counts.
    """
    ct = pd.crosstab(df[group_col], df[label_col])
    ct = ct.sort_index()
    return ct


def greedy_group_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Greedy assignment of whole groups (customers) to splits to match:
      - target row counts per split
      - target label proportions per split (same as global)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    rng = np.random.default_rng(seed)

    # Group x label counts
    g_counts = compute_label_counts(df, group_col, label_col)
    labels = g_counts.columns.tolist()

    # Group sizes
    g_sizes = g_counts.sum(axis=1).astype(int)

    # Global totals
    total_rows = int(g_sizes.sum())
    global_label_totals = g_counts.sum(axis=0).astype(float)
    global_label_props = global_label_totals / global_label_totals.sum()

    # Target split sizes (in rows)
    target_sizes = {
        "train": int(round(train_frac * total_rows)),
        "val": int(round(val_frac * total_rows)),
        "test": total_rows - int(round(train_frac * total_rows)) - int(round(val_frac * total_rows)),
    }

    # Target label totals per split (in rows)
    target_label_totals = {
        split: (global_label_props * target_sizes[split]).to_dict()
        for split in ["train", "val", "test"]
    }

    # Sort groups: big + label-diverse first helps stability
    diversity = (g_counts.gt(0).sum(axis=1)).astype(int)
    order = pd.DataFrame({"size": g_sizes, "div": diversity}, index=g_sizes.index)
    order = order.sort_values(["size", "div"], ascending=[False, False]).index.tolist()

    # Deterministic shuffle within ties via noise
    noise = pd.Series(rng.random(len(order)), index=order)
    order = sorted(order, key=lambda g: (-g_sizes[g], -diversity[g], noise[g]))

    # Accumulators
    splits = {k: [] for k in ["train", "val", "test"]}
    split_size = {k: 0 for k in ["train", "val", "test"]}
    split_label = {k: {lab: 0.0 for lab in labels} for k in ["train", "val", "test"]}

    def score_add(split_name: str, group) -> float:
        """
        Lower is better. Measures how far the split would be from its targets if we add this group.
        Combines label deviation + size deviation.
        """
        new_size = split_size[split_name] + g_sizes[group]
        size_dev = abs(new_size - target_sizes[split_name]) / max(1, target_sizes[split_name])

        dev = 0.0
        for lab in labels:
            new_lab = split_label[split_name][lab] + float(g_counts.loc[group, lab])
            dev += abs(new_lab - target_label_totals[split_name][lab]) / max(1.0, target_label_totals[split_name][lab])

        return (2.0 * dev) + (1.0 * size_dev)

    for g in order:
        best_split = None
        best_score = None

        for s in ["train", "val", "test"]:
            if split_size[s] > 1.10 * target_sizes[s]:
                continue
            sc = score_add(s, g)
            if best_score is None or sc < best_score:
                best_score = sc
                best_split = s

        if best_split is None:
            best_split = min(["train", "val", "test"], key=lambda s: score_add(s, g))

        splits[best_split].append(g)
        split_size[best_split] += int(g_sizes[g])
        for lab in labels:
            split_label[best_split][lab] += float(g_counts.loc[g, lab])

    train_df = df[df[group_col].isin(splits["train"])].copy()
    val_df   = df[df[group_col].isin(splits["val"])].copy()
    test_df  = df[df[group_col].isin(splits["test"])].copy()

    return train_df, val_df, test_df


def print_split_stats(df_full, train_df, val_df, test_df, label_col):
    def stats(name, d):
        vc = d[label_col].value_counts(dropna=False)
        pct = (vc / max(1, len(d)) * 100.0).round(2)
        return name, len(d), vc.to_dict(), pct.to_dict()

    full = stats("FULL", df_full)
    tr = stats("TRAIN", train_df)
    va = stats("VAL", val_df)
    te = stats("TEST", test_df)

    print("\n=== Split Summary (rows + class %) ===")
    for name, n, vc, pct in [full, tr, va, te]:
        print(f"\n{name}: {n} rows")
        for lab in sorted(vc.keys(), key=lambda x: str(x)):
            print(f"  {lab}: {vc.get(lab,0)}  ({pct.get(lab,0)}%)")


def read_parquet_any(path: Path) -> pd.DataFrame:
    """
    Accepts:
      - a single .parquet file
      - a directory containing a parquet dataset (e.g. part-*.parquet)
    """
    if path.is_dir():
        return pd.read_parquet(path)  # pandas will load dataset directory
    if path.is_file() and path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"--in_parquet must be a .parquet file or a parquet directory. Got: {path}")

def run_split(
    in_parquet: str,
    out_dir: str,
    group_col: str = "cust_id",
    label_col: str = "action_id",
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    compression: str = "snappy",
):
    """
    Import-friendly entry point for customer-level stratified split.
    Writes:
      - trajectories_train.parquet
      - trajectories_val.parquet
      - trajectories_test.parquet
    Returns (train_path, val_path, test_path)
    """
    in_path = Path(in_parquet)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    df = read_parquet_any(in_path)

    if group_col not in df.columns:
        raise ValueError(f"Missing group_col '{group_col}' in columns={list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label_col '{label_col}' in columns={list(df.columns)}")

    train_df, val_df, test_df = greedy_group_stratified_split(
        df,
        group_col=group_col,
        label_col=label_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    train_path = out_dir_p / "trajectories_train.parquet"
    val_path   = out_dir_p / "trajectories_val.parquet"
    test_path  = out_dir_p / "trajectories_test.parquet"

    train_df.to_parquet(train_path, index=False, engine="pyarrow", compression=compression)
    val_df.to_parquet(val_path, index=False, engine="pyarrow", compression=compression)
    test_df.to_parquet(test_path, index=False, engine="pyarrow", compression=compression)

    # Report + sanity checks (keep behavior same as CLI)
    print_split_stats(df, train_df, val_df, test_df, label_col)

    tr_c = set(train_df[group_col].unique())
    va_c = set(val_df[group_col].unique())
    te_c = set(test_df[group_col].unique())
    assert tr_c.isdisjoint(va_c) and tr_c.isdisjoint(te_c) and va_c.isdisjoint(te_c), "Customer leakage detected!"

    print("\n****Saved (PARQUET):*****")
    print(train_path)
    print(val_path)
    print(test_path)

    return str(train_path), str(val_path), str(test_path)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True, help="Input trajectories parquet file OR parquet dataset directory")
    ap.add_argument("--out_dir", required=True, help="Output directory (parquet outputs)")
    ap.add_argument("--group_col", default="cust_id", help="Customer/group column")
    ap.add_argument("--label_col", default="action_id", help="Class label column (e.g., action_id)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    ap.add_argument("--compression", default="snappy", choices=["snappy", "zstd", "gzip", "brotli", "lz4_raw"])
    args = ap.parse_args()
    run_split(
        in_parquet=args.in_parquet,
        out_dir=args.out_dir,
        group_col=args.group_col,
        label_col=args.label_col,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        compression=args.compression,
    )




if __name__ == "__main__":
    main()
