from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class CustomerStoreConfig:
    features_parquet: str               # parquet containing cust_id + s_* features (latest snapshot)
    cust_id_col: str = "cust_id"
    feature_prefix: str = "s_"


class CustomerStore:
    """
    Simple in-memory store:
      - loads a parquet (file or dataset dir) that contains cust_id and s_* feature columns.
      - provides:
          list_customers()
          get_features(cust_id) -> dict
          get_features_many(cust_ids) -> list[(cust_id, features_dict)]
    """
    def __init__(self, cfg: CustomerStoreConfig):
        self.cfg = cfg
        self.df = self._load_any_parquet(cfg.features_parquet)
        if cfg.cust_id_col not in self.df.columns:
            raise RuntimeError(f"Missing '{cfg.cust_id_col}' in {cfg.features_parquet}")

        # keep only cust_id + s_* columns
        feat_cols = [c for c in self.df.columns if c.startswith(cfg.feature_prefix)]
        keep = [cfg.cust_id_col] + feat_cols
        self.df = self.df[keep].copy()

        # make cust_id string for stable matching
        self.df[cfg.cust_id_col] = self.df[cfg.cust_id_col].astype(str)

        # index for fast lookup
        self.df = self.df.drop_duplicates(subset=[cfg.cust_id_col], keep="last").set_index(cfg.cust_id_col)

    def _load_any_parquet(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")
        return pd.read_parquet(p).replace([np.inf, -np.inf], np.nan)

    def list_customers(self, limit: int = 5000) -> List[str]:
        ids = self.df.index.astype(str).tolist()
        return ids[:limit]

    def get_features(self, cust_id: str) -> Optional[Dict[str, Any]]:
        cid = str(cust_id)
        if cid not in self.df.index:
            return None
        row = self.df.loc[cid]
        # row is Series; convert to python scalars
        feat = {}
        for k, v in row.to_dict().items():
            if pd.isna(v):
                feat[k] = 0.0
            else:
                feat[k] = float(v) if isinstance(v, (int, float, np.number)) else v
        return feat

    def get_features_many(self, cust_ids: List[str]) -> List[Dict[str, Any]]:
        out = []
        for cid in cust_ids:
            feat = self.get_features(cid)
            if feat is not None:
                out.append({"cust_id": str(cid), "features": feat})
        return out
