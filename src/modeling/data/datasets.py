# # from __future__ import annotations

# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import List, Optional, Dict, Any

# # import numpy as np
# # import pandas as pd
# # import torch
# # from torch.utils.data import Dataset


# # # -----------------------------
# # # Column inference helpers
# # # -----------------------------
# # def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
# #     for c in candidates:
# #         if c in df.columns:
# #             return c
# #     return None


# # def infer_columns(df: pd.DataFrame) -> Dict[str, Any]:
# #     """
# #     Infer standard RL columns from a parquet dataframe.

# #     Supports common naming conventions:
# #       states:   s_* or state_* or obs_*
# #       next:     s1_* or next_* or obs1_*
# #       action:   a, action, action_id, action_taken
# #       reward:   r, reward (optional for gate/dir but present in RL traj)
# #       done:     done, terminal (optional)
# #       weight:   w, weight, sample_weight (optional, defaults to 1.0)
# #     """
# #     cols = df.columns

# #     s_cols = [c for c in cols if c.startswith("s_")] or \
# #              [c for c in cols if c.startswith("state_")] or \
# #              [c for c in cols if c.startswith("obs_")]

# #     s1_cols = [c for c in cols if c.startswith("s1_")] or \
# #               [c for c in cols if c.startswith("next_")] or \
# #               [c for c in cols if c.startswith("obs1_")]

# #     if not s1_cols:
# #         s1_cols = s_cols

# #     a_col = _pick_first(df, ["a", "action", "action_id", "action_taken"])
# #     r_col = _pick_first(df, ["r", "reward"])
# #     done_col = _pick_first(df, ["done", "terminal", "is_terminal"])
# #     w_col = _pick_first(df, ["w", "weight", "sample_weight"])

# #     if not s_cols:
# #         raise ValueError("Could not infer state columns. Expected s_* or state_* or obs_*.")
# #     if a_col is None:
# #         raise ValueError("Could not infer action column. Expected a/action/action_id/action_taken.")
# #     # reward is optional in some classification-only parquets, but AWAC expects it.
# #     if r_col is None:
# #         # allow missing reward: set to zeros
# #         r_col = None

# #     return {
# #         "s_cols": s_cols,
# #         "s1_cols": s1_cols,
# #         "a_col": a_col,
# #         "r_col": r_col,
# #         "done_col": done_col,
# #         "w_col": w_col,
# #     }


# # # -----------------------------
# # # Base Parquet Trajectory Dataset
# # # -----------------------------
# # @dataclass
# # class TrajDatasetConfig:
# #     parquet_path: str
# #     max_rows: Optional[int] = None
# #     seed: int = 42
# #     float_dtype: torch.dtype = torch.float32


# # class ParquetTrajDataset(Dataset):
# #     """
# #     Loads a parquet table and yields tuples:
# #         (s, a, r, s1, done, w)

# #     Also exposes numpy arrays expected by trainers:
# #         .s .a .r .s1 .done .w
# #         .mu .sd .state_cols .action_counts
# #     """

# #     def __init__(self, cfg: TrajDatasetConfig):
# #         self.cfg = cfg
# #         p = Path(cfg.parquet_path)
# #         if not p.exists():
# #             raise FileNotFoundError(f"Parquet not found: {p}")

# #         df = pd.read_parquet(p)

# #         # Optional subsample for smoke runs
# #         if cfg.max_rows is not None and len(df) > cfg.max_rows:
# #             df = df.sample(n=cfg.max_rows, random_state=cfg.seed).reset_index(drop=True)

# #         meta = infer_columns(df)

# #         self.state_cols: List[str] = meta["s_cols"]
# #         self.next_cols: List[str] = meta["s1_cols"]
# #         self.a_col: str = meta["a_col"]
# #         self.r_col: Optional[str] = meta["r_col"]
# #         self.done_col: Optional[str] = meta["done_col"]
# #         self.w_col: Optional[str] = meta["w_col"]

# #         # ---- arrays ----
# #         self._s = df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
# #         self._s1 = df[self.next_cols].to_numpy(dtype=np.float32, copy=True)

# #         a_raw = df[self.a_col].to_numpy(copy=True)

# #         # Robust action conversion
# #         if np.issubdtype(a_raw.dtype, np.integer):
# #             self._a = a_raw.astype(np.int64, copy=False)
# #         else:
# #             # try numeric cast, else factorize
# #             try:
# #                 self._a = a_raw.astype(np.int64)
# #             except Exception:
# #                 codes, _ = pd.factorize(a_raw)
# #                 self._a = codes.astype(np.int64)

# #         if self.r_col is None:
# #             self._r = np.zeros(len(df), dtype=np.float32)
# #         else:
# #             self._r = df[self.r_col].to_numpy(dtype=np.float32, copy=True)

# #         if self.done_col is None:
# #             self._done = np.zeros(len(df), dtype=np.float32)
# #         else:
# #             self._done = df[self.done_col].to_numpy(dtype=np.float32, copy=True)

# #         if self.w_col is None:
# #             self._w = np.ones(len(df), dtype=np.float32)
# #         else:
# #             self._w = df[self.w_col].to_numpy(dtype=np.float32, copy=True)

# #         # ---- public views expected by trainers ----
# #         self.s = self._s
# #         self.s1 = self._s1
# #         self.a = self._a
# #         self.r = self._r
# #         self.done = self._done
# #         self.w = self._w

# #         # ---- stats used for checkpoint scaler dict ----
# #         self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
# #         self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
# #         self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)

# #         # ---- action counts (for logging + samplers) ----
# #         # ensure keys exist for 0/1/2 even if not present
# #         bc = np.bincount(self._a, minlength=3)
# #         self.action_counts = {i: int(bc[i]) for i in range(len(bc))}

# #         self._n = len(df)

# #     def __len__(self) -> int:
# #         return self._n

# #     def __getitem__(self, idx: int):
# #         s = torch.tensor(self._s[idx], dtype=self.cfg.float_dtype)
# #         a = torch.tensor(self._a[idx], dtype=torch.long)
# #         r = torch.tensor(self._r[idx], dtype=self.cfg.float_dtype)
# #         s1 = torch.tensor(self._s1[idx], dtype=self.cfg.float_dtype)
# #         done = torch.tensor(self._done[idx], dtype=self.cfg.float_dtype)
# #         w = torch.tensor(self._w[idx], dtype=self.cfg.float_dtype)
# #         return s, a, r, s1, done, w


# # # -----------------------------
# # # Gate/Dir wrappers (trainer-compatible)
# # # -----------------------------
# # def _apply_scaler_numpy(s: np.ndarray, s1: np.ndarray, scaler) -> tuple[np.ndarray, np.ndarray]:
# #     """
# #     Supports:
# #       - sklearn-like scaler with .transform()
# #       - dict {"mu":..., "sd":...}
# #     """
# #     if scaler is None:
# #         return s, s1

# #     if hasattr(scaler, "transform"):
# #         s = scaler.transform(s).astype(np.float32, copy=False)
# #         s1 = scaler.transform(s1).astype(np.float32, copy=False)
# #         return s, s1

# #     if isinstance(scaler, dict) and "mu" in scaler and "sd" in scaler:
# #         mu = np.asarray(scaler["mu"], dtype=np.float32).reshape(1, -1)
# #         sd = np.asarray(scaler["sd"], dtype=np.float32).reshape(1, -1)
# #         sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32, copy=False)
# #         s = ((s - mu) / sd).astype(np.float32, copy=False)
# #         s1 = ((s1 - mu) / sd).astype(np.float32, copy=False)
# #         return s, s1

# #     raise TypeError("Unsupported scaler type. Use sklearn .transform() or dict {'mu','sd'}.")


# # class TrajDatasetGATE(ParquetTrajDataset):
# #     def __init__(self, parquet_path: str, scaler=None, max_rows: Optional[int] = None, seed: int = 42,
# #                  float_dtype: torch.dtype = torch.float32):
# #         super().__init__(TrajDatasetConfig(parquet_path=parquet_path, max_rows=max_rows, seed=seed, float_dtype=float_dtype))
# #         self.scaler = scaler
# #         if self.scaler is not None:
# #             self._s, self._s1 = _apply_scaler_numpy(self._s, self._s1, self.scaler)
# #             self.s, self.s1 = self._s, self._s1
# #             self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
# #             self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
# #             self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)


# # class TrajDatasetDIR(ParquetTrajDataset):
# #     def __init__(self, parquet_path: str, scaler=None, max_rows: Optional[int] = None, seed: int = 42,
# #                  float_dtype: torch.dtype = torch.float32):
# #         super().__init__(TrajDatasetConfig(parquet_path=parquet_path, max_rows=max_rows, seed=seed, float_dtype=float_dtype))
# #         self.scaler = scaler
# #         if self.scaler is not None:
# #             self._s, self._s1 = _apply_scaler_numpy(self._s, self._s1, self.scaler)
# #             self.s, self.s1 = self._s, self._s1
# #             self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
# #             self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
# #             self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)
# # src/modeling/data/datasets.py
# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Optional, Dict, Any, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset


# # -----------------------------
# # Column inference helpers
# # -----------------------------
# def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
#     for c in candidates:
#         if c in df.columns:
#             return c
#     return None


# def infer_columns(df: pd.DataFrame) -> Dict[str, Any]:
#     """
#     Infer standard RL columns from a parquet dataframe.

#     Supports common naming conventions:
#       states:   s_* or state_* or obs_*
#       next:     s1_* or next_* or obs1_* (if missing, uses states)
#       action:   a, action, action_id, action_taken
#       reward:   r, reward (optional, if missing -> zeros)
#       done:     done, terminal, is_terminal (optional, if missing -> zeros)
#       weight:   w, weight, sample_weight (optional, if missing -> ones)

#     NOTE:
#       - We do NOT include y_dir here on purpose to avoid accidentally affecting Gate.
#       - DIR has its own strict label resolution logic in TrajDatasetDIR.
#     """
#     cols = df.columns

#     s_cols = [c for c in cols if c.startswith("s_")] or \
#              [c for c in cols if c.startswith("state_")] or \
#              [c for c in cols if c.startswith("obs_")]

#     s1_cols = [c for c in cols if c.startswith("s1_")] or \
#               [c for c in cols if c.startswith("next_")] or \
#               [c for c in cols if c.startswith("obs1_")]

#     if not s1_cols:
#         s1_cols = s_cols

#     a_col = _pick_first(df, ["a", "action", "action_id", "action_taken"])
#     r_col = _pick_first(df, ["r", "reward"])
#     done_col = _pick_first(df, ["done", "terminal", "is_terminal"])
#     w_col = _pick_first(df, ["w", "weight", "sample_weight"])

#     if not s_cols:
#         raise ValueError("Could not infer state columns. Expected s_* or state_* or obs_*.")
#     if a_col is None:
#         raise ValueError("Could not infer action column. Expected a/action/action_id/action_taken.")

#     return {
#         "s_cols": s_cols,
#         "s1_cols": s1_cols,
#         "a_col": a_col,
#         "r_col": r_col,          # may be None
#         "done_col": done_col,    # may be None
#         "w_col": w_col,          # may be None
#     }


# # -----------------------------
# # Base Parquet Trajectory Dataset
# # -----------------------------
# @dataclass
# class TrajDatasetConfig:
#     parquet_path: str
#     max_rows: Optional[int] = None
#     seed: int = 42
#     float_dtype: torch.dtype = torch.float32


# class ParquetTrajDataset(Dataset):
#     """
#     Loads a parquet table and yields tuples:
#         (s, a, r, s1, done, w)

#     Also exposes numpy arrays expected by trainers:
#         .s .a .r .s1 .done .w
#         .mu .sd .state_cols .next_cols .action_counts
#     """

#     def __init__(self, cfg: TrajDatasetConfig):
#         self.cfg = cfg
#         p = Path(cfg.parquet_path)
#         if not p.exists():
#             raise FileNotFoundError(f"Parquet not found: {p}")

#         df = pd.read_parquet(p)

#         # Optional subsample for smoke runs
#         if cfg.max_rows is not None and len(df) > cfg.max_rows:
#             df = df.sample(n=cfg.max_rows, random_state=cfg.seed).reset_index(drop=True)

#         meta = infer_columns(df)

#         # Deterministic column ordering across train/val (IMPORTANT)
#         self.state_cols: List[str] = sorted(meta["s_cols"])
#         self.next_cols: List[str] = sorted(meta["s1_cols"])
#         self.a_col: str = meta["a_col"]
#         self.r_col: Optional[str] = meta["r_col"]
#         self.done_col: Optional[str] = meta["done_col"]
#         self.w_col: Optional[str] = meta["w_col"]

#         # Robust numeric coercion + NaN/inf handling on features
#         feat_cols = list(dict.fromkeys(self.state_cols + self.next_cols))
#         for c in feat_cols:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#         df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

#         # ---- arrays ----
#         self._s = df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
#         self._s1 = df[self.next_cols].to_numpy(dtype=np.float32, copy=True)

#         # Actions (generic); DIR overrides with strict mapping
#         a_raw = df[self.a_col].to_numpy(copy=True)
#         if np.issubdtype(a_raw.dtype, np.integer):
#             self._a = a_raw.astype(np.int64, copy=False)
#         else:
#             try:
#                 self._a = a_raw.astype(np.int64)
#             except Exception:
#                 codes, _ = pd.factorize(a_raw)
#                 self._a = codes.astype(np.int64)

#         if self.r_col is None:
#             self._r = np.zeros(len(df), dtype=np.float32)
#         else:
#             self._r = pd.to_numeric(df[self.r_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)

#         if self.done_col is None:
#             self._done = np.zeros(len(df), dtype=np.float32)
#         else:
#             self._done = pd.to_numeric(df[self.done_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)

#         if self.w_col is None:
#             self._w = np.ones(len(df), dtype=np.float32)
#         else:
#             self._w = pd.to_numeric(df[self.w_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=True)

#         # ---- public views expected by trainers ----
#         self.s = self._s
#         self.s1 = self._s1
#         self.a = self._a
#         self.r = self._r
#         self.done = self._done
#         self.w = self._w

#         # ---- stats used for checkpoint scaler dict ----
#         self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
#         self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
#         self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)

#         # ---- action counts (for logging + samplers) ----
#         bc = np.bincount(self._a, minlength=3)
#         self.action_counts = {i: int(bc[i]) for i in range(len(bc))}

#         self._n = len(df)

#     def __len__(self) -> int:
#         return self._n

#     def __getitem__(self, idx: int):
#         s = torch.tensor(self._s[idx], dtype=self.cfg.float_dtype)
#         a = torch.tensor(self._a[idx], dtype=torch.long)
#         r = torch.tensor(self._r[idx], dtype=self.cfg.float_dtype)
#         s1 = torch.tensor(self._s1[idx], dtype=self.cfg.float_dtype)
#         done = torch.tensor(self._done[idx], dtype=self.cfg.float_dtype)
#         w = torch.tensor(self._w[idx], dtype=self.cfg.float_dtype)
#         return s, a, r, s1, done, w


# # -----------------------------
# # Scaler helper
# # -----------------------------
# def _apply_scaler_numpy(s: np.ndarray, s1: np.ndarray, scaler) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Supports:
#       - sklearn-like scaler with .transform()
#       - dict {"mu":..., "sd":...}
#     """
#     if scaler is None:
#         return s, s1

#     if hasattr(scaler, "transform"):
#         s = scaler.transform(s).astype(np.float32, copy=False)
#         s1 = scaler.transform(s1).astype(np.float32, copy=False)
#         return s, s1

#     if isinstance(scaler, dict) and "mu" in scaler and "sd" in scaler:
#         mu = np.asarray(scaler["mu"], dtype=np.float32).reshape(1, -1)
#         sd = np.asarray(scaler["sd"], dtype=np.float32).reshape(1, -1)
#         sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32, copy=False)
#         s = ((s - mu) / sd).astype(np.float32, copy=False)
#         s1 = ((s1 - mu) / sd).astype(np.float32, copy=False)
#         return s, s1

#     raise TypeError("Unsupported scaler type. Use sklearn .transform() or dict {'mu','sd'}.")


# # -----------------------------
# # Gate/Dir wrappers (trainer-compatible)
# # -----------------------------
# class TrajDatasetGATE(ParquetTrajDataset):
#     """
#     Gate dataset: binary HOLD vs NONHOLD is handled by trainer/model logic.
#     This class only applies optional scaler.
#     """

#     def __init__(
#         self,
#         parquet_path: str,
#         scaler=None,
#         max_rows: Optional[int] = None,
#         seed: int = 42,
#         float_dtype: torch.dtype = torch.float32,
#     ):
#         super().__init__(TrajDatasetConfig(parquet_path=parquet_path, max_rows=max_rows, seed=seed, float_dtype=float_dtype))
#         self.scaler = scaler
#         if self.scaler is not None:
#             self._s, self._s1 = _apply_scaler_numpy(self._s, self._s1, self.scaler)
#             self.s, self.s1 = self._s, self._s1
#             # NOTE: do not recompute mu/sd here; keep train-fit stats from the dataset that created the scaler.


# class TrajDatasetDIR(Dataset):
#     """
#     Direction dataset (MIDSEM-PARITY, SAFE):

#     Your confirmed RL encoding:
#       0 = HOLD
#       1 = CLD
#       2 = CLI

#     This class guarantees DIR labels are binary:
#       y_dir = 0 (CLD), y_dir = 1 (CLI)

#     It supports both input styles:
#       - RL traj parquet with action_id/action_taken containing {0,1,2}
#       - Prebuilt DIR parquet that already has binary labels {0,1} (optionally in y_dir)
#     """

#     def __init__(
#         self,
#         parquet_path: str,
#         scaler=None,
#         max_rows: Optional[int] = None,
#         seed: int = 42,
#         float_dtype: torch.dtype = torch.float32,
#     ):
#         self.cfg = TrajDatasetConfig(parquet_path=parquet_path, max_rows=max_rows, seed=seed, float_dtype=float_dtype)
#         p = Path(parquet_path)
#         if not p.exists():
#             raise FileNotFoundError(f"Parquet not found: {p}")

#         df = pd.read_parquet(p)

#         # Optional subsample
#         if max_rows is not None and len(df) > max_rows:
#             df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

#         meta = infer_columns(df)

#         # Deterministic column ordering
#         self.state_cols: List[str] = sorted(meta["s_cols"])
#         self.next_cols: List[str] = sorted(meta["s1_cols"])
#         self.r_col: Optional[str] = meta["r_col"]
#         self.done_col: Optional[str] = meta["done_col"]
#         self.w_col: Optional[str] = meta["w_col"]

#         # Clean feature columns
#         feat_cols = list(dict.fromkeys(self.state_cols + self.next_cols))
#         for c in feat_cols:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#         df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

#         # ---- DIR label resolution (SAFE, MIDSEM-PARITY) ----
#         # Prefer y_dir if present; otherwise prefer action_id (0/1/2) for mapping
#         if "y_dir" in df.columns:
#             a = pd.to_numeric(df["y_dir"], errors="coerce").astype("Int64")
#             src = "y_dir"
#         else:
#             if "action_id" in df.columns:
#                 a = pd.to_numeric(df["action_id"], errors="coerce").astype("Int64")
#                 src = "action_id"
#             elif "action_taken" in df.columns:
#                 a = pd.to_numeric(df["action_taken"], errors="coerce").astype("Int64")
#                 src = "action_taken"
#             elif "action" in df.columns:
#                 a = pd.to_numeric(df["action"], errors="coerce").astype("Int64")
#                 src = "action"
#             elif "a" in df.columns:
#                 a = pd.to_numeric(df["a"], errors="coerce").astype("Int64")
#                 src = "a"
#             else:
#                 raise ValueError("DIR requires y_dir or one of action_id/action_taken/action/a")

#         a_nonan = a.dropna()
#         uniq = sorted(a_nonan.unique().tolist())

#         # Handle encodings
#         if set(uniq) == {0, 1, 2}:
#             # RL encoding: 0=HOLD,1=CLD,2=CLI
#             mask = (a == 0)
#             df = df.loc[~mask].reset_index(drop=True)
#             a = a.loc[~mask]
#             a = (a - 1).astype(int)  # 1->0 (CLD), 2->1 (CLI)
#             print(f"[DIR] {src} encoding {{0,1,2}} detected -> filtered HOLD + mapped 1/2 to y_dir 0/1")
#         elif set(uniq) == {1, 2}:
#             a = (a - 1).astype(int)
#             print(f"[DIR] {src} encoding {{1,2}} detected -> mapped to y_dir 0/1 via (a-1)")
#         elif set(uniq).issubset({0, 1}):
#             a = a.astype(int)
#             print(f"[DIR] {src} binary labels {{0,1}} detected -> using as-is (y_dir)")
#         else:
#             raise ValueError(f"Unsupported DIR label values in {src}: {uniq}")

#         # Final guardrails
#         u = set(pd.Series(a).unique().tolist())
#         if not u.issubset({0, 1}):
#             raise ValueError(f"DIR labels must be binary {{0,1}} after mapping, got {sorted(u)} from {src}")

#         # Warn if one class missing (this makes training meaningless)
#         vc = pd.Series(a).value_counts().sort_index()
#         c0 = int(vc.get(0, 0))
#         c1 = int(vc.get(1, 0))
#         print(f"[DIR] label counts: {{0(CL D): {c0}, 1(CLI): {c1}}}")
#         if c0 == 0 or c1 == 0:
#             raise ValueError(
#                 "DIR dataset has only one class after mapping. "
#                 f"Got counts: {{0: {c0}, 1: {c1}}}. "
#                 "Check that your source parquet actually contains both actions (CLD and CLI)."
#             )

#         # ---- arrays ----
#         self._s = df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
#         self._s1 = df[self.next_cols].to_numpy(dtype=np.float32, copy=True)
#         self._a = pd.Series(a).to_numpy(dtype=np.int64, copy=True)

#         if self.r_col is None:
#             self._r = np.zeros(len(df), dtype=np.float32)
#         else:
#             self._r = pd.to_numeric(df[self.r_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)

#         if self.done_col is None:
#             self._done = np.zeros(len(df), dtype=np.float32)
#         else:
#             self._done = pd.to_numeric(df[self.done_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)

#         if self.w_col is None:
#             self._w = np.ones(len(df), dtype=np.float32)
#         else:
#             self._w = pd.to_numeric(df[self.w_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=True)

#         # Apply scaler (train-fit stats should come from ds_tr; val uses same)
#         self.scaler = scaler
#         if self.scaler is not None:
#             self._s, self._s1 = _apply_scaler_numpy(self._s, self._s1, self.scaler)

#         # Public views for trainers
#         self.s = self._s
#         self.s1 = self._s1
#         self.a = self._a
#         self.r = self._r
#         self.done = self._done
#         self.w = self._w

#         # Scaler stats computed on THIS dataset's current s (train will be used to create scaler)
#         self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
#         self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
#         self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)

#         # Binary action counts for DIR
#         bc = np.bincount(self._a, minlength=2)
#         self.action_counts = {0: int(bc[0]), 1: int(bc[1])}

#         self._n = len(df)

#     def __len__(self) -> int:
#         return self._n

#     def __getitem__(self, idx: int):
#         s = torch.tensor(self._s[idx], dtype=self.cfg.float_dtype)
#         a = torch.tensor(self._a[idx], dtype=torch.long)
#         r = torch.tensor(self._r[idx], dtype=self.cfg.float_dtype)
#         s1 = torch.tensor(self._s1[idx], dtype=self.cfg.float_dtype)
#         done = torch.tensor(self._done[idx], dtype=self.cfg.float_dtype)
#         w = torch.tensor(self._w[idx], dtype=self.cfg.float_dtype)
#         return s, a, r, s1, done, w
# src/modeling/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =============================
# Global convention (FINAL)
# =============================
A_HOLD = 0
A_CLI  = 1
A_CLD  = 2


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_columns(df: pd.DataFrame) -> Dict[str, Any]:
    cols = df.columns

    s_cols = [c for c in cols if c.startswith("s_")] or \
             [c for c in cols if c.startswith("state_")] or \
             [c for c in cols if c.startswith("obs_")]
    s1_cols = [c for c in cols if c.startswith("s1_")] or \
              [c for c in cols if c.startswith("next_")] or \
              [c for c in cols if c.startswith("obs1_")]
    if not s1_cols:
        s1_cols = s_cols

    a_col = _pick_first(df, ["a", "action", "action_id", "action_taken"])
    r_col = _pick_first(df, ["r", "reward"])
    done_col = _pick_first(df, ["done", "terminal", "is_terminal"])
    w_col = _pick_first(df, ["w", "weight", "sample_weight"])

    if not s_cols:
        raise ValueError("Could not infer state columns. Expected s_* or state_* or obs_*.")
    if a_col is None:
        raise ValueError("Could not infer action column. Expected a/action/action_id/action_taken.")

    return {"s_cols": s_cols, "s1_cols": s1_cols, "a_col": a_col, "r_col": r_col, "done_col": done_col, "w_col": w_col}


def _clean_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class TrajDatasetConfig:
    parquet_path: str
    max_rows: Optional[int] = None
    seed: int = 42
    float_dtype: torch.dtype = torch.float32


class ParquetTrajDataset(Dataset):
    """Generic RL parquet dataset: (s, a, r, s1, done, w)."""

    def __init__(self, cfg: TrajDatasetConfig):
        self.cfg = cfg
        p = Path(cfg.parquet_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")

        df = pd.read_parquet(p)

        if cfg.max_rows is not None and len(df) > cfg.max_rows:
            df = df.sample(n=cfg.max_rows, random_state=cfg.seed).reset_index(drop=True)

        meta = infer_columns(df)

        # Deterministic feature ordering
        self.state_cols = sorted(meta["s_cols"])
        self.next_cols = sorted(meta["s1_cols"])
        self.a_col = meta["a_col"]
        self.r_col = meta["r_col"]
        self.done_col = meta["done_col"]
        self.w_col = meta["w_col"]

        feat_cols = list(dict.fromkeys(self.state_cols + self.next_cols))
        _clean_numeric(df, feat_cols)

        self._s = df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
        self._s1 = df[self.next_cols].to_numpy(dtype=np.float32, copy=True)

        a_raw = df[self.a_col].to_numpy(copy=True)
        try:
            self._a = a_raw.astype(np.int64)
        except Exception:
            codes, _ = pd.factorize(a_raw)
            self._a = codes.astype(np.int64)

        if self.r_col is None:
            self._r = np.zeros(len(df), dtype=np.float32)
        else:
            self._r = pd.to_numeric(df[self.r_col], errors="coerce").fillna(0.0).to_numpy(np.float32)

        if self.done_col is None:
            self._done = np.zeros(len(df), dtype=np.float32)
        else:
            self._done = pd.to_numeric(df[self.done_col], errors="coerce").fillna(0.0).to_numpy(np.float32)

        if self.w_col is None:
            self._w = np.ones(len(df), dtype=np.float32)
        else:
            self._w = pd.to_numeric(df[self.w_col], errors="coerce").fillna(1.0).to_numpy(np.float32)

        # public views
        self.s, self.s1, self.a, self.r, self.done, self.w = self._s, self._s1, self._a, self._r, self._done, self._w

        self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
        self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
        self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)

        bc = np.bincount(self._a, minlength=3)
        self.action_counts = {i: int(bc[i]) for i in range(len(bc))}

        self._n = len(df)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self._s[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._a[idx], dtype=torch.long),
            torch.tensor(self._r[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._s1[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._done[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._w[idx], dtype=self.cfg.float_dtype),
        )


def _apply_scaler_numpy(s: np.ndarray, s1: np.ndarray, scaler) -> Tuple[np.ndarray, np.ndarray]:
    if scaler is None:
        return s, s1
    if hasattr(scaler, "transform"):
        return scaler.transform(s).astype(np.float32, copy=False), scaler.transform(s1).astype(np.float32, copy=False)
    if isinstance(scaler, dict) and "mu" in scaler and "sd" in scaler:
        mu = np.asarray(scaler["mu"], dtype=np.float32).reshape(1, -1)
        sd = np.asarray(scaler["sd"], dtype=np.float32).reshape(1, -1)
        sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32, copy=False)
        return ((s - mu) / sd).astype(np.float32, copy=False), ((s1 - mu) / sd).astype(np.float32, copy=False)
    raise TypeError("Unsupported scaler type")


class TrajDatasetGATE(ParquetTrajDataset):
    """Gate dataset: keep generic action_id (your pipeline already overwrites it to 0/1)."""

    def __init__(self, parquet_path: str, scaler=None, max_rows: Optional[int] = None, seed: int = 42, float_dtype: torch.dtype = torch.float32):
        super().__init__(TrajDatasetConfig(parquet_path=parquet_path, max_rows=max_rows, seed=seed, float_dtype=float_dtype))
        self.scaler = scaler
        if self.scaler is not None:
            self._s, self._s1 = _apply_scaler_numpy(self._s, self._s1, self.scaler)
            self.s, self.s1 = self._s, self._s1


class TrajDatasetDIR(Dataset):
    """
    DIR dataset aligned to:
      0=HOLD, 1=CLI, 2=CLD  (3-class)

    Produces binary labels:
      y_dir = 0 (CLD), 1 (CLI)
    """

    def __init__(self, parquet_path: str, scaler=None, max_rows: Optional[int] = None, seed: int = 42, float_dtype: torch.dtype = torch.float32):
        self.cfg = TrajDatasetConfig(parquet_path=parquet_path, max_rows=max_rows, seed=seed, float_dtype=float_dtype)
        p = Path(parquet_path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")

        df = pd.read_parquet(p)
        if max_rows is not None and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        meta = infer_columns(df)

        self.state_cols = sorted(meta["s_cols"])
        self.next_cols = sorted(meta["s1_cols"])
        self.r_col = meta["r_col"]
        self.done_col = meta["done_col"]
        self.w_col = meta["w_col"]

        feat_cols = list(dict.fromkeys(self.state_cols + self.next_cols))
        _clean_numeric(df, feat_cols)

        # ---- derive from 3-class source ----
        # Prefer action_id_3cls if present (most robust), else action_id if it contains {0,1,2}
        if "action_id_3cls" in df.columns:
            a3 = pd.to_numeric(df["action_id_3cls"], errors="coerce").fillna(A_HOLD).astype(int)
            src = "action_id_3cls"
        elif "action_id" in df.columns:
            a3 = pd.to_numeric(df["action_id"], errors="coerce").fillna(A_HOLD).astype(int)
            src = "action_id"
        else:
            raise ValueError("DIR requires action_id_3cls or action_id")

        uniq = set(np.unique(a3).tolist())
        if not uniq.issubset({A_HOLD, A_CLI, A_CLD}):
            raise ValueError(f"Bad {src} values: {sorted(uniq)} (expected subset of {{0,1,2}})")

        # Keep only NON-HOLD (CLI/CLD)
        mask = (a3 != A_HOLD)
        df = df.loc[mask].reset_index(drop=True)
        a3 = a3[mask]

        # Binary DIR label: 1 for CLI, 0 for CLD
        y_dir = (a3 == A_CLI).astype(np.int64)

        # Guardrail: must contain both classes
        bc = np.bincount(y_dir, minlength=2)
        print(f"[DIR] {src} -> y_dir counts {{0(CL D): {int(bc[0])}, 1(CLI): {int(bc[1])}}}")
        if bc[0] == 0 or bc[1] == 0:
            raise ValueError("DIR dataset has only one class after mapping; check input parquet content.")

        self._s = df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
        self._s1 = df[self.next_cols].to_numpy(dtype=np.float32, copy=True)
        self._a = y_dir  # binary labels

        if self.r_col is None:
            self._r = np.zeros(len(df), dtype=np.float32)
        else:
            self._r = pd.to_numeric(df[self.r_col], errors="coerce").fillna(0.0).to_numpy(np.float32)

        if self.done_col is None:
            self._done = np.zeros(len(df), dtype=np.float32)
        else:
            self._done = pd.to_numeric(df[self.done_col], errors="coerce").fillna(0.0).to_numpy(np.float32)

        if self.w_col is None:
            self._w = np.ones(len(df), dtype=np.float32)
        else:
            self._w = pd.to_numeric(df[self.w_col], errors="coerce").fillna(1.0).to_numpy(np.float32)

        # Apply scaler if provided
        self.scaler = scaler

        # public views
        self.s, self.s1, self.a, self.r, self.done, self.w = self._s, self._s1, self._a, self._r, self._done, self._w

        self.mu = self._s.mean(axis=0).astype(np.float32, copy=False)
        self.sd = self._s.std(axis=0).astype(np.float32, copy=False)
        self.sd = np.where(self.sd < 1e-8, 1.0, self.sd).astype(np.float32, copy=False)

        bc2 = np.bincount(self._a, minlength=2)
        self.action_counts = {0: int(bc2[0]), 1: int(bc2[1])}

        self._n = len(df)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self._s[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._a[idx], dtype=torch.long),
            torch.tensor(self._r[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._s1[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._done[idx], dtype=self.cfg.float_dtype),
            torch.tensor(self._w[idx], dtype=self.cfg.float_dtype),
        )
