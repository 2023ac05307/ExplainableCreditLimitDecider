from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """
    Atomic write to avoid partial files (safe for checkpoints/outputs).
    """
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: PathLike, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False))


def read_table(path: PathLike, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Read CSV/Parquet based on extension.
    """
    path = str(path)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=columns)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    return df


def write_table(df: pd.DataFrame, path: PathLike, *, index: bool = False) -> None:
    """
    Write CSV/Parquet based on extension (atomic write).
    """
    path = Path(path)
    ensure_dir(path.parent)

    if str(path).lower().endswith(".parquet"):
        # parquet writes are not trivially atomic; write to temp and move
        with tempfile.TemporaryDirectory(dir=str(path.parent)) as td:
            tmp = Path(td) / path.name
            df.to_parquet(tmp, index=index)
            tmp.replace(path)
    elif str(path).lower().endswith(".csv"):
        buf = io.StringIO()
        df.to_csv(buf, index=index)
        atomic_write_text(path, buf.getvalue())
    else:
        raise ValueError(f"Unsupported table format: {path}")


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
