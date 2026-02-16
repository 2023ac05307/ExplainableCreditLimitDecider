"""
common package

Production utilities shared across data, training, inference, and serving.
"""

from .logging_utils import configure_logging, get_logger, LogConfig
from .io_utils import (
    read_table,
    write_table,
    ensure_dir,
    atomic_write_bytes,
    atomic_write_text,
    atomic_write_json,
    sha256_file,
)
from .s3_utils import (
    S3Config,
    parse_s3_uri,
    is_s3_uri,
    s3_download_file,
    s3_upload_file,
    s3_download_bytes,
    s3_upload_bytes,
)
from .schema import (
    PredictionRow,
    PredictionBatch,
    ModelMeta,
    CheckpointMeta,
    DataSchemaError,
)

__all__ = [
    # logging
    "configure_logging",
    "get_logger",
    "LogConfig",
    # io
    "read_table",
    "write_table",
    "ensure_dir",
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "sha256_file",
    # s3
    "S3Config",
    "parse_s3_uri",
    "is_s3_uri",
    "s3_download_file",
    "s3_upload_file",
    "s3_download_bytes",
    "s3_upload_bytes",
    # schema
    "PredictionRow",
    "PredictionBatch",
    "ModelMeta",
    "CheckpointMeta",
    "DataSchemaError",
]
