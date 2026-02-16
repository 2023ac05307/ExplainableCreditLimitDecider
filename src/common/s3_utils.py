from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import re

# boto3 is optional: only required when actually using S3 functions
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover
    boto3 = None
    ClientError = Exception


_S3_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<key>.+)$")


@dataclass
class S3Config:
    region: Optional[str] = None
    profile: Optional[str] = None


def is_s3_uri(uri: str) -> bool:
    return bool(_S3_RE.match(uri))


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    m = _S3_RE.match(uri)
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return m.group("bucket"), m.group("key")


def _client(cfg: Optional[S3Config] = None):
    if boto3 is None:
        raise RuntimeError("boto3 is not installed. Install boto3 to use S3 utilities.")
    cfg = cfg or S3Config()
    if cfg.profile:
        session = boto3.Session(profile_name=cfg.profile, region_name=cfg.region)
        return session.client("s3")
    return boto3.client("s3", region_name=cfg.region)


def s3_download_file(s3_uri: str, local_path: str, cfg: Optional[S3Config] = None) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    p = Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = _client(cfg)
    try:
        c.download_file(bucket, key, str(p))
    except ClientError as e:
        raise RuntimeError(f"Failed to download {s3_uri} -> {local_path}: {e}") from e


def s3_upload_file(local_path: str, s3_uri: str, cfg: Optional[S3Config] = None) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    c = _client(cfg)
    try:
        c.upload_file(str(local_path), bucket, key)
    except ClientError as e:
        raise RuntimeError(f"Failed to upload {local_path} -> {s3_uri}: {e}") from e


def s3_download_bytes(s3_uri: str, cfg: Optional[S3Config] = None) -> bytes:
    bucket, key = parse_s3_uri(s3_uri)
    c = _client(cfg)
    try:
        obj = c.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        raise RuntimeError(f"Failed to download bytes from {s3_uri}: {e}") from e


def s3_upload_bytes(data: bytes, s3_uri: str, cfg: Optional[S3Config] = None, content_type: Optional[str] = None) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    c = _client(cfg)
    kwargs = {"Bucket": bucket, "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    try:
        c.put_object(**kwargs)
    except ClientError as e:
        raise RuntimeError(f"Failed to upload bytes to {s3_uri}: {e}") from e
