#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# upload_artifacts_s3.sh
# - Upload local artifacts to S3
# - Requires awscli configured (aws configure / IAM role)
# ------------------------------------------------------------

log() { echo "[s3] $(date '+%Y-%m-%d %H:%M:%S') | $*"; }

S3_PREFIX="${S3_PREFIX:-s3://your-bucket/ExplainableCreditLimitDecider}"
DRY_RUN="${DRY_RUN:-0}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
REPORTS_DIR="${REPORTS_DIR:-reports}"
LOGS_DIR="${LOGS_DIR:-logs}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: awscli not found. Install with: pip install awscli"
  exit 1
fi

sync_dir() {
  local src="$1"
  local dst="$2"
  if [ ! -d "$src" ]; then
    log "Skip missing dir: $src"
    return
  fi

  log "Sync: $src -> $dst"
  if [ "$DRY_RUN" = "1" ]; then
    aws s3 sync "$src" "$dst" --dryrun
  else
    aws s3 sync "$src" "$dst"
  fi
}

log "S3_PREFIX=${S3_PREFIX}"
log "DRY_RUN=${DRY_RUN}"

sync_dir "$CHECKPOINTS_DIR" "$S3_PREFIX/checkpoints"
sync_dir "$REPORTS_DIR"     "$S3_PREFIX/reports"
sync_dir "$LOGS_DIR"        "$S3_PREFIX/logs"
sync_dir "$ARTIFACTS_DIR"   "$S3_PREFIX/artifacts"

log "Upload complete."
