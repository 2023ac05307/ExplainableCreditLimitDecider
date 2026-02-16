#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# run_local_infer.sh
# - Runs next-month prediction pipeline locally
# ------------------------------------------------------------

log() { echo "[infer] $(date '+%Y-%m-%d %H:%M:%S') | $*"; }

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

INPUT_PARQUET="${INPUT_PARQUET:-data/curated/infer_input.parquet}"
OUT_PATH="${OUT_PATH:-reports/predictions_next_month.parquet}"

CKPT_GATE="${CKPT_GATE:-checkpoints/gate_awac.pt}"
CKPT_DIR="${CKPT_DIR:-checkpoints/dir_awac.pt}"
CKPT_MAG_CLI="${CKPT_MAG_CLI:-checkpoints/mag_cli_beta.pt}"
CKPT_MAG_CLD="${CKPT_MAG_CLD:-checkpoints/mag_cld_beta.pt}"

mkdir -p "$(dirname "$OUT_PATH")"

if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

export DEVICE

log "INPUT_PARQUET=${INPUT_PARQUET}"
log "OUT_PATH=${OUT_PATH}"
log "Using ckpts:"
log "  GATE=${CKPT_GATE}"
log "  DIR=${CKPT_DIR}"
log "  MAG_CLI=${CKPT_MAG_CLI}"
log "  MAG_CLD=${CKPT_MAG_CLD}"

# Assumes your repo has a unified inference entrypoint
# If your real entrypoint is different, tell me the module path and args.
"$PYTHON_BIN" -m src.inference.predict_next_month_action_limit \
  --input-parquet "$INPUT_PARQUET" \
  --out "$OUT_PATH" \
  --ckpt-gate "$CKPT_GATE" \
  --ckpt-dir "$CKPT_DIR" \
  --ckpt-mag-cli "$CKPT_MAG_CLI" \
  --ckpt-mag-cld "$CKPT_MAG_CLD" \
  --device "$DEVICE"

log "Inference complete: $OUT_PATH"
