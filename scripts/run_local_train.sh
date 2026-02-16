#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# run_local_train.sh
# - Runs training for gate/dir/magnitude using repo python entrypoints
# - Configure via env vars or inline defaults below
# ------------------------------------------------------------

log() { echo "[train] $(date '+%Y-%m-%d %H:%M:%S') | $*"; }

VENV_DIR="${VENV_DIR:-.venv}"
DEVICE="${DEVICE:-cuda}"              # cuda or cpu
PYTHON_BIN="${PYTHON_BIN:-python}"

TRAIN_GATE="${TRAIN_GATE:-1}"
TRAIN_DIR="${TRAIN_DIR:-1}"
TRAIN_MAG="${TRAIN_MAG:-1}"

# Paths (update to your curated parquet paths)
GATE_TRAIN_PARQUET="${GATE_TRAIN_PARQUET:-data/curated/gate_train.parquet}"
GATE_VAL_PARQUET="${GATE_VAL_PARQUET:-data/curated/gate_val.parquet}"

DIR_TRAIN_PARQUET="${DIR_TRAIN_PARQUET:-data/curated/dir_train.parquet}"
DIR_VAL_PARQUET="${DIR_VAL_PARQUET:-data/curated/dir_val.parquet}"

MAG_CLI_TRAIN_PARQUET="${MAG_CLI_TRAIN_PARQUET:-data/curated/mag_cli_train.parquet}"
MAG_CLI_VAL_PARQUET="${MAG_CLI_VAL_PARQUET:-data/curated/mag_cli_val.parquet}"

MAG_CLD_TRAIN_PARQUET="${MAG_CLD_TRAIN_PARQUET:-data/curated/mag_cld_train.parquet}"
MAG_CLD_VAL_PARQUET="${MAG_CLD_VAL_PARQUET:-data/curated/mag_cld_val.parquet}"

OUT_DIR="${OUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Activate venv if exists
if [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

log "DEVICE=${DEVICE}"
log "TRAIN_GATE=${TRAIN_GATE} TRAIN_DIR=${TRAIN_DIR} TRAIN_MAG=${TRAIN_MAG}"

export DEVICE  # allow python scripts to read it if they want

run_cmd() {
  local name="$1"; shift
  local logfile="$LOG_DIR/${name}_$(date '+%Y%m%d_%H%M%S').log"
  log "Running: $name"
  log "Log: $logfile"
  # tee preserves exit code using PIPESTATUS
  ( "$@" ) 2>&1 | tee "$logfile"
  test "${PIPESTATUS[0]}" -eq 0
}

# ---------------- Gate (AWAC / PPO / IQL) ----------------
if [ "$TRAIN_GATE" = "1" ]; then
  run_cmd "train_gate_awac" \
    "$PYTHON_BIN" -m src.training.train_gate_awac \
      --train-parquet "$GATE_TRAIN_PARQUET" \
      --val-parquet "$GATE_VAL_PARQUET" \
      --out "$OUT_DIR/gate_awac.pt" \
      --device "$DEVICE"
fi

# ---------------- Dir ----------------
if [ "$TRAIN_DIR" = "1" ]; then
  run_cmd "train_dir_awac" \
    "$PYTHON_BIN" -m src.training.train_dir_awac \
      --train-parquet "$DIR_TRAIN_PARQUET" \
      --val-parquet "$DIR_VAL_PARQUET" \
      --out "$OUT_DIR/dir_awac.pt" \
      --device "$DEVICE"
fi

# ---------------- Magnitude CLI/CLD (Beta) ----------------
if [ "$TRAIN_MAG" = "1" ]; then
  run_cmd "train_mag_cli_beta" \
    "$PYTHON_BIN" -m src.training.train_cli_beta \
      --train-parquet "$MAG_CLI_TRAIN_PARQUET" \
      --val-parquet "$MAG_CLI_VAL_PARQUET" \
      --out "$OUT_DIR/mag_cli_beta.pt" \
      --device "$DEVICE"

  run_cmd "train_mag_cld_beta" \
    "$PYTHON_BIN" -m src.training.train_cld_beta \
      --train-parquet "$MAG_CLD_TRAIN_PARQUET" \
      --val-parquet "$MAG_CLD_VAL_PARQUET" \
      --out "$OUT_DIR/mag_cld_beta.pt" \
      --device "$DEVICE"
fi

log "Training complete. Checkpoints in: $OUT_DIR"
