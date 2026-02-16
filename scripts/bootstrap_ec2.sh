#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# bootstrap_ec2.sh
# - Prepare an EC2 instance to run training/inference
# - Works on: Amazon Linux 2, Amazon Linux 2023, Ubuntu
# ------------------------------------------------------------

log() { echo "[bootstrap] $(date '+%Y-%m-%d %H:%M:%S') | $*"; }

OS=""
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS="${ID:-}"
fi

REPO_DIR="${REPO_DIR:-$HOME/ExplainableCreditLimitDecider}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INSTALL_GPU_TOOLS="${INSTALL_GPU_TOOLS:-0}"  # set 1 only if you want to attempt cuda toolkit install
EXTRA_PIP_FLAGS="${EXTRA_PIP_FLAGS:-}"

log "Detected OS=${OS}"
log "REPO_DIR=${REPO_DIR}"
log "VENV_DIR=${VENV_DIR}"
log "INSTALL_GPU_TOOLS=${INSTALL_GPU_TOOLS}"

install_packages_ubuntu() {
  log "Installing packages (Ubuntu/Debian)..."
  sudo apt-get update -y
  sudo apt-get install -y \
    git curl wget unzip jq \
    build-essential \
    python3 python3-venv python3-pip \
    ca-certificates
}

install_packages_amzn() {
  log "Installing packages (Amazon Linux)..."
  sudo yum update -y
  sudo yum install -y \
    git curl wget unzip jq \
    gcc gcc-c++ make \
    python3 python3-pip \
    ca-certificates
}

case "$OS" in
  ubuntu|debian)
    install_packages_ubuntu
    ;;
  amzn)
    install_packages_amzn
    ;;
  *)
    log "OS not recognized. Proceeding without OS-specific package install."
    ;;
esac

# Ensure repo directory exists
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

if [ ! -d ".git" ]; then
  log "Repo folder has no .git. If you want auto-clone, set REPO_URL."
  if [ -n "${REPO_URL:-}" ]; then
    log "Cloning repo from REPO_URL=${REPO_URL}"
    rm -rf "$REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
  fi
fi

# Create venv
log "Creating venv..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

# Activate venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install requirements
if [ -f "requirements.txt" ]; then
  log "Installing requirements.txt..."
  pip install -r requirements.txt $EXTRA_PIP_FLAGS
else
  log "No requirements.txt found. Skipping."
fi

# Optional: AWS CLI
if ! command -v aws >/dev/null 2>&1; then
  log "Installing awscli..."
  pip install awscli $EXTRA_PIP_FLAGS || true
fi

# Optional: GPU helper packages (DO NOT force)
if [ "$INSTALL_GPU_TOOLS" = "1" ]; then
  log "INSTALL_GPU_TOOLS=1 set. Not installing CUDA toolkit automatically (varies by AMI)."
  log "If you use NVIDIA AMI, CUDA drivers are preinstalled; just install torch w/ CUDA."
fi

# Create common folders
mkdir -p checkpoints artifacts logs reports data

log "Bootstrap complete."
log "Next:"
log "  source $VENV_DIR/bin/activate"
log "  ./scripts/run_local_train.sh"
