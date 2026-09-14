#!/usr/bin/env bash
# ==============================================================================
# setup.sh
#
# Creates a uv-managed virtual environment for Field-AnnotationPipeline at
# <repository>/.venv. pyproject.toml pins torch/torchvision to the cu126
# wheel index (https://download.pytorch.org/whl/cu126) via [tool.uv.sources],
# so a plain `uv sync` installs that CUDA 12.6 build directly -- no separate
# install step needed. If you're on a host with a different GPU driver /
# CUDA support, update that index in pyproject.toml accordingly.
#
# Usage:
#   bash setup.sh
#   PYTHON_VERSION=3.11 bash setup.sh
# ==============================================================================

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${REPO_DIR}/.venv"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if ! command -v uv >/dev/null 2>&1; then
    log "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    hash -r
    command -v uv >/dev/null 2>&1 || die "uv installation failed"
fi
log "uv version: $(uv --version)"

log "Creating virtual environment: ${VENV_DIR}"
uv venv --python "${PYTHON_VERSION}" --clear "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Installing dependencies (torch/torchvision backend auto-selected per pyproject.toml)"
uv sync

log "Verifying installation"
python - <<'PYTHON'
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("Basic imports passed successfully.")
PYTHON

cat <<EOF

==================================================
Setup completed successfully
==================================================

Environment:
  ${VENV_DIR}

Activate it with:
  source ${VENV_DIR}/bin/activate

Or run commands directly without activating, via:
  uv run field-annotation --help
EOF
