#!/bin/bash
#SBATCH --job-name=drugzip-vae-full
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=120:00:00
#SBATCH --chdir=/tudelft.net/staff-umbrella/nbrouwer1/drugzip
#SBATCH --output=logs/drugzip_vae_%j.out
#SBATCH --error=logs/drugzip_vae_%j.err

# Full DrugZip VAE (drugzip.py) inside ccsynergy.sif (GPU).
# ~1.23M drugs; input matrix alone is ~32GB float64 — keep --mem >= 128G.
# Smoke: DRUGZIP_MAX_SAMPLES=8192 DRUGZIP_EPOCHS=2 DRUGZIP_PATIENCE=1 sbatch run_drugzip.sh
#
# Submit:
#   mkdir -p logs
#   sbatch run_drugzip.sh

set -euo pipefail

DRUGZIP_ROOT="${DRUGZIP_ROOT:-/tudelft.net/staff-umbrella/nbrouwer1/drugzip}"
# SIF lives under ambient PROJECTDIR/containers (umbrella), not inside drugzip/
PROJECTDIR="${PROJECTDIR:-/tudelft.net/staff-umbrella/nbrouwer1}"
IMAGE="${IMAGE:-$PROJECTDIR/containers/ccsynergy.sif}"
PY_SCRIPT="${PY_SCRIPT:-$DRUGZIP_ROOT/drugzip.py}"
DRUGZIP_DATA_DIR="${DRUGZIP_DATA_DIR:-/tudelft.net/staff-bulk/ewi/insy/DBL/mmahmoudi/CCSynergy/Sanger_Full}"
WANDB_DIR="${WANDB_DIR:-$DRUGZIP_ROOT/wandb_logs}"
CACHE_ROOT="${CACHE_ROOT:-$DRUGZIP_ROOT/.cache}"
DRUGZIP_OUT_DIR="${DRUGZIP_OUT_DIR:-$DRUGZIP_ROOT/results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Full-run defaults (match drugzip.py). Empty MAX_SAMPLES => all drugs.
DRUGZIP_MAX_SAMPLES="${DRUGZIP_MAX_SAMPLES:-}"
DRUGZIP_EPOCHS="${DRUGZIP_EPOCHS:-500}"
DRUGZIP_BATCH_SIZE="${DRUGZIP_BATCH_SIZE:-256}"
DRUGZIP_PATIENCE="${DRUGZIP_PATIENCE:-10}"
DRUGZIP_VERBOSE="${DRUGZIP_VERBOSE:-1}"
DRUGZIP_RUN_NAME="${DRUGZIP_RUN_NAME:-vae-full-b${DRUGZIP_BATCH_SIZE}}"

# Apptainer often inherits a bad HOME (e.g. .../containers); wandb then
# stages artifacts there and chmod fails on NFS. Keep HOME under drugzip.
HOME_DIR="${HOME_DIR:-$DRUGZIP_ROOT/.home}"

mkdir -p logs "$WANDB_DIR" "$HOME_DIR" "$DRUGZIP_OUT_DIR" \
  "$CACHE_ROOT/matplotlib" \
  "$CACHE_ROOT/wandb" \
  "$CACHE_ROOT/wandb_cache" \
  "$CACHE_ROOT/wandb_config"

if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${DRUGZIP_ROOT}/.wandb_api_key" ]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${DRUGZIP_ROOT}/.wandb_api_key")"
  export WANDB_API_KEY
fi

echo "DRUGZIP_ROOT=$DRUGZIP_ROOT"
echo "IMAGE=$IMAGE"
echo "PY_SCRIPT=$PY_SCRIPT"
echo "DRUGZIP_DATA_DIR=$DRUGZIP_DATA_DIR"
echo "DRUGZIP_MAX_SAMPLES=${DRUGZIP_MAX_SAMPLES:-<all>} EPOCHS=$DRUGZIP_EPOCHS"
echo "DRUGZIP_OUT_DIR=$DRUGZIP_OUT_DIR"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<none>}"

if [ ! -f "$IMAGE" ]; then
  echo "ERROR: image not found: $IMAGE" >&2
  exit 1
fi
if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: python script not found: $PY_SCRIPT" >&2
  exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY unset and ${DRUGZIP_ROOT}/.wandb_api_key missing." >&2
  exit 1
fi
if [ ! -f "${DRUGZIP_DATA_DIR}/input_matrix.npy" ] && [ ! -f "${DRUGZIP_DATA_DIR}/A1.csv" ]; then
  echo "ERROR: no input data in ${DRUGZIP_DATA_DIR}" >&2
  exit 1
fi

module use /opt/insy/modulefiles || true
# Host NVIDIA driver for apptainer --nv (INSY / DAIC)
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

ALLOW_CPU="${ALLOW_CPU:-0}"

srun apptainer exec --nv \
  --bind "$DRUGZIP_ROOT:$DRUGZIP_ROOT" \
  --bind "$DRUGZIP_DATA_DIR:$DRUGZIP_DATA_DIR" \
  "$IMAGE" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate ccsynergy && \
           cd \"$DRUGZIP_ROOT\" && \
           export ALLOW_CPU=\"$ALLOW_CPU\" && \
           # shellcheck source=/dev/null
           source \"$DRUGZIP_ROOT/scripts/gpu_env.sh\" && \
           export PYTHONPATH=\"$DRUGZIP_ROOT:\$PYTHONPATH\" && \
           export DRUGZIP_DATA_DIR=\"$DRUGZIP_DATA_DIR\" && \
           export DRUGZIP_OUT_DIR=\"$DRUGZIP_OUT_DIR\" && \
           export DRUGZIP_MAX_SAMPLES=\"$DRUGZIP_MAX_SAMPLES\" && \
           export DRUGZIP_EPOCHS=\"$DRUGZIP_EPOCHS\" && \
           export DRUGZIP_BATCH_SIZE=\"$DRUGZIP_BATCH_SIZE\" && \
           export DRUGZIP_PATIENCE=\"$DRUGZIP_PATIENCE\" && \
           export DRUGZIP_VERBOSE=\"$DRUGZIP_VERBOSE\" && \
           export DRUGZIP_RUN_NAME=\"$DRUGZIP_RUN_NAME\" && \
           export HOME=\"$HOME_DIR\" && \
           export TMPDIR=\"/tmp/\${SLURM_JOB_ID:-drugzip}\" && \
           mkdir -p \"\$TMPDIR\" && \
           export WANDB_DIR=\"$WANDB_DIR\" && \
           export WANDB_CACHE_DIR=\"\$TMPDIR/wandb_cache\" && \
           export WANDB_CONFIG_DIR=\"$CACHE_ROOT/wandb_config\" && \
           export WANDB_API_KEY=\"$WANDB_API_KEY\" && \
           export XDG_CACHE_HOME=\"\$TMPDIR/xdg_cache\" && \
           export XDG_DATA_HOME=\"\$TMPDIR/xdg_data\" && \
           mkdir -p \"\$WANDB_CACHE_DIR\" \"\$XDG_CACHE_HOME\" \"\$XDG_DATA_HOME\" && \
           export MPLCONFIGDIR=\"$CACHE_ROOT/matplotlib\" && \
           python \"$PY_SCRIPT\" $EXTRA_ARGS"
