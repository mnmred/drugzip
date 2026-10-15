#!/bin/bash
#SBATCH --job-name=drugzip-vae-smoke
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/tudelft.net/staff-umbrella/nbrouwer1/drugzip
#SBATCH --output=logs/drugzip_vae_%j.out
#SBATCH --error=logs/drugzip_vae_%j.err

# Smoke-test DrugZip VAE (drugzip.py) inside ccsynergy.sif (GPU).
# Uses a data subset + few epochs. For a full run, unset DRUGZIP_MAX_SAMPLES /
# raise epochs and bump --qos/--time/--mem/--cpus-per-task.
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
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Smoke-test defaults
DRUGZIP_MAX_SAMPLES="${DRUGZIP_MAX_SAMPLES:-8192}"
DRUGZIP_EPOCHS="${DRUGZIP_EPOCHS:-2}"
DRUGZIP_BATCH_SIZE="${DRUGZIP_BATCH_SIZE:-256}"
DRUGZIP_PATIENCE="${DRUGZIP_PATIENCE:-1}"
DRUGZIP_VERBOSE="${DRUGZIP_VERBOSE:-1}"
DRUGZIP_RUN_NAME="${DRUGZIP_RUN_NAME:-vae-smoke-b${DRUGZIP_BATCH_SIZE}}"

# Apptainer often inherits a bad HOME (e.g. .../containers); wandb then
# stages artifacts there and chmod fails on NFS. Keep HOME under drugzip.
HOME_DIR="${HOME_DIR:-$DRUGZIP_ROOT/.home}"

mkdir -p logs "$WANDB_DIR" "$HOME_DIR" \
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
echo "DRUGZIP_MAX_SAMPLES=$DRUGZIP_MAX_SAMPLES EPOCHS=$DRUGZIP_EPOCHS"
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

srun apptainer exec --nv \
  --bind "$DRUGZIP_ROOT:$DRUGZIP_ROOT" \
  --bind "$DRUGZIP_DATA_DIR:$DRUGZIP_DATA_DIR" \
  "$IMAGE" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate ccsynergy && \
           cd \"$DRUGZIP_ROOT\" && \
           export PYTHONPATH=\"$DRUGZIP_ROOT:\$PYTHONPATH\" && \
           export DRUGZIP_DATA_DIR=\"$DRUGZIP_DATA_DIR\" && \
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
