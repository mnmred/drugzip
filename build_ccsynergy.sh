#!/bin/bash
#SBATCH --job-name=build-ccsynergy
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/tudelft.net/staff-umbrella/nbrouwer1/drugzip
#SBATCH --output=logs/build_ccsynergy_%j.out
#SBATCH --error=logs/build_ccsynergy_%j.err

# Rebuild ccsynergy.sif with tensorflow[and-cuda], install to $PROJECTDIR/containers/.
#
#   mkdir -p logs
#   sbatch build_ccsynergy.sh
# Or on a login/build node:
#   bash build_ccsynergy.sh

set -euo pipefail

DRUGZIP_ROOT="${DRUGZIP_ROOT:-/tudelft.net/staff-umbrella/nbrouwer1/drugzip}"
PROJECTDIR="${PROJECTDIR:-/tudelft.net/staff-umbrella/nbrouwer1}"
DEF="${DRUGZIP_ROOT}/containers/ccsynergy.def"
OUT_LOCAL="${DRUGZIP_ROOT}/containers/ccsynergy.sif"
OUT_DEPLOY="${PROJECTDIR}/containers/ccsynergy.sif"

mkdir -p logs "${DRUGZIP_ROOT}/containers" "${PROJECTDIR}/containers"
cd "$DRUGZIP_ROOT"

module use /opt/insy/modulefiles || true
command -v apptainer >/dev/null || module load apptainer 2>/dev/null || true

echo "Building ${OUT_LOCAL} from ${DEF}"
apptainer build --force "$OUT_LOCAL" "$DEF"

echo "Deploying to ${OUT_DEPLOY}"
cp -f "$OUT_LOCAL" "$OUT_DEPLOY"
ls -lh "$OUT_DEPLOY"
echo "Done. Resubmit run_ae.sh / run_drugzip.sh"
