#!/bin/bash
# Small end-to-end example for DrugZip (drugzip.py) and the AE baseline (ae.py)
# on 200 molecules of the Chemical Checker signatures in ./data.
#
# Usage, from the repository root:
#   bash examples/drugzip/run_example.sh
#   PYTHON="apptainer exec $PWD/synergy/synergy.sif python" bash examples/drugzip/run_example.sh
#
# PYTHON is the command used to run the scripts (default: python). W&B logging is
# disabled unless you set WANDB_MODE yourself (e.g. WANDB_MODE=online).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/output"
PYTHON="${PYTHON:-python}"

# Work on a copy of the data, so the cached input_matrix.npy stays out of ./data
mkdir -p "$OUT/data"
cp "$HERE"/data/*.csv "$OUT/data/"
rm -f "$OUT/data/input_matrix.npy"

export DRUGZIP_DATA_DIR="$OUT/data/"
export WANDB_MODE="${WANDB_MODE:-disabled}"

cd "$OUT"

echo "=== 1/2  DrugZip (VAE): drugzip.py ==="
$PYTHON "$REPO/drugzip.py"

echo "=== 2/2  Autoencoder baseline: ae.py ==="
$PYTHON "$REPO/ae.py"

echo
echo "Done. Outputs are in $OUT:"
ls -1 "$OUT" | grep -v '^data$'
