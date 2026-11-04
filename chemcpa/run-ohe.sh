#!/bin/bash
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 1 day
#SBATCH --array=0-9                      # 1 base config x 10 hyperparameter combinations
#SBATCH --output=logs/finetune_%A_%a.out
#SBATCH --error=logs/finetune_%A_%a.err
#SBATCH --qos=<your-qos>

# Finetuning hyperparameter sweep with one-hot encoded (OHE) drugs.
# Each drug is a 0/1 vector with a single 1 (one column per drug), so no SMILES
# featurization and no row indices into the 1.2M-drug files are needed.

# --- Paths (edit PROJECTDIR if needed) ---
IMAGE=$BULK/containers/updated_chemCPA.sif
PROJECT_ROOT=$BULK/chemCPA-reproduction
SCRIPT=$BULK/chemCPA-reproduction/chemCPA/train_hydra.py
HPARAM_FILE=$PROJECT_ROOT/finetune_hparam_combinations.json

# --- Drug embedding input: one-hot encoding ---
# The OHE rows use the same SMILES index (the drugs chemCPA matches on) as the rdkit parquet
PARQUET_IN=$PROJECT_ROOT/embeddings/rdkit/chemCPA-rdkit_embeddings/sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet
NICKNAME=ohe

EMB_TAG=ohe
EMB_DIR=$PROJECT_ROOT/embeddings/$EMB_TAG
EMB_PARQUET=$EMB_DIR/sciplex_complete_middle_subset_lincs_genes_v2_${EMB_TAG}_embedding.parquet

# Load optional local env config (do not commit secrets)
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Standardized secret source used throughout repo
WANDB_API_KEY="${WANDB_API_KEY:-<your-wandb-api-key>}"  # set it in .env or replace the placeholder

# Number of hyperparameter combinations per base config (matching config_sciplex_finetune_num_genes.yaml)
NUM_HPARAM_COMBOS=10

# Generate hyperparameter combinations if they don't exist
if [ ! -f "$HPARAM_FILE" ]; then
    echo "Generating hyperparameter combinations inside container..."
    apptainer exec --bind "$BULK:$BULK" "$IMAGE" \
      python "$PROJECT_ROOT/generate_finetune_hparam_combinations.py" \
      --num-samples $NUM_HPARAM_COMBOS --seed 42 --output "$HPARAM_FILE"
fi

# Build the OHE drug-embedding parquet if it doesn't exist yet:
# row k = drug k (SMILES index), column k = 1, all other columns = 0
if [ ! -f "$EMB_PARQUET" ]; then
    echo "Building one-hot drug embeddings from the SMILES index of $PARQUET_IN..."
    TMP_SUFFIX=tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    mkdir -p "$EMB_DIR"
    apptainer exec --bind "$BULK:$BULK" "$IMAGE" env PYTHONNOUSERSITE=1 python3 -c "
import os
import sys
import numpy as np
import pandas as pd

parquet_in, parquet_out = sys.argv[1], sys.argv[2]
index = pd.read_parquet(parquet_in).index  # canonical SMILES that chemCPA matches on
if not index.is_unique:
    raise ValueError('Duplicate SMILES in ' + parquet_in + ', the one-hot columns would be ambiguous')

n_drugs = len(index)
ohe = np.eye(n_drugs, dtype=np.float32)
df = pd.DataFrame(ohe, index=index, columns=[f'latent_{i+1}' for i in range(n_drugs)])

os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
df.to_parquet(parquet_out)
print(f'OHE size: {n_drugs} drugs, saved {df.shape} to {parquet_out}')
" "$PARQUET_IN" "$EMB_PARQUET.$TMP_SUFFIX" || { echo "Embedding build failed"; exit 1; }
    mv -f "$EMB_PARQUET.$TMP_SUFFIX" "$EMB_PARQUET"
fi

# Calculate base config ID and hyperparameter combination ID
BASE_CONFIG_ID=$((SLURM_ARRAY_TASK_ID / NUM_HPARAM_COMBOS))
HPARAM_ID=$((SLURM_ARRAY_TASK_ID % NUM_HPARAM_COMBOS))

# Define base experiments as arrays
# finetune_morgan is reused: its embedding.datapath is overridden with the OHE parquet below
CONFIG_NAMES=(
    "finetune_morgan"
)

LOAD_PRETRAINED=(
    "false"
)

BASE_RUN_NAMES=(
    "finetune_${NICKNAME}"
)

# Get the base experiment parameters
CONFIG_NAME=${CONFIG_NAMES[$BASE_CONFIG_ID]}
LOAD_PRET=${LOAD_PRETRAINED[$BASE_CONFIG_ID]}
BASE_RUN_NAME=${BASE_RUN_NAMES[$BASE_CONFIG_ID]}

# Get hyperparameters for this array task
# We'll use Python to extract the hyperparameters from JSON
HPARAM_OVERRIDES=$(apptainer exec --bind "$BULK:$BULK" "$IMAGE" python3 -c "
import json
import sys
with open('$HPARAM_FILE', 'r') as f:
    combos = json.load(f)
idx = int(sys.argv[1])
hparams = combos[idx]
overrides = []
for key, value in hparams.items():
    overrides.append(f'model.hparams.{key}={value}')
print(' '.join(overrides))
" $HPARAM_ID)
# Create unique run name with hyperparameter ID
RUN_NAME="${BASE_RUN_NAME}_hparam${HPARAM_ID}"

# Add timestamp to run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WANDB_NAME="${RUN_NAME}_${TIMESTAMP}"
WANDB_PROJECT="chemCPA-Paper"

# Ensure log dir exists
mkdir -p logs

# Print job information
echo "=========================================="
echo "Finetune Hyperparameter Sweep (OHE drugs)"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Base Config ID: $BASE_CONFIG_ID"
echo "Hyperparameter ID: $HPARAM_ID"
echo "Config: $CONFIG_NAME"
echo "Embedding: $EMB_PARQUET"
echo "Load pretrained: $LOAD_PRET"
echo "Base run name: $BASE_RUN_NAME"
echo "Run name: $RUN_NAME"
echo "Hyperparameter overrides:"
echo "$HPARAM_OVERRIDES" | tr ' ' '\n' | sed 's/^/  /'
echo "=========================================="

module use /path/to/your/modulefiles || true
# module load cuda/12.1

# Run via srun so SLURM tracks CPU/mem usage
srun apptainer exec \
  --nv \
  --bind "$BULK:$BULK" \
  "$IMAGE" \
  bash -c "export PYTHONNOUSERSITE=1 && \
           export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH && \
           cd $PROJECT_ROOT && \
           python $SCRIPT \
             --config-name $CONFIG_NAME \
             model.load_pretrained=$LOAD_PRET \
             model.embedding.datapath=$EMB_PARQUET \
             $HPARAM_OVERRIDES \
             wandb.name=$WANDB_NAME \
             wandb.project=$WANDB_PROJECT \
             hydra.run.dir=logs/$RUN_NAME"
echo ""
echo "=========================================="
echo "Experiment $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="
echo "Results saved to: logs/$RUN_NAME"
echo "Check W&B for metrics: $WANDB_NAME"
echo ""