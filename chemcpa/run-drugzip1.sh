#!/bin/bash
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00                  # 1 day
#SBATCH --array=0-9                            # Array job for 40 experiments (4 base configs × 10 hyperparameter combinations)
# NOTE: This script runs hyperparameter sweeps for finetuning similar to config_sciplex_finetune_num_genes.yaml
# NOTE: Set to --array=0-39 to run all 40 experiments (4 base configs × 10 hparam combinations)
#SBATCH --output=logs/finetune_%A_%a.out
#SBATCH --error=logs/finetune_%A_%a.err
#SBATCH --qos=<your-qos>

# This script runs finetuning hyperparameter sweeps with/without pretrained weights and for two gene sets:
# Base configurations:
#   - Config 0: LINCS genes, with pretrained
#   - Config 1: LINCS genes, from scratch
#   - Config 2: 2000 genes, with pretrained
#   - Config 3: 2000 genes, from scratch
# Each base config runs 10 hyperparameter combinations (matching config_sciplex_finetune_num_genes.yaml)
# Array task structure: task_id = base_config_id * 10 + hparam_id
#   - Tasks 0-9:   LINCS genes, with pretrained, hparam 0-9
#   - Tasks 10-19: LINCS genes, from scratch, hparam 0-9
#   - Tasks 20-29: 2000 genes, with pretrained, hparam 0-9
#   - Tasks 30-39: 2000 genes, from scratch, hparam 0-9

# --- Paths (edit PROJECTDIR if needed) ---
IMAGE=$BULK/containers/updated_chemCPA.sif
PROJECT_ROOT=$BULK/chemCPA-reproduction
SCRIPT=$BULK/chemCPA-reproduction/chemCPA/train_hydra.py
HPARAM_FILE=$PROJECT_ROOT/finetune_hparam_combinations.json
# --- Drug embedding input (edit these two) ---
LATENT_CSV=$BULK/CCSynergy/drugzip-25in1-full-1.0-b256.csv   # 1.2M-row file, same row order as all_drugs_inchikeys.csv
FEATURE_SIZE=128
NICKNAME=drugzip-1.0

# Derived names, so each input file gets its own embedding, logs and W&B project
EMB_TAG=$(basename "$LATENT_CSV" .csv)
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

# Build the drug-embedding parquet if it doesn't exist yet ADDED Oct 2026
if [ ! -f "$EMB_PARQUET" ]; then
    echo "Building drug embeddings from $LATENT_CSV (feature size $FEATURE_SIZE)..."
    TMP_SUFFIX=tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    apptainer exec --bind "$BULK:$BULK" "$IMAGE" bash -c "export PYTHONNOUSERSITE=1 && cd $PROJECT_ROOT && \
      python misc.py \
        --latent-csv $LATENT_CSV \
        --feature-size $FEATURE_SIZE \
        --selected-csv $EMB_DIR/selected_vectors_${TMP_SUFFIX}.csv \
        --parquet-out $EMB_PARQUET.$TMP_SUFFIX" || { echo "Embedding build failed"; exit 1; }
    mv -f "$EMB_PARQUET.$TMP_SUFFIX" "$EMB_PARQUET"
    mv -f "$EMB_DIR/selected_vectors_${TMP_SUFFIX}.csv" "$EMB_DIR/selected_vectors.csv"
fi

# Calculate base config ID and hyperparameter combination ID
BASE_CONFIG_ID=$((SLURM_ARRAY_TASK_ID / NUM_HPARAM_COMBOS))
HPARAM_ID=$((SLURM_ARRAY_TASK_ID % NUM_HPARAM_COMBOS))

# Define base experiments as arrays
CONFIG_NAMES=(
    "finetune_mana_vae"
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
echo "Finetune Hyperparameter Sweep"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Base Config ID: $BASE_CONFIG_ID"
echo "Hyperparameter ID: $HPARAM_ID"
echo "Config: $CONFIG_NAME"
echo "Load pretrained: $LOAD_PRET"
echo "Base run name: $BASE_RUN_NAME"
echo "Run name: $RUN_NAME"
echo "Hyperparameter overrides:"
echo "$HPARAM_OVERRIDES" | tr ' ' '\n' | sed 's/^/  /'
echo "=========================================="

module use /path/to/your/modulefiles || true
# module load cuda/12.1

# Run via srun so SLURM tracks CPU/mem usage
# Activate conda environment and run the training script
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
