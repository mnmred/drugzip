#!/bin/sh
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --qos=<your-qos>           # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=10:00:00        # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=8     # Request number of CPUs (threads) per task. Default is 1
#SBATCH --mem=64GB             # Request memory per node. Default is 1024MB (1GB)
#SBATCH --mail-type=END       # Set mail type to 'END' to receive a mail when the job finishes
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err  # Set name of error log. %j is the Slurm jobId

# Set MODEL_FILE to a CSV name/path relative to CCSynergy (without .csv), e.g. Sanger_Full/A1.
MODEL_FILE="${MODEL_FILE:-drugzip-25in1-full-1.0-b256}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Some debugging logs
which python3 1>&2
python3 --version

export WANDB_DIR="/path/to/your/wandb_logs"
mkdir -p $WANDB_DIR

# Run training with srun using Python 3
srun python3 model2_training.py --model_file "$MODEL_FILE"
srun python3 model2_training_cv2.py --model_file "$MODEL_FILE"
srun python3 model2_training_cv3.py --model_file "$MODEL_FILE"
srun python3 model2_training_cv4.py --model_file "$MODEL_FILE"
