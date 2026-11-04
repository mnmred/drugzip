#!/bin/sh
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --qos=<your-qos>           # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=36:00:00        # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=8     # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=64GB             # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END       # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err  # Set name of error log. %j is the Slurm jobId

# Load Anaconda module
# source activate /path/to/your/ccsynergy_env

# Some debugging logs
which python 1>&2  # Write path to Python binary to standard error
python --version   # Write Python version to standard error

export WANDB_DIR="$BULK/wandb_logs"
mkdir -p $WANDB_DIR

# Run your script with the `srun` command:
srun python script.py
