#!/bin/sh
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --qos=<your-qos>           # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=10:00:00        # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=8     # Request number of CPUs (threads) per task. Default is 1
#SBATCH --mem=32GB             # Request memory per node. Default is 1024MB (1GB)
#SBATCH --mail-type=END       # Set mail type to 'END' to receive a mail when the job finishes
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err  # Set name of error log. %j is the Slurm jobId

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Some debugging logs
which python3 1>&2
python3 --version

export WANDB_DIR="$BULK/wandb_logs"
mkdir -p $WANDB_DIR

# Run training with srun using Python 3
# srun python3 model_training.py 
srun python3 model_training_cv2.py 
# srun python3 model_training_cv3.py 
# srun python3 model_training_cv4.py 

