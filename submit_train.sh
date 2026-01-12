#!/bin/bash
#SBATCH --job-name=trans_dist      # Job name
#SBATCH --partition=gpu            # Submit to GPU partition
#SBATCH --nodes=1                  # Number of nodes (start with 1 node, 2 GPUs for simplicity)
#SBATCH --ntasks-per-node=2        # One task per GPU (2 GPUs available per node )
#SBATCH --gres=gpu:2               # Request 2 GPUs per node
#SBATCH --cpus-per-task=12         # CPU cores per task
#SBATCH --time=04:00:00            # Wall time limit (Max is usually 4 days [cite: 3141])
#SBATCH --output=train_%j.out      # Standard output log [cite: 4840]
#SBATCH --error=train_%j.err       # Standard error log [cite: 4841]

# 1. Load Environment
# We purge system modules to avoid conflicts, then initialize your local conda
module purge [cite: 4844]
source /home/ingenx/miniconda3/etc/profile.d/conda.sh
conda activate transformer_dist

# 2. Debugging Info (Optional but recommended)
echo "Node List: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 3. Set Distributed Variables for PyTorch (DDP)
# These allow the GPUs to talk to each other
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"

# 4. Run the Training Command
# 'torchrun' is the standard way to launch distributed transformers
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./train_distributed.py
