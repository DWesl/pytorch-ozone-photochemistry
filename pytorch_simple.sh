#!/bin/bash

#SBATCH --account=fv3-cpu

#SBATCH --comment="Fit neural net to ozone tendencies against o3mr/t/S"
#SBATCH --job-name="increment-nn"
#SBATCH --mail-type=all
#SBATCH --mail-user=daniel.wesloh@noaa.gov
#SBATCH --output=%x-%A-%a.log
#SBATCH --error=%x-%A-%a.log
#SBATCH --profile=Task

#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

set -euo pipefail

PATH=~/"minimambaforge/bin/:${PATH}"
eval "$(mamba shell hook --shell bash)"

module load cuda/12.9.1

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

date --iso=sec
# If using conda instead of mamba, remove the "mamba run" line from
# the srun command and uncomment the following line after changing it
# to match
# conda activate -p /scratch3/NCEPDEV/global/${USER}/AIML/conda/env

# Would love to set nproc_per_node to SLURM_GPUS_PER_TASK
# Unfortunately, that doesn't appear to be something SLURM sets
command time -v srun --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --mpi=none \
	--gpu-bind=map_gpu:0,1 --gres=gpu:2 \
	mamba run -p /scratch3/NCEPDEV/global/${USER}/AIML/conda/env \
	torchrun --nnodes=${SLURM_NNODES} --node-rank=${SLURM_NODEID} \
	--nproc_per_node=2 \
	pytorch_model.py
