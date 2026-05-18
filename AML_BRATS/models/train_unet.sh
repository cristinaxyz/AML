#!/bin/bash

#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB

module load uv
srun uv run python -m AML_BRATS.models.train_unet
