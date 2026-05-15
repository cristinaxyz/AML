#SBATCH --gpus-per-node=v100:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=8GB

srun .venv/bin/python -m AML_BRATS.models.train_unet