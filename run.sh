#!/bin/bash

#SBATCH --qos=gpushort #priority #short

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
##SBATCH --time=6-10:00:00
#SBATCH --time=0-5:15:00
#SBATCH --job-name=train_model
#SBATCH --gres=gpu:1
#SBATCH --nice=0
#SBATCH --mem=200G
# Don't change anything below this line:
#SBATCH --output=/utrain-%j.log
#SBATCH --error=utrain-%j.err

# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge
module load anaconda/2025 
export PYTHONPATH="${PYTHONPATH}:"


# optional: if you have a Conda env, activate it here:
source activate /p/projects/ou/labs/ai/Nils/condadiff 



srun python -u sample_bridge.py --batch_size=1 -of="test" --diffusion_model='consistency' "$@"

srun python -u main.py --n_epochs=200 --batch_size=4 --diffusion_model='consistency' "$@"

srun python -u sample_consistency.py --batch_size=1 -of="test" --diffusion_model='consistency' "$@"

