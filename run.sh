#!/bin/sh
# ---------------------------------------------------------------------------
# SLURM launcher.
#
# Everything above `python -m main` is cluster-specific -- the account, the
# partition, the module names, the conda prefix and the data root are all from
# the machine this was developed on. Edit them for yours before submitting.
# ---------------------------------------------------------------------------

#SBATCH --job-name=medcure
#SBATCH -A <your-slurm-account>
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -G 8
#SBATCH --time=48:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=64

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <your-conda-env>

# Root of the CXR corpora; see constants.py
export MEDCURE_DATA_ROOT=/path/to/your/cxr/corpora

python -m main --config_name medcure
