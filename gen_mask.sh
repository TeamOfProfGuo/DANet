#!/bin/bash

#SBATCH --job-name=mask_0307
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=44GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=90:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3797@nyu.edu # put your email here if you want emails
#SBATCH --output=mask_ef.out
#SBATCH --error=mask_ef.err
#SBATCH --gres=gpu:1 # How much gpu need, n is the number
#SBATCH -p aquila

cd /gpfsnyu/scratch/hl3797/SS_current/ 

module purge
module load anaconda3
module load cuda/10.0
module load gcc/7.3

# conda info --envs

# use your own env name
# conda deactivate
source activate dl
which python
echo "start training"
python rgbd_model_mask.py
echo "FINISH"

