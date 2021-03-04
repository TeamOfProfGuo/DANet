#!/bin/bash

#SBATCH --job-name=psp_0302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=44GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=90:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3797@nyu.edu # put your email here if you want emails
#SBATCH --output=psp_lf_wod.out
#SBATCH --error=psp_lf_wod.err
#SBATCH --gres=gpu:4 # How much gpu need, n is the number
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
python experiments/train_psp_with_dep.py --config './results/psp_resnet50/config.yaml'
# /gpfsnyu/home/hl3797/.conda/envs/dl/bin/python experiments/train_psp_with_dep.py --config './results/psp_resnet50/config.yaml'
echo "FINISH"

