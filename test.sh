#!/bin/bash

#SBATCH --job-name=deepLabv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15000
#SBATCH --gres=gpu:aquila
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --output=deeplab_test_%j.out
#SBATCH --error=test_error_%j.out
#SBATCH --gres=gpu:1 # How much gpu need, n is the number
#SBATCH -p aquila

module purge
source ~/.bashrc
source activate python36
echo "start testing">>test.log
module load cuda/10.0
module load gcc/7.3
#module load cudnn/7.5

python test.py >test.log 2>& 1
echo "FINISH"

