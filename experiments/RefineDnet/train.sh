#!/bin/bash

#SBATCH --job-name=refined
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=88GB
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=refined_%j.out
#SBATCH --error=refined_%j.out
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p gpu
#SBATCH --constraint=2080Ti

module purge
source ~/.bashrc
source activate python36
echo "start training">>train.log
module load cuda/10.0
#module load cudnn/7.5

python train.py >train.log 2>& 1
echo "FINISH"

