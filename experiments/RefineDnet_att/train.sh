#!/bin/bash

#SBATCH --job-name=RefineD_AG2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=88GB
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=RefineD_AG2_%j.out
#SBATCH --error=RefineD_AG2_%j.out
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p gpu
#SBATCH --constraint=2080Ti

module purge
source ~/.bashrc
source activate python36
echo "start training">>train.log
module load cuda/10.0
#module load cudnn/7.5

python train.py --att_type 'AG2' --with_att >train_Att_AG2.log 2>& 1
echo "FINISH"

