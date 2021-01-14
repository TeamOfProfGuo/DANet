#!/bin/bash

#SBATCH --job-name=deepLabv2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=email@address # put your email here if you want emails
#SBATCH --output=deeplab_%j.out
#SBATCH --error=deeplab_%j.err
#SBATCH --gres=gpu:1 # How much gpu need, n is the number
#SBATCH -p aquila

module purge
source ~/.bashrc
# use your own env name
source activate python36
module load cuda/10.0
module load gcc/7.3 
echo "start training">>train.log

python train.py --config './results/deeplab_resnet50/config.yaml' >train.log 2>& 1
echo "FINISH"

