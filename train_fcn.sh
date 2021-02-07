#!/bin/bash

#SBATCH --job-name=fcn
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
module load anaconda3
# use your own env name
module load cuda/10.0
module load gcc/7.3
cd /gpfsnyu/scratch/zz1763/DeepLearning/DANet 
echo "start training"
source activate dl
python train.py --config './results/fcn_resnet50/config.yaml'
echo "FINISH"

