#!/bin/bash
#SBATCH --job-name=DANet_pam_ef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=88GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=100:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=email@address # put your email here if you want emails
#SBATCH --output=danet_sunrgb_origin_%j.out
#SBATCH --error=danet_sunrgb_origin_%j.err
#SBATCH --gres=gpu:4 # How much gpu need, n is the number
#SBATCH -p gpu
#SBATCH --constraint=2080Ti

echo "Your NetID is: $1"
echo "Your environment is: $2"

module purge
module load anaconda3
module load cuda/10.0
module load gcc/7.3
cd /gpfsnyu/scratch/$1/DANet
# update the encoding lib
rm -r /gpfsnyu/home/$1/.conda/envs/dl/lib/python3.6/site-packages/encoding/
cp encoding -r /gpfsnyu/home/$1/.conda/envs/dl/lib/python3.6/site-packages/encoding/

echo "start training"
source activate $2
python sun_experiments/danet/train_danet.py --config './results/danet_resnet50/config.yaml'
echo "FINISH"

