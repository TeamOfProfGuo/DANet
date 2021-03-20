#!/bin/bash
#SBATCH --job-name=DDANet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=44GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=email@address # put your email here if you want emails
#SBATCH --output=ddanet_%j.out
#SBATCH --error=ddanet_%j.err
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p aquila

echo "Your NetID is: $1"
echo "Your environment is: $2"

module purge
module load anaconda3
module load cuda/10.0
module load gcc/7.3
cd /gpfsnyu/scratch/$1/DeepLearning/DANet
# update the encoding lib
rm -r /gpfsnyu/home/$1/.conda/envs/dl/lib/python3.6/site-packages/encoding/
cp encoding -r /gpfsnyu/home/$1/.conda/envs/dl/lib/python3.6/site-packages/encoding/

echo "start training"
source activate $2
python experiments/ddanet/train_ddanet.py --config './results/ddanet_resnet50/config.yaml'
echo "FINISH"

