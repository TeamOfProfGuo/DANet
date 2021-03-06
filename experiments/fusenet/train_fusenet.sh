#!/bin/bash
#SBATCH --job-name=FuseNet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22GB
#SBATCH --gres=gpu:aquila
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=email@address # put your email here if you want emails
#SBATCH --output=fusenet_%j.out
#SBATCH --error=fusenet_%j.err
#SBATCH --gres=gpu:1 # How much gpu need, n is the number
#SBATCH -p aquila

echo "Your NetID is: $1"
echo "Your environment is $2"

# update the encoding lib

rm -r /gpfsnyu/home/$1/.conda/envs/dl/lib/python3.6/site-packages/encoding/
cp encoding -r /gpfsnyu/home/$1/.conda/envs/dl/lib/python3.6/site-packages/encoding/

module purge
module load anaconda3
# use your own env name
module load cuda/10.0
module load gcc/7.3
cd /gpfsnyu/scratch/zz1763/DeepLearning/DANet
echo "start training"
source activate $2
python experiments/fusenet/train_fusenet.py --config './results/fusenet_vgg/config.yaml'
echo "FINISH"

