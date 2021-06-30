#!/bin/bash

#SBATCH --job-name=rfd_at2_2cbr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=88GB
#SBATCH --time=50:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=rfd_at2_2cbr.out
#SBATCH --error=rfd_at2_2cbr.out
#SBATCH --gres=gpu:2 # How much gpu need, n is the number
#SBATCH -p aquila
#SBATCH --constraint=2080Ti

module purge
source ~/.bashrc
source activate python36
module load cuda/10.0
#module load cudnn/7.5

python train.py --att_type2 'AT2' --n_cbr 2 >train_at2_2cbr.log 2>& 1
echo "FINISH"

