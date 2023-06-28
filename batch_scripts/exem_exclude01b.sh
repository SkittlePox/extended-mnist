#!/bin/bash
#SBATCH --mem=32G
#SBATCH -t 0:20:00
#SBATCH -J exem_mini_exclude01b
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o out/slurm-exem_mini_exclude01b-%j.out

module load anaconda/2022.05
module load cuda
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate deepcompose

cd ..

python3 generate_dataset.py --dataset_name exclude01b --exclude 01

conda deactivate
