#!/bin/bash --login
#$ -cwd
#SBATCH --job-name=grade_test
#SBATCH --out=base_model.out.%J
#SBATCH --err=base_model.err.%J
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --account=hpcuser
#SBATCH --export=ALL
conda activate IBERS_GPU
nvidia-smi
python CNN_test.py

