#!/bin/bash

#SBATCH --job-name final_project_fz19792
#SBATCH --partition gpu
#SBATCH --time 1-06:00:00
#SBATCH --mem 32GB
#SBATCH --nodes 1
#SBATCH --account=cosc027924
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge

# load in the module dependencies for this script
module load CUDA
module load languages/anaconda3/2022.11-3.9.13-tensorflow-2.11

python lstm_architecture.py
# python utils.py
