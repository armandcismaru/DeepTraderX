#!/bin/bash

#SBATCH --job-name coursework_fz19792
#SBATCH --partition teach_gpu
#SBATCH --time 2:00:00
#SBATCH --mem 32GB
#SBATCH --nodes 1
#SBATCH --account=COSC028844
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge

# load in the module dependencies for this script
module load CUDA
module load languages/anaconda3/2022.11-3.9.13-tensorflow-2.11

# python lstm_architecture.py
python test_gpu.py
