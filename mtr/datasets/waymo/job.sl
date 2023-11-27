#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB

#SBATCH --time=15:00:00
#SBATCH --output=gs.out
#SBATCH --error=gs.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:2

module purge 
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
module load git

module load conda

mamba init bash
source ~/.bashrc

mamba activate mtr_test
# export CUDA_VISIBLE_DEVICES='1'
python data_preprocess.py ../../../../scenario/  ../../../data/waymo_lidar

