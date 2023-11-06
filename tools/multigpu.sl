#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2  # Request 2 GPUs per node
#SBATCH --nodes=4  # Request 4 nodes
#SBATCH --ntasks-per-node=1  # Each task uses 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB

#SBATCH --time=01:00:00
#SBATCH --output=trial.out
#SBATCH --error=trial.err

module purge
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
module load git

module load conda

mamba init bash
source ~/.bashrc

mamba activate mtr_test

# Adjust the number of GPUs to use in your command
bash scripts/torchrun_train.sh 2 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 20 --epochs 30 --extra_tag my_second_exp

