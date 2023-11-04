#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB

#SBATCH --time=01:00:00
#SBATCH --output=trial.out
#SBATCH --error=trial.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:8

module purge 
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
module load git

module load conda

mamba init bash
source ~/.bashrc

mamba activate mtr_test

bash scripts/torchrun_train.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 80 --epochs 30 --extra_tag my_second_exp