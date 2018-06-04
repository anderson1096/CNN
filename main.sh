#!/bin/bash

#SBATCH --job-name=cnn_inference
#SBATCH --output=main.out
#SBATCH --ntasks=1
#SBATCH --nodes=2
#SBATCH  --gres=gpu:1


export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

python main.py