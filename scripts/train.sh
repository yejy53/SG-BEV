#!/usr/bin/env bash

# 
export CUDA_VISIBLE_DEVICES=4,5,6,7   # Specify GPU IDs to use
NUM_GPUS=4                            # Number of GPUs to use (⚠️ make sure it matches the 'world_size' in config file)
export PYTHONPATH=$PYTHONPATH:`pwd`   # Add current directory to PYTHONPATH

config_path='configs/GP.py'

python dist_train_tensorboard.py --config=${config_path}