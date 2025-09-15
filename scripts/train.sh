#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='/home/ly/code/SG_BEV/configs/GP.py'

python dist_train_tensorboard.py --config=${config_path}