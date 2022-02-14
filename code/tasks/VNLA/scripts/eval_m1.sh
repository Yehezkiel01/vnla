#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Usage: bash eval_m1.sh [seen|unseen] [gpu_id]
# Example: bash eval_m1.sh seen 0
source define_vars.sh

cd ../

exp_name="m1-learned"
split=$1
device=${2:-0}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

model_name="${output_dir}_nav_sample_ask_sample"

extra="-load_path $PT_OUTPUT_DIR/$model_name/${model_name}_val_${split}.ckpt -multi_seed 1 -success_radius 2"

command="time python -u m1_train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command
