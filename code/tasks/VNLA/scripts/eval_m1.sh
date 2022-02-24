#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_type = $1
exp_name="m1-learned"
split=$2
device=${3:-0}

if [ "$exp_type" == "none" ]
then
   extra="-no_ask 1"
elif [ "$exp_type" == "random" ]
then
  extra="-random_ask 1"
elif [ "$exp_type" == "teacher" ]
then
  extra="-teacher_ask 1"
elif [ "$exp_type" == "rl" ]
then
  extra=""
else
  echo "Usage: bash eval_m1.sh [rl|teacher|random|none] [seen|unseen] [gpu_id]"
  echo "Example: bash eval_m1.sh rl seen 0"
  exit
fi

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

model_name="${output_dir}_nav_sample_ask_sample"

extra="-load_path $PT_OUTPUT_DIR/$model_name/${model_name}_val_${split}.ckpt -multi_seed 1 -success_radius 2"

command="time python -u m1_train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command
