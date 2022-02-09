#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# To run reinforcement learning for M1, run this after the main_results

source define_vars.sh

cd ../

exp_name="m1-learned"
device=${2:-0}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

extra=""

if [ "$exp_name" == "none" ]
then
   extra="-no_ask 1"
elif [ "$exp_name" == "first" ]
then
  extra="-ask_first 1"
elif [ "$exp_name" == "random" ]
then
  extra="-random_ask 1"
elif [ "$exp_name" == "teacher" ]
then
  extra="-teacher_ask 1"
elif [ "$exp_name" == "learned" ]
then
  extra=""
else
  echo "Usage: bash train_m1.sh [none|first|random|teacher|learned] [gpu_id]"
  echo "Example: bash train_m1.sh learned 0"
  exit
fi

command="python -u m1_train.py -config $config_file -exp $output_dir $extra -device $device"
echo $command
$command






