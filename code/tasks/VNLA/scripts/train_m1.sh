#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# To run reinforcement learning for M1, run this after the main_results
# Example Usage: bash train_m1.sh [seen|unseen] [question_set=1|2] 0

source define_vars.sh

cd ../

exp_name="m1-learned"
split=${1:-seen}
question_set=${2:-1}
device=${3:-0}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

start_path="$PT_OUTPUT_DIR/main_learned_nav_sample_ask_sample/main_learned_nav_sample_ask_sample_val_${split}.ckpt"

extra=""

if [ $question_set == 1 ]
then
  extra="$extra -advisor verbal_qa"
elif [ $question_set == 2 ]
then
  extra="$extra -advisor verbal_qa2"
else
  echo "Usage: bash train_m1.sh [seen|unseen] [question_set=1|2] [gpu_id]"
  echo "Example: bash train_m1.sh seen 2 0"
  exit
fi

# Smaller-scale training
command="time python -u m1_train.py -config $config_file -exp $output_dir -device $device -start_path $start_path -log_every 100 -save_every 100 -n_iters 15000 -batch_size 50 $extra"
echo $command
$command
