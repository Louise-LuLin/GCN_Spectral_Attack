#!/bin/bash

dataset=$1
loss=$2
lr=$3
ep=$4
reg=$5

sbatch --job-name=$dataset$loss$reg --output=./result/${dataset}_${loss}_reg${reg}.out --error=./result/${dataset}_${loss}_reg${reg}.err slurm_run.sh $dataset $loss $lr $ep $reg
