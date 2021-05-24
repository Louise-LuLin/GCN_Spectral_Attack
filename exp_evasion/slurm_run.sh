#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

echo "$HOSTNAME"

module load anaconda3
module load cuda-toolkit-10.1
source activate torch180

dataset=$1
loss=$2
lr=$3
ep=$4
reg=$5

cd ..

for i in $(seq 0.05 0.05 0.5);
do
    python -m evasion_attack --dataset $dataset --loss_type ${loss} --att_lr ${lr} --perturb_epochs $ep --ptb_rate $i --reg_weight $reg --gnn_path "./nat_model_saved/${dataset}_gcn.pt" --sanitycheck no;
done
