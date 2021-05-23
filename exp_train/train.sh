#!/bin/bash

#SBATCH --job-name="train"
#SBATCH --error="train.err"
#SBATCH --output="train.output"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

echo "$HOSTNAME"

module load anaconda3
module load cuda-toolkit-10.1
source activate pythono37

cd ..

python -m nat_train --dataset cora --data_seed 123 --gnn_epochs 1000 --lr 0.005 --model_dir ./nat_model_saved/
python -m nat_train --dataset citeseer --data_seed 123 --gnn_epochs 1000 --lr 0.005 --model_dir ./nat_model_saved/
python -m nat_train --dataset polblogs --data_seed 123 --gnn_epochs 1000 --lr 0.005 --model_dir ./nat_model_saved/
python -m nat_train --dataset blogcatalog --data_seed 123 --gnn_epochs 1000 --lr 0.005 --model_dir ./nat_model_saved/
python -m nat_train --dataset pubmed --data_seed 123 --gnn_epochs 1000 --lr 0.005 --model_dir ./nat_model_saved/