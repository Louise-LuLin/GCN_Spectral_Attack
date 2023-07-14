# SPAC

This is the code for the paper "Graph Structural Attack by Perturbing Spectral Distance" accepted by KDD 2022.

## Requirement

Code is tested in **Python 3.10.10**. Some major requirements are listed below:
```
$pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$pip install torch_geometric
$pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
$pip install dgl
$pip install networkx
$pip install numba
```

## Run the code

To execute the attack, you will first train and save a clean model; then attack a loaded model in both training time (poisoning attack) and test time (evasion attack). 
The following example commands on Cora data are run under the root folder.

#### Train a Clean Model
<code> python run_train_nat.py --dataset cora --gnn_epochs 200 --lr 0.01 </code>

By default, the clean model will be saved at **./log/nat_model_saved/cora_GCN.pt**

#### Attack the Trained Model

When pairing SPAC with other attacks, taking PGD-CE as an example, the argument **--spac_weight** controls the strength of SPAC term, and **--loss_weight** controls PGD-CE's task loss term.  

For evasion attack: --attacker can choice from [PGD, random]

For poisoning attack: --attacker can choice from [minmax, Meta-Self, Meta-Train, random]

- evasion attack via SPAC alone: 
<code> python run_attack_evasion.py --gnn_path ./log/nat_model_saved/cora_GCN.pt --spac_weight 1.0 --loss_weight 0.0 </code>

- poisoning attack:
python run_attack_poison.py --gnn_path ./log/nat_model_saved/cora_GCN.pt
<code> python -m poison_attack --dataset cora --ptb_rate 0.05 --reg_weight 0.0 --model minmax --model_path ./nat_model_saved/cora_gcn.pt </code>

