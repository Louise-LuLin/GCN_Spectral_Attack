# SPAC

This is the code for the paper "Graph Structural Attack by Perturbing Spectral Distance" accepted by KDD 2022.

## Requirement

Code is tested in **Python 3.10.10**. Some major requirements are listed below:
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install dgl
pip install networkx
pip install numba
```

## Run the code

To execute the attack, you will first train and save a clean model; then attack a loaded model in both training time (poisoning attack) and test time (evasion attack). 
The following example commands on Cora data are run under the root folder.

### Train a Clean Model
```
python run_train_nat.py --dataset cora --gnn_epochs 200 --lr 0.01
```

By default, the clean model will be saved at **./log/nat_model_saved/cora_GCN.pt**

### Attack the Trained Model

When pairing SPAC with other attacks, the argument **--spac_weight** controls the strength of SPAC term, and **--loss_weight** controls the other attack's task loss term.  

- For evasion attack: --attacker can choice from [PGD, random]
- For poisoning attack: --attacker can choice from [minmax, Meta-Self, Meta-Train, random]

#### Evasion Attack
Run SPAC (spectral attack) alone: 
```
python run_attack_evasion.py --gnn_path ./log/nat_model_saved/cora_GCN.pt --spac_weight 1.0 --loss_weight 0.0 
```

Run SPAC-CE (PGD-CE paired with SPAC):
```
python run_attack_evasion.py --gnn_path ./log/nat_model_saved/cora_GCN.pt --spac_weight 1.0 --loss_weight 1.0
```

#### Poisoning Attack
Run SPAC (spectral attack) alone: 
```
python run_attack_poison.py --gnn_path ./log/nat_model_saved/cora_GCN.pt --spac_weight 1.0 --loss_weight 0.0 
```

Run SPAC-Min (Max-Min paired with SPAC):
```
python run_attack_poison.py --gnn_path ./log/nat_model_saved/cora_GCN.pt --spac_weight 1.0 --loss_weight 1.0 
```

## Cite

Please cite our paper if you find this repo useful for your research or development.

```
@inproceedings{lin2022graph,
  title={Graph structural attack by perturbing spectral distance},
  author={Lin, Lu and Blaser, Ethan and Wang, Hongning},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={989--998},
  year={2022}
}
```
