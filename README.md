# GCN_Spectral_Attack

### Requirement
- python >= 3.6
- pytorch >= 1.2.0

For example: python: 3.7, pytorch: 1.5.0+cu101

### Exucution Command
You can train and save a clean model; attack a loaded model in both training time (poisoning attack) and test time (evasion attack). The following commands are run under the root folder.

#### Train a Natural Model
<code> python -m nat_train --dataset cora --gnn_epochs 200 --lr 0.01 --model_dir ./nat_model_saved/ </code>

#### Attack a Trained Model
Using the value of --reg_weight to control whether include the spectral distance term in the attack objective.
For evasion attack: --model can choice from [PGD, random]
For poisoning attack: --model can choice from [minmax, Meta-Self, Meta-Train, random]

- evasion attack: 
<code> python -m attack --dataset cora --ptb_rate 0.05 --reg_weight 0.0 --model PGD --target_node test --model_path ./nat_model_saved/cora_gcn.pt </code>

- poisoning attack:
<code> python -m attack --dataset cora --ptb_rate 0.05 --reg_weight 0.0 --model min-max --target_node train --model_path ./nat_model_saved/cora_gcn.pt </code>

#### Train a Robust Model Adversarially
<code> python -m adv_train --dataset cora --adv_epochs 200 --lr 0.0005 --ptb_rate 0.05 --reg_weight 0.0 --model_dir ./rob_model_saved/ </code>



