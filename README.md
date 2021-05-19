# GCN_Spectral_Attack

### Requirement
- python >= 3.6
- pytorch >= 1.2.0

For example: python: 3.7, pytorch: 1.5.0+cu101

### Exucution Command
You can train and save a clean model; attack a loaded model in both training time (poisoning attack) and test time (evasion attack). The following commands are run under the root folder.

#### Train a Natural Model
<code> python -m examples.mygraph.train --data_seed 123 --dataset cora --gcn_epochs 1000 --lr 0.01 </code>

#### Attack a Trained Model
- evasion attack:
<code> python -m examples.mygraph.attack --data_seed 123 --dataset cora --ptb_rate 0.05 --reg_weight 0.0 --model PGD --target_node test </code>

- poisoning attack:
<code> python -m examples.mygraph.attack --data_seed 123 --dataset cora --ptb_rate 0.05 --reg_weight 0.0 --model min-max --target_node train </code>



