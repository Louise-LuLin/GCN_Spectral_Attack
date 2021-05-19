import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import argparse
import pickle as pkl

from deeprobust.graph.gnns import GCN
from deeprobust.graph.global_attack import SpectralAttack, MinMaxSpectral
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from myutils import calc_acc, save_all

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, 
                    help='cuda')
parser.add_argument('--seed', type=int, default=123, 
                    help='Random seed for model.')
parser.add_argument('--data_seed', type=int, default=123, 
                    help='Random seed for data split.')

parser.add_argument('--dataset', type=str, default='cora', 
                    choices=['blogcatalog', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], 
                    help='dataset')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path of saved model.')

parser.add_argument('--perturb_epochs', type=int, default=100,
                    help='Number of epochs to poisoning loop.')
parser.add_argument('--ptb_rate', type=float, default=0.05,  
                    help='pertubation rate')
parser.add_argument('--reg_weight', type=float, default=0.0,  
                    help='regularization weight')

parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--loss_type', type=str, default='CE', 
                    choices=['CE', 'CW'], help='loss type')
parser.add_argument('--model', type=str, default='PGD', 
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--target_node', type=str, default='test', 
                    choices=['test', 'train', 'both'], help='target nodes to attack')

parser.add_argument('--data_dir', type=str, default='./tmp/',
                    help='Directory to download dataset.')

args = parser.parse_args()

#########################################################
# Setting environment
print ('==== Environment ====')
device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1) # limit cpu use
print ('  pytorch version: ', torch.__version__)
print ('  device: ', device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

#########################################################
# Load data for node classification task
print ('==== Dataset ====')
seed = None if args.data_seed == 0 else args.data_seed
if not osp.exists(args.data_dir):
    os.makedirs(args.data_dir)
data = Dataset(root=args.data_dir, name=args.dataset, setting='gcn', seed=seed)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

print ('  adj shape: ', adj.shape)
print ('  feature shape: ', features.shape)
print ('  label number: ', labels.max().item()+1)
print ('  split seed: ', seed)
print ('  train|valid|test set: {}|{}|{}'.format(idx_train.shape, idx_val.shape, idx_test.shape))

#########################################################
# Load victim model and test it on clean training nodes
victim_model = GCN(nfeat=features.shape[1], 
                   nclass=labels.max().item()+1, 
                   nhid=args.hidden,
                   dropout=args.dropout, 
                   weight_decay=args.weight_decay, 
                   lr=args.lr,
                   device=device)

victim_model = victim_model.to(device)
if not osp.exists(args.model_path):
    raise AssertionError ("No model found under {}!".format(args.model_path))
victim_model.load_state_dict(torch.load(args.model_path))
victim_model.eval()

print ('==== GCN natural performance ====')
output = victim_model.predict(features, adj).cpu()
loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)
print("-- train loss = {:.4f} | ".format(loss_train_clean),
      "train acc = {:.4f} | ".format(acc_train_clean),
      "test loss = {:.4f} | ".format(loss_test_clean),
      "test acc = {:.4f} | ".format(acc_test_clean))

#########################################################
# Setup attack model
if args.model == 'PGD':
    model = SpectralAttack(model=victim_model, 
                        nnodes=adj.shape[0], 
                        loss_type=args.loss_type, 
                        regularization_weight=args.reg_weight,
                        device=device)
else:
    model = MinMaxSpectral(model=victim_model, 
                        nnodes=adj.shape[0], 
                        loss_type=args.loss_type, 
                        regularization_weight=args.reg_weight,
                        device=device)

model = model.to(device)

#########################################################
# Attack and evaluate
print ('==== Attacking ====')
perturbations = int(args.ptb_rate * (adj.sum()/2))
def main():

    # Setting swithc: target/untargeted attack
    if args.target_node == 'test':
        idx_target = idx_test # targeted
    elif args.target_node == 'train':
        idx_target = idx_train # ??
    else:
        idx_target = np.hstack((idx_test, idx_train)).astype(np.int) # untargeted
    
    # Start attack
    model.attack(features, adj, labels, idx_target, perturbations, epochs=args.perturb_epochs)
    modified_adj = model.modified_adj
    
    loss = model.loss
    regularization = 0
    norm = 0
    if args.reg_weight != 0:
        regularization = model.regularization
        norm = model.norm
    print ('-- loss = {:.4f} | regu = {:.4f} | norm = {:.4f}'.format(loss, regularization, norm))

    # evaluation
    print ('==== Victim GCN performance on perturbed graph ====')
    modified_adj = model.modified_adj
    output = victim_model.predict(features, modified_adj).cpu()
    loss_test, acc_test = calc_acc(output, labels, idx_test)
    loss_train, acc_train = calc_acc(output, labels, idx_train)

    print ('==== GCN attack performance ====')
    print("-- train loss = {:.4f}->{:.4f} | ".format(loss_train_clean, loss_train),
          "train acc = {:.4f}->{:.4f} | ".format(acc_train_clean, acc_train),
          "test loss = {:.4f}->{:.4f} | ".format(loss_test_clean, loss_test),
          "test acc = {:.4f}->{:.4f} | ".format(acc_test_clean, acc_test))

    # # if you want to save the modified adj/features, uncomment the code below
    root = './sanitycheck_evasion/{}dseed_{}_{}_{}rate_{}reg_{}_{}target'.format( 
             args.data_seed,args.dataset, args.loss_type, args.ptb_rate, 
             args.reg_weight, args.model, args.target_node)

    # uncomment the the following line to store intermediate results
    # save_all(root, model)

    print ("==== Parameter ====")
    print ('  Data seed: ', args.data_seed)
    print ('  Dataset: ', args.dataset)
    print ('  Loss type: ', args.loss_type)
    print ('  Perturbation Rate: ', args.ptb_rate)
    print ('  Reg weight: ', args.reg_weight)
    print ('  Attack: ', args.model)
    print ('  Target: ', args.target_node)

if __name__ == '__main__':
    main()

