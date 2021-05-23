import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import argparse
import pickle as pkl
import copy

from deeprobust.graph.gnns import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset

from deeprobust.graph.global_attack import SpectralAttack, MinMaxSpectral
from deeprobust.graph.global_attack import IGAttack
from deeprobust.graph.global_attack import MetaApprox, Metattack

from deeprobust.graph.myutils import calc_acc, save_all

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=-1, 
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

parser.add_argument('--loss_type', type=str, default='CE', 
                    choices=['CE', 'CW'], help='loss type')
parser.add_argument('--att_lr', type=float, default=200,
                    help='Initial learning rate.')
parser.add_argument('--perturb_epochs', type=int, default=100,
                    help='Number of epochs to poisoning loop.')

parser.add_argument('--ptb_rate', type=float, default=0.05,  
                    help='pertubation rate')
parser.add_argument('--reg_weight', type=float, default=0.0,  
                    help='regularization weight')
parser.add_argument('--reduction', type=str, default='mean',  
                    help='eigenvalue mse reduction type')                   

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--model', type=str, default='PGD', 
                    choices=['PGD', 'min-max'], 
                    help='model variant')
parser.add_argument('--target_node', type=str, default='test', 
                    choices=['test', 'train', 'both'], help='target nodes to attack')

parser.add_argument('--data_dir', type=str, default='./tmp/',
                    help='Directory to download dataset.')

parser.add_argument('--sanitycheck', type=str, default='no',
                    help='whether store the intermediate results.')

parser

args = parser.parse_args()

#########################################################
# Setting environment
print ('==== Environment ====')
if torch.cuda.is_available():
    if args.device == -1:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')
# device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
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
idx_unlabeled = np.union1d(idx_val, idx_test)

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
elif args.model == 'min-max':
    model = MinMaxSpectral(model=victim_model, 
                        nnodes=adj.shape[0], 
                        loss_type=args.loss_type, 
                        regularization_weight=args.reg_weight,
                        device=device)
else:
    raise AssertionError ("Attack {} not found!".format(args.model))

model = model.to(device)

#########################################################
# Attack and evaluate
print ('==== Attacking ====')
perturbations = int(args.ptb_rate * (adj.sum()/2))
nat_adj = copy.deepcopy(adj)

# seeds = [123, 234, 345, 567, 678]
seeds = [123]
acc = []
for s in seeds:
    print ('***************** seed {} *****************'.format(s))

    np.random.seed(s)
    torch.manual_seed(s)
    if device != 'cpu':
        torch.cuda.manual_seed(s)

    # Setting swithc: target/untargeted attack
    if args.target_node == 'test':
        idx_target = idx_test # targeted
    elif args.target_node == 'train':
        idx_target = idx_train # ??
    else:
        idx_target = np.hstack((idx_test, idx_train)).astype(np.int) # untargeted
    
    # Start attack
    model.attack(features, nat_adj, labels, idx_target, idx_train, idx_test, 
                 perturbations, att_lr=args.att_lr, epochs=args.perturb_epochs, 
                 verbose=True, reduction=args.reduction)

    modified_adj = model.modified_adj
    
    # evaluation
    print ('==== Victim GCN performance on perturbed graph ====')
    modified_adj = model.modified_adj

    victim_model.load_state_dict(torch.load(args.model_path)) # reset to clean model
    victim_model.eval()
    output = victim_model.predict(features, modified_adj).cpu()
    loss_test, acc_test = calc_acc(output, labels, idx_test)
    loss_train, acc_train = calc_acc(output, labels, idx_train)

    print ('==== GCN attack performance ====')
    print("-- train loss = {:.4f}->{:.4f} | ".format(loss_train_clean, loss_train),
          "train acc = {:.4f}->{:.4f} | ".format(acc_train_clean, acc_train),
          "test loss = {:.4f}->{:.4f} | ".format(loss_test_clean, loss_test),
          "test acc = {:.4f}->{:.4f} | ".format(acc_test_clean, acc_test))
    
    acc.append(acc_test)

    # # if you want to save the modified adj/features, uncomment the code below
    root = './sanitycheck_evasion/{}_{}_{}lr_{}epoch_{}rate_{}reg_{}_{}target'.format( 
             args.dataset, args.loss_type, args.att_lr, args.perturb_epochs, args.ptb_rate, 
             args.reg_weight, args.model, args.target_node)

    # uncomment the the following line to store intermediate results
    if args.sanitycheck == 'yes':
        save_all(root, model)

    print ("==== Parameter ====")
    print ('  Data seed: ', args.data_seed)
    print ('  Dataset: ', args.dataset)
    print ('  Loss type: ', args.loss_type)
    print ('  Perturbation Rate: ', args.ptb_rate)
    print ('  Reg weight: ', args.reg_weight)
    print ('  Attack: ', args.model)
    print ('  Target: ', args.target_node)
    print ('  attack seed: ', s)

print ('Overall acc', acc)
