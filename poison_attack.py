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

from deeprobust.graph.global_attack import MinMaxSpectral
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

parser.add_argument('--gnn_path', type=str, required=True,
                    help='Path of saved model.')
parser.add_argument('--gnn_base', type=str, default='gcn', 
                    choices=['gcn', 'gat', 'sgc'], help='base gnn models.')

parser.add_argument('--model', type=str, default='minmax', 
                    choices=['minmax', 'Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], 
                    help='model variant')
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
parser.add_argument('--reduction', type=str, default='sum',  # this is important to take sum for eigenvalue mse
                    help='eigenvalue mse reduction type') 

# train GNN: --patience 100 --gnn_epochs 1000 --lr 0.01 --dropout 0.6
parser.add_argument('--gnn_lr', type=float, default=0.01,  
                    help='Initial learning rate.')
parser.add_argument('--gnn_epochs', type=int, default=500,
                    help='Number of epochs to train the gnn.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200,
                    help='patience for early stopping.')

parser.add_argument('--target_node', type=str, default='train', 
                    choices=['test', 'train', 'both'], help='target nodes to attack')

parser.add_argument('--data_dir', type=str, default='./tmp/',
                    help='Directory to download dataset.')

parser.add_argument('--sanitycheck', type=str, default='no',
                    help='whether store the intermediate results.')

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
print ('  torch seed: ', args.seed)

#########################################################
# Load data for node classification task
print ('==== Dataset ====')
seed = None if args.data_seed == 0 else args.data_seed
if not osp.exists(args.data_dir):
    os.makedirs(args.data_dir)
data = Dataset(root=args.data_dir, name=args.dataset, setting='gcn', seed=seed)
adj, features, labels = data.process(process_adj=False, process_feature=False, device=device)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)

print ('  adj shape: ', adj.shape)
print ('  feature shape: ', features.shape)
print ('  label number: ', labels.max().item()+1)
print ('  split seed: ', seed)
print ('  train|valid|test set: {}|{}|{}'.format(idx_train.shape, idx_val.shape, idx_test.shape))

#########################################################
# Load victim model and test it on clean training nodes
if args.gnn_base == 'gcn':
    victim_model = GCN(nfeat=features.shape[1], 
                       nclass=labels.max().item()+1, 
                       nhid=args.hidden,
                       dropout=args.dropout, 
                       weight_decay=args.weight_decay, 
                       lr=args.gnn_lr,
                       device=device)
else:
    assert AssertionError ("GNN model {} not ready!".format(args.gnn_base))

victim_model = victim_model.to(device)
if not osp.exists(args.gnn_path):
    raise AssertionError ("No model found under {}!".format(args.gnn_path))
victim_model.load_state_dict(torch.load(args.gnn_path))
victim_model.eval()

output = victim_model.predict(features, adj)

loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)
print("-- train loss = {:.4f} | ".format(loss_train_clean),
      "train acc = {:.4f} | ".format(acc_train_clean),
      "test loss = {:.4f} | ".format(loss_test_clean),
      "test acc = {:.4f} | ".format(acc_test_clean))

#########################################################
# Setup attack model
if args.model == 'minmax':
    model = MinMaxSpectral(model=victim_model, 
                           nnodes=adj.shape[0], 
                           loss_type=args.loss_type, 
                           regularization_weight=args.reg_weight,
                           device=device)
elif 'Meta' in args.model: # 'Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'
    if 'Self' in args.model:
        lambda_ = 0
    if 'Train' in args.model:
        lambda_ = 1
    if 'Both' in args.model:
        lambda_ = 0.5

    if 'A' in args.model:
        model = MetaApprox(model=victim_model, 
                           nnodes=adj.shape[0], 
                           attack_structure=True, 
                           attack_features=False, 
                           regularization_weight=args.reg_weight,
                           device=device, 
                           lambda_=lambda_)
    else:
        model = Metattack(model=victim_model, 
                          nnodes=adj.shape[0], 
                          attack_structure=True, 
                          attack_features=False, 
                          regularization_weight=args.reg_weight,
                          device=device, 
                          lambda_=lambda_)
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

    # Setting switch: target/untargeted attack
    if args.target_node == 'test':
        idx_target = idx_test # targeted
    elif args.target_node == 'train':
        idx_target = idx_train # ??
    else:
        idx_target = np.hstack((idx_test, idx_train)).astype(np.int) # untargeted
    
    # Start attack
    if 'Meta' in args.model:
        model.attack(features, nat_adj, labels, idx_train, idx_unlabeled, perturbations, 
                     ll_constraint=False, verbose=True)
    else:
        model.attack(features, nat_adj, labels, idx_target, idx_train, idx_test, 
                     perturbations, att_lr=args.att_lr, epochs=args.perturb_epochs,
                     verbose=True, reduction=args.reduction)

    modified_adj = model.modified_adj
    
    # evaluation
    #########################################################
    # retrain victim model on clean graph
    print ('==== GNN trained on clean graph ====')
    victim_model.initialize()

    if args.gnn_base == 'gcn': 
        victim_model.fit(features, adj, labels, idx_train, idx_val, train_iters=args.gnn_epochs, patience=args.patience, verbose=True)
        output = victim_model.predict(features, adj)
    else:
        assert AssertionError ("GNN model {} not ready!".format(args.gnn_base))

    loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
    loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)
    
    print ('==== clean GNN tested on clean graph ====')
    print("-- train loss = {:.4f} | ".format(loss_train_clean),
        "train acc = {:.4f} | ".format(acc_train_clean),
        "test loss = {:.4f} | ".format(loss_test_clean),
        "test acc = {:.4f} | ".format(acc_test_clean))

    #########################################################
    # test victim model on perturbed graph
    print ('==== clean GNN tested on perturbed graph ====')

    if args.gnn_base == 'gcn': 
        output = victim_model.predict(features, modified_adj)
    else:
        assert AssertionError ("GNN model {} not ready!".format(args.gnn_base))

    loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
    loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)
    
    print("-- train loss = {:.4f} | ".format(loss_train_clean),
        "train acc = {:.4f} | ".format(acc_train_clean),
        "test loss = {:.4f} | ".format(loss_test_clean),
        "test acc = {:.4f} | ".format(acc_test_clean))

    #########################################################
    # retrain victim model on perturbed graph
    print ('==== GNN trained on perturbed graph ====')
    victim_model.initialize()

    if args.gnn_base == 'gcn': 
        victim_model.fit(features, modified_adj, labels, idx_train, idx_val, train_iters=args.gnn_epochs, patience=args.patience, verbose=True)
        output = victim_model.predict(features, modified_adj)
    else:
        assert AssertionError ("GNN model {} not ready!".format(args.gnn_base))

    loss_test, acc_test = calc_acc(output, labels, idx_test)
    loss_train, acc_train = calc_acc(output, labels, idx_train)

    print ('==== poisoned GNN tested on perturbed graph ====')
    print("-- train loss = {:.4f} | ".format(loss_train),
          "train acc = {:.4f} | ".format(acc_train),
          "test loss = {:.4f} | ".format(loss_test),
          "test acc = {:.4f} | ".format(acc_test))
    
    #########################################################
    # test poisoned model on clean graph
    print ('==== poisoned GNN tested on clean graph ====')

    if args.gnn_base == 'gcn': 
        output = victim_model.predict(features, adj)
    else:
        assert AssertionError ("GNN model {} not ready!".format(args.gnn_base))

    loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
    loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)
    
    print("-- train loss = {:.4f} | ".format(loss_train_clean),
        "train acc = {:.4f} | ".format(acc_train_clean),
        "test loss = {:.4f} | ".format(loss_test_clean),
        "test acc = {:.4f} | ".format(acc_test_clean))

    acc.append(acc_test)

    # # if you want to save the modified adj/features, uncomment the code below
    root = './sanitycheck_poison/{}_{}_{}_{}lr_{}epoch_{}rate_{}reg_{}target'.format( 
             args.dataset, args.model, args.loss_type, args.att_lr, args.perturb_epochs, args.ptb_rate, 
             args.reg_weight, args.target_node)

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

print ('Overall acc', acc)


