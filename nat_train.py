import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import argparse
import pickle as pkl

from deeprobust.graph.gnns import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.myutils import calc_acc

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, 
                    help='cuda')
parser.add_argument('--seed', type=int, default=123, 
                    help='Random seed for model.')
parser.add_argument('--data_seed', type=int, default=123, 
                    help='Random seed for data split.')

parser.add_argument('--dataset', type=str, default='cora', 
                    choices=['blogcatalog', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], 
                    help='Dataset.')

parser.add_argument('--gnn_base', type=str, default='gcn', 
                    choices=['gcn', 'gat'], help='base gnn models.')
parser.add_argument('--gnn_epochs', type=int, default=500,
                    help='Number of epochs to train the gnn.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--model_dir', type=str, default='./nat_model_saved/',
                    help='Directory to save the trained model.')
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

print ('  Data seed: ', args.data_seed)
print ('  Dataset: ', args.dataset)
print ('  adj shape: ', adj.shape)
print ('  feature shape: ', features.shape)
print ('  label number: ', labels.max().item()+1)
print ('  split seed: ', seed)
print ('  train|valid|test set: {}|{}|{}'.format(idx_train.shape, idx_val.shape, idx_test.shape))


def main():

    #########################################################
    # Setup gnn model and fit it on clean graph
    if args.gnn_base == 'gcn':
        nat_model = GCN(nfeat=features.shape[1], 
                        nclass=labels.max().item()+1, 
                        nhid=args.hidden,
                        dropout=args.dropout, 
                        weight_decay=args.weight_decay, 
                        lr=args.lr,
                        device=device)
    else:
        print ("implemeting...")

    nat_model = nat_model.to(device) 
    nat_model.fit(features, adj, labels, idx_train, train_iters=args.gnn_epochs, verbose=True)

    print ('==== GCN performance ====')
    output = nat_model.predict(features, adj).cpu()
    loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
    loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)

    print("-- train loss = {:.4f} | ".format(loss_train_clean),
          "train acc = {:.4f} | ".format(acc_train_clean),
          "test loss = {:.4f} | ".format(loss_test_clean),
          "test acc = {:.4f} | ".format(acc_test_clean))

    #########################################################
    # Save the trained model
    if not osp.exists(args.model_dir):
        os.makedirs(args.model_dir)
    path = args.model_dir + '{}_{}_{}.pt'.format(args.dataset, args.gnn_base, args.data_seed)
    torch.save(nat_model.state_dict(), path)
    
if __name__ == '__main__':
    main()

