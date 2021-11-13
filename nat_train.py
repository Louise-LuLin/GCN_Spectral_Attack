import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import argparse
import pickle as pkl
import networkx as nx

from deeprobust.graph.gnns import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.myutils import calc_acc


def check_victim_model_performance(victim_model, features, adj, labels, idx_test, idx_train):
    output = victim_model.predict(features, adj)
    loss_test_clean, acc_test_clean = calc_acc(output, labels, idx_test)
    loss_train_clean, acc_train_clean = calc_acc(output, labels, idx_train)

    log = 'train loss: {:.4f}, train acc: {:.4f}, train misacc: {:.4f}'
    print(log.format(loss_train_clean, acc_train_clean, 1-acc_train_clean))
    log = 'test loss: {:.4f}, test acc: {:.4f}, test misacc: {:.4f}'
    print(log.format(loss_test_clean, acc_test_clean, 1-acc_test_clean))


# Set the random seed so things involved torch.randn are repetable
def set_random_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if device != 'cpu':
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1, help='cuda')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
    parser.add_argument('--data_seed', type=int, default=123, help='Random seed for data split')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--gnn_base', type=str, default='gcn', help='base gnn models')
    parser.add_argument('--gnn_epochs', type=int, default=500, help='Number of epochs to train the gnn')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--patience', type=int, default=200, help='patience for early stopping')
    parser.add_argument('--model_dir', type=str, default='./nat_model_saved/', help='Directory to save the trained model.')
    parser.add_argument('--data_dir', type=str, default='./tmp/', help='Directory to download dataset.')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1) # limit cpu use
    
    set_random_seed(args.seed, args.device)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    
    print('==== Environment ====')
    print(f'torch version: {torch.__version__}')
    print(f'device: {args.device}')
    print(f'torch seed: {args.seed}')

    #########################################################
    # Load data for node classification task
    data = Dataset(root=args.data_dir, name=args.dataset, setting='gcn', seed=args.data_seed)
    adj, features, labels = data.process(process_adj=False, process_feature=False, device=args.device)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    print('==== Dataset ====')
    print(f'density: {nx.density(nx.from_numpy_array(adj.cpu().numpy()))}')
    print(f'adj shape: {adj.shape}')
    print(f'feature shape: {features.shape}')
    print(f'label number: {labels.max().item()+1}')
    print(f'split seed: {args.data_seed}')
    print(f'train|valid|test set: {idx_train.shape}|{idx_val.shape}|{idx_test.shape}')

    #########################################################
    # Setup gnn model and fit it on clean graph
    if args.gnn_base == 'gcn':
        nat_model = GCN(nfeat=features.shape[1], 
                        nclass=labels.max().item()+1, 
                        nhid=args.hidden,
                        dropout=args.dropout, 
                        weight_decay=args.weight_decay, 
                        lr=args.lr,
                        device=args.device)
    elif args.gnn_base == 'gat':
        nat_model = GAT(nfeat=features.shape[1],
                              nclass=labels.max().item() + 1,
                              nhid=args.hidden, 
                              heads=8,
                              dropout=args.dropout, 
                              weight_decay=args.weight_decay, 
                              lr=args.lr,
                              device=args.device)
    elif args.gnn_base == 'sgc':
        nat_model = SGC(nfeat=features.shape[1],
                        nclass=labels.max().item() + 1,
                        lr=args.lr, 
                        device=args.device)
    else:
        assert AssertionError ("GNN model {} not found!".format(args.gnn_base))

    nat_model = nat_model.to(args.device) 

    if args.gnn_base == 'gcn':
        nat_model.fit(features, adj, labels, idx_train, idx_val=None, train_iters=args.gnn_epochs, patience=args.patience, verbose=True)
    else:
        raise AssertionError('Model Not ready')

    print ('==== {} performance ===='.format(args.gnn_base))
    check_victim_model_performance(nat_model, features, adj, labels, idx_test, idx_train)

    #########################################################
    # Save the trained model
    path = args.model_dir + '{}_{}.pt'.format(args.dataset, args.gnn_base)
    torch.save(nat_model.state_dict(), path)
    

if __name__ == '__main__':
    main()
    