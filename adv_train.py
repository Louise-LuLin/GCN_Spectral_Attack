import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import copy
from tqdm import tqdm
import argparse

from deeprobust.graph.gnns import GCN
from deeprobust.graph.global_attack import SpectralAttack, MinMaxSpectral
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PtbDataset
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

parser.add_argument('--gnn_base', type=str, default='gcn', 
                    choices=['gcn', 'gat'], help='base gnn models.')

parser.add_argument('--adv_epochs', type=int, default=150,
                    help='Number of epochs to adv loop.')
parser.add_argument('--ptb_rate', type=float, default=0.05,  
                    help='pertubation rate')
parser.add_argument('--reg_weight', type=float, default=0.0,  
                    help='regularization weight')


parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--loss_type', type=str, default='CE', 
                    choices=['CE', 'CW'], help='loss type')
parser.add_argument('--model', type=str, default='PGD', 
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--target_node', type=str, default='train', 
                    choices=['test', 'train', 'both'], help='target nodes to attack')

parser.add_argument('--model_dir', type=str, default='./rob_model_saved/',
                    help='Directory to save the trained model.')

parser.add_argument('--data_dir', type=str, default='./tmp/',
                    help='Directory to download dataset.')

args = parser.parse_args()

#########################################################
# Setting environment
print ('==== Environment ====')
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
if not osp.exists(args.data_dir):
    os.makedirs(args.data_dir)
data = Dataset(root=args.data_dir, name=args.dataset, setting='gcn', seed=args.data_seed)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

print ('  adj shape: ', adj.shape)
print ('  feature shape: ', features.shape)
print ('  label number: ', labels.max().item()+1)
print ('  train|valid|test set: {}|{}|{}'.format(idx_train.shape, idx_val.shape, idx_test.shape))

#########################################################
# Set victim model for adversarial training
adv_train_model = GCN(nfeat=features.shape[1], 
                   nclass=labels.max().item()+1, 
                   nhid=args.hidden,
                   dropout=args.dropout, 
                   weight_decay=args.weight_decay, 
                   lr=args.lr,
                   device=device)


adv_train_model = adv_train_model.to(device)
adv_train_model.initialize()

if args.model == 'PGD':
    model = SpectralAttack(model=adv_train_model, 
                        nnodes=adj.shape[0], 
                        loss_type=args.loss_type, 
                        regularization_weight=args.reg_weight,
                        device=device)
else:
    model = MinMaxSpectral(model=adv_train_model, 
                        nnodes=adj.shape[0], 
                        loss_type=args.loss_type, 
                        regularization_weight=args.reg_weight,
                        device=device)

model = model.to(device)

#########################################################
# Start adversarial training
nat_adj = copy.deepcopy(adj)
adv_adj = new_adv_adj = copy.deepcopy(adj)

print('==== Adversarial Training for Evasion Attack ====')
perturbations = int(args.ptb_rate * (adj.sum()//2))
for i in range(args.adv_epochs):
    # Setting swithc: target/untargeted attack
    if args.target_node == 'test':
        idx_target = idx_test # targeted
    elif args.target_node == 'train':
        idx_target = idx_train # ??
    else:
        idx_target = np.hstack((idx_test, idx_train)).astype(np.int) # untargeted
    
    # modified_adj = adversary.attack(features, adj)
    old_adv_adj = adv_adj
    adv_adj = new_adv_adj

    adv_train_model.train()
    adv_train_model.fit(features, adv_adj, labels, idx_target, train_iters=1, initialize=False)
   
    adv_train_model.eval()
    model.attack(features, adj, labels, idx_target, perturbations, epochs=40)
    new_adv_adj = model.modified_adj

    print('[adj diff] {}', np.sum(np.absolute(new_adv_adj.cpu().numpy()-adv_adj.cpu().numpy())))
        
    
    print ('############################## Epoch {} ###########################'.format(i))
    # evaluate
    output1 = adv_train_model.predict(features, nat_adj).cpu()
    loss_test_nat, acc_test_nat = calc_acc(output1, labels, idx_test)
    loss_train_nat, acc_train_nat = calc_acc(output1, labels, idx_train)

    print("[nat adj] train loss = {:.4f} | ".format(loss_train_nat),
            "train acc = {:.4f} | ".format(acc_train_nat),
            "test loss = {:.4f} | ".format(loss_test_nat),
            "test acc = {:.4f} | ".format(acc_test_nat))

    output2 = adv_train_model.predict(features, new_adv_adj).cpu()
    loss_test_adv, acc_test_adv = calc_acc(output2, labels, idx_test)
    loss_train_adv, acc_train_adv = calc_acc(output2, labels, idx_train)
    print("[adv adj] train loss = {:.4f} | ".format(loss_train_adv),
            "train acc = {:.4f} | ".format(acc_train_adv),
            "test loss = {:.4f} | ".format(loss_test_adv),
            "test acc = {:.4f} | ".format(acc_test_adv))

print ("==== Parameter ====")
print ('  Data seed: ', args.data_seed)
print ('  Dataset: ', args.dataset)
print ('  Loss type: ', args.loss_type)
print ('  Perturbation Rate: ', args.ptb_rate)
print ('  Reg weight: ', args.reg_weight)
print ('  Attack: ', args.model)
print ('  Target: ', args.target_node)

# save the adv_trained model
if not osp.exists(args.model_dir):
    os.makedirs(args.model_dir)
path = args.model_dir + '{}_{}_reg{}_ptb{}_epoch{}_node{}.pt'.format(args.dataset, 
       args.gnn_base, args.reg_weight, args.ptb_rate, args.adv_epochs, args.target_node)
torch.save(adv_train_model.state_dict(), path)

