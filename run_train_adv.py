import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import copy
from tqdm import tqdm
import argparse
import networkx as nx

from deeprobust.graph.gnns import GCN
from deeprobust.graph.global_attack import SpectralAttack, MinMaxSpectral
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PtbDataset
from deeprobust.graph.myutils import calc_acc, save_all


# print the victim model's performance given graph info
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

    parser.add_argument('--cuda', type=int, default=0, help='cuda')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for model.')
    parser.add_argument('--data_seed', type=int, default=123, help='Random seed for data split.')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--gnn_base', type=str, default='gcn', help='base gnn models.')
    parser.add_argument('--adv_epochs', type=int, default=100,  help='Number of epochs to adv loop.')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
    parser.add_argument('--att_lr', type=float, default=200, help='Initial learning rate')
    parser.add_argument('--perturb_epochs', type=int, default=5, help='Number of epochs to poisoning loop')
    parser.add_argument('--reg_weight', type=float, default=0.0, help='regularization weight')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--loss_type', type=str, default='CE', help='loss type')
    parser.add_argument('--model', type=str, default='PGD', help='model variant')
    parser.add_argument('--target_node', type=str, default='train', help='target nodes to attack')
    parser.add_argument('--model_dir', type=str, default='./rob_model_saved/', help='Directory to save the trained model.')
    parser.add_argument('--data_dir', type=str, default='./tmp/', help='Directory to download dataset.')
    
    parser.add_argument('--distance_type', type=str, default='l2', help='distance type')
    parser.add_argument('--sample_type', type=str, default='sample', help='sample type')
    

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1) # limit cpu use
    
    set_random_seed(args.seed, args.device)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

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

    # adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

    #########################################################
    # Set victim model for adversarial training
    adv_train_model = GCN(nfeat=features.shape[1], 
                    nclass=labels.max().item()+1, 
                    nhid=args.hidden,
                    dropout=args.dropout, 
                    weight_decay=args.weight_decay, 
                    lr=args.lr,
                    device=args.device)

    adv_train_model = adv_train_model.to(args.device)
    adv_train_model.initialize()

    if args.model == 'PGD':
        model = SpectralAttack(model=adv_train_model, 
                            nnodes=adj.shape[0], 
                            loss_type=args.loss_type, 
                            regularization_weight=args.reg_weight,
                            device=args.device)
    else:
        model = MinMaxSpectral(model=adv_train_model, 
                            nnodes=adj.shape[0], 
                            loss_type=args.loss_type, 
                            regularization_weight=args.reg_weight,
                            device=args.device)

    model = model.to(args.device)

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
        model.set_model(adv_train_model)
        model.attack(features, adj, labels, idx_target, perturbations, args.att_lr, epochs=args.perturb_epochs, 
                     distance_type=args.distance_type,
                     sample_type=args.sample_type,
                     verbose=False)

        new_adv_adj = model.modified_adj
            
        if i%20 == 0:
            # evaluate
            print(f'***** Epoch {i} *****')
            print('[adj diff] {}', np.sum(np.absolute(new_adv_adj.cpu().numpy()-adv_adj.cpu().numpy())))
            print('==== Adv Trained Model on Clean Graph ====')
            check_victim_model_performance(adv_train_model, features, nat_adj, labels, idx_test, idx_train)
            print('==== Adv Trained Model on Purturbed Graph ====')
            check_victim_model_performance(adv_train_model, features, new_adv_adj, labels, idx_test, idx_train)

    print("==== Parameter ====")
    print(f'Data seed: {args.data_seed}')
    print(f'Dataset: {args.dataset}')
    print(f'Loss type: {args.loss_type}')
    print(f'Perturbation Rate: {args.ptb_rate}')
    print(f'Reg weight: {args.reg_weight}')
    print(f'Attack: {args.model}')
    print(f'Attack seed: {args.seed}')

    # save the adv_trained model
    path = args.model_dir + '{}_{}_reg{}_ptb{}_epoch{}_node{}.pt'.format(args.dataset, 
        args.gnn_base, args.reg_weight, args.ptb_rate, args.adv_epochs, args.target_node)
    torch.save(adv_train_model.state_dict(), path)


if __name__ == '__main__':
    main()