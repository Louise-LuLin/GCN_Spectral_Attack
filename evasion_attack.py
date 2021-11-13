import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import pickle as pkl
import copy
import networkx as nx

from deeprobust.graph.gnns import GCN, GAT, SGC
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg

from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.global_attack import IGAttack
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.global_attack import Random

from deeprobust.graph.myutils import calc_acc, save_all, save_utility


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
    parser.add_argument('--seed', type=int, default=123, help='Random seed for model')
    parser.add_argument('--data_seed', type=int, default=123,help='Random seed for data split')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--gnn_path', type=str, required=True, help='Path of saved model')
    parser.add_argument('--model', type=str, default='PGD', help='model variant')  # PGD, random
    parser.add_argument('--loss_type', type=str, default='CE', help='loss type')
    parser.add_argument('--att_lr', type=float, default=200, help='Initial learning rate')
    parser.add_argument('--perturb_epochs', type=int, default=100, help='Number of epochs to poisoning loop')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='pertubation rate')
    parser.add_argument('--reg_weight', type=float, default=0.0, help='regularization weight')
    parser.add_argument('--loss_weight', type=float, default=1.0, help='loss weight')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--data_dir', type=str, default='./tmp/', help='Directory to download dataset')
    parser.add_argument('--sanitycheck', type=str, default='no', help='whether store the intermediate results')
    parser.add_argument('--sanity_dir', type=str, default='./sanitycheck_evasion/', help='Directory to store the intermediate results')
    
    parser.add_argument('--distance_type', type=str, default='l2', help='distance type')
    parser.add_argument('--sample_type', type=str, default='sample', help='sample type')
    

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(1) # limit cpu use
    
    set_random_seed(args.seed, args.device)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.exists(args.sanity_dir):
        os.mkdir(args.sanity_dir)
    if not os.path.exists(args.gnn_path):
        raise AssertionError (f'No trained model found under {args.gnn_path}!')

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
    # Load victim model and test it on clean training nodes
    victim_model = GCN(
        nfeat=features.shape[1], 
        nclass=labels.max().item()+1, 
        nhid=args.hidden,
        dropout=args.dropout, 
        weight_decay=args.weight_decay,
        device=args.device)
    victim_model = victim_model.to(args.device)
    
    victim_model.load_state_dict(torch.load(args.gnn_path))
    victim_model.eval()

    print('==== Victim Model on Clean Graph ====')
    check_victim_model_performance(victim_model, features, adj, labels, idx_test, idx_train)

    #########################################################
    # Setup attack model
    if args.model == 'PGD':
        model = PGDAttack(
            model=victim_model, 
            nnodes=adj.shape[0], 
            loss_type=args.loss_type, 
            loss_weight=args.loss_weight,
            regularization_weight=args.reg_weight,
            device=args.device)
        model = model.to(args.device)
    elif args.model == 'random':
        model = Random()
    else:
        raise AssertionError (f'Attack {args.model} not found!')
        
    #########################################################
    # Attack and evaluate
    print('***************** seed {} *****************'.format(args.seed))
    print('==== Attacking ====')

    perturbations = int(args.ptb_rate * (adj.sum()/2))
    nat_adj = copy.deepcopy(adj)

    # global attack on whole testing nodes
    idx_target = idx_test
    
    # utility = model.calc_utility(
    #     features, 
    #     nat_adj, 
    #     labels, 
    #     idx_target)
    # os.makedirs(f'./{args.dataset}', exist_ok=True)
    # save_utility(f'./{args.dataset}/utility.bin', utility)
    # exit('done')
    
    # gradients = model.attack(
    #         features, 
    #         nat_adj, 
    #         labels, 
    #         idx_target,
    #         perturbations, 
    #         att_lr=args.att_lr, 
    #         epochs=args.perturb_epochs,
    #         distance_type=args.distance_type,
    #         sample_type=args.sample_type)
    # os.makedirs(f'./{args.dataset}', exist_ok=True)
    # save_utility(f'./{args.dataset}/grad_{args.reg_weight}.bin', gradients)
    # exit('done')
    
    if args.model == 'random':
        model.attack(nat_adj, perturbations, 'flip')
    else:
        model.attack(
            features, 
            nat_adj, 
            labels, 
            idx_target,
            perturbations, 
            att_lr=args.att_lr, 
            epochs=args.perturb_epochs,
            distance_type=args.distance_type,
            sample_type=args.sample_type)

    modified_adj = model.modified_adj
    
    # evaluation
    victim_model.load_state_dict(torch.load(args.gnn_path)) # reset to clean model
    victim_model.eval()
    
    print('==== Victim Model on Perturbed Graph ====')
    check_victim_model_performance(victim_model, features, modified_adj, labels, idx_test, idx_train)

    print("==== Parameter ====")
    print(f'Data seed: {args.data_seed}')
    print(f'Dataset: {args.dataset}')
    print(f'Loss type: {args.loss_type}')
    print(f'Perturbation Rate: {args.ptb_rate}')
    print(f'Reg weight: {args.reg_weight}')
    print(f'Attack: {args.model}')
    print(f'Attack seed: {args.seed}')

    # if you want to save the modified adj/features, uncomment the code below
    if args.sanitycheck == 'yes':
        root = args.sanity_dir + '{}_{}_{}_{}_{}_{}lr_{}epoch_{}rate_{}reg1_{}reg2_{}seed'
        root = root.format(args.dataset, args.distance_type, args.sample_type, args.model, 
                           args.loss_type, args.att_lr, args.perturb_epochs, args.ptb_rate, args.loss_weight, args.reg_weight, args.seed)
        save_all(root, model)


if __name__ == '__main__':
    main()