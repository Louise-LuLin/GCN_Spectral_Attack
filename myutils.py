import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path as osp
import argparse
import pickle as pkl
from deeprobust.graph.utils import *

def calc_acc(output, labels, idx):
    # Performance on test set
    loss = F.nll_loss(output[idx], labels[idx])
    acc = accuracy(output[idx], labels[idx])
    return loss.item(), acc.item()

def save_all(root, model):
    ''' Save intermediate results '''
    
    os.makedirs(root, exist_ok=True)
    # save modified_adj, adj, labels, ori_e, ori_v, e, v
    torch.save(model.modified_adj, '{}/modified_adj.pt'.format(root))
    torch.save(model.adj, '{}/adj.pt'.format(root))
    torch.save(model.labels, '{}/labels.pt'.format(root))
    torch.save(model.ori_e, '{}/ori_e.pt'.format(root))
    torch.save(model.ori_v, '{}/ori_v.pt'.format(root))
    torch.save(model.e, '{}/e.pt'.format(root)) 
    torch.save(model.v, '{}/v.pt'.format(root))
    
    print ('==== intermediate results saved to {} ===='.format(root))

    # if type(adj) is torch.Tensor:
    #     sparse_adj = to_scipy(adj)
    #     sp.save_npz(osp.join(root, name), sparse_adj)
    # else:
    #     sp.save_npz(osp.join(root, name), adj)