"""
Paper: Graph Structural Attack by Spectral Disctance
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm

from spac.global_attack import BaseAttack


def tensor2onehot(labels):
    """
    Convert label tensor to label onehot tensor
    """

    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)


def normalize_adj_tensor(adj, device='cuda:0'):
    """
    Normalize adjacency tensor matrix
    """
    device = torch.device(device if adj.is_cuda else "cpu")
    
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    
    return mx


def calc_acc(output, labels, idx):
    """
    Calculate the loss and accuracy on data points at idx
    """
    loss = F.nll_loss(output[idx], labels[idx])
    acc = accuracy(output[idx], labels[idx])
    return loss.item(), acc.item()


def accuracy(output, labels):
    """
    Return accuracy of output compared to labels
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class SpectralAttack(BaseAttack):
    """
    Spectral attack for graph data
    """
    
    def __init__(self, 
                 model=None, 
                 nnodes=None, 
                 loss_type='CE', 
                 regularization_weight=0.0,
                 device='cpu'):
        
        super(SpectralAttack, self).__init__()

        assert nnodes is not None, 'Please give nnodes='

        self.surrogate = model
        self.nnodes = nnodes
        self.loss_type = loss_type
        self.regularization_weight = regularization_weight
        self.device = device

        self.modified_adj = None
        self.modified_features = None

        self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
        self.adj_changes.data.fill_(0)

        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_target,
               n_perturbations, att_lr,  epochs=200, verbose=False, **kwargs):
        """
        Generate perturbations on the input graph
        """

        victim_model = self.surrogate

        ori_adj_norm = normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)

        victim_model.eval()
        # for t in tqdm(range(epochs), desc='Perturb Adj'):
        for t in range(epochs):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = normalize_adj_tensor(modified_adj, device=self.device)
            output = victim_model(ori_features, adj_norm) # forward of gcn need to normalize adj first
            self.loss = self._loss(output[idx_target], labels[idx_target])

            # spectral distance term for spectral distance
            eigen_mse, eigenvector_mse = 0, 0
            eigen_norm = self.norm = torch.norm(ori_e)
            if self.regularization_weight != 0:
                # add noise to make the graph asymmetric
                modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = normalize_adj_tensor(modified_adj_noise, device=self.device)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_mse = torch.norm(ori_e-e)
            
            spectral_dis = eigen_mse / eigen_norm
            
            if verbose and t%20 == 0:
                loss_target, acc_target = calc_acc(output, labels, idx_target)
                print ('-- Epoch {}, '.format(t), 
                       'class loss = {:.4f} | '.format(self.loss.item()),
                       'spectral distance = {:.8f} | '.format(spectral_dis),
                       'mse/norm = {:8f} | '.format(eigen_mse / eigen_norm),
                       'eigen_mse = {:.8f} | '.format(eigen_mse),
                       'acc/mis_acc = {:.4f}/{:.4f}'.format(acc_target, 1-acc_target))

            self.loss += self.regularization_weight * spectral_dis

            adj_grad = torch.autograd.grad(self.loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                # lr = 200 / np.sqrt(t+1)
                lr = att_lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                # lr = 0.1 / np.sqrt(t+1)
                lr = att_lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)
            
            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_target, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

        # for sanity check
        ori_adj_norm = normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        adj_norm = normalize_adj_tensor(self.modified_adj, device=self.device)
        e, v = torch.symeig(adj_norm, eigenvectors=True)

        self.adj = ori_adj.detach()
        self.labels = labels.detach()
        self.ori_e = ori_e
        self.ori_v = ori_v
        self.e = e
        self.v = v


    def random_sample(self, ori_adj, ori_features, labels, idx_target, n_perturbations):
        K = 10
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)
                # randm = np.random.uniform(size=s.shape[0])
                # sampled = np.where(s > randm, 1, 0)

                # if sampled.sum() > n_perturbations:
                #     continue
                while sampled.sum() > n_perturbations:
                    sampled = np.random.binomial(1, s)
                    
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = normalize_adj_tensor(modified_adj, device=self.device)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_target], labels[idx_target])
                # loss = F.nll_loss(output[idx_target], labels[idx_target])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def get_modified_adj(self, ori_adj):

        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj

    def add_random_noise(self, ori_adj):
        noise = 1e-4 * torch.rand(self.nnodes, self.nnodes).to(self.device)
        return (noise + torch.transpose(noise, 0, 1))/2.0 + ori_adj


    def projection(self, n_perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-4)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1).detach()
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, all-zero diagonal.
        """
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1, "Max value should be 1!"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"