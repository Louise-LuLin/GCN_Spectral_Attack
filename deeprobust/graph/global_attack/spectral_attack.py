"""
Our attacking based on spectral of graph Laplacian

"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack

from deeprobust.graph.data import Dataset, Dpr2Pyg

from deeprobust.graph.myutils import calc_acc, save_all

class SpectralAttack(BaseAttack):
    """
    Spectral attack for graph data.

    Parameters

    """
    
    def __init__(self, 
                 model=None, 
                 nnodes=None, 
                 loss_type='CE', 
                 feature_shape=None,
                 attack_structure=True, 
                 attack_features=False,
                 regularization_weight=0.0,
                 device='cpu'):
        
        super(SpectralAttack, self).__init__(model, 
                                             nnodes, 
                                             attack_structure,
                                             attack_features,
                                             device)

        assert attack_structure or attack_features, 'attack_feature or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.regularization_weight = regularization_weight

        if attack_features:
            assert True, 'Current Spectral Attack does not support attack feature'

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            self.adj_changes.data.fill_(0)

        self.complementary = None

    def attack(self, ori_features, ori_adj, labels, idx_target, idx_train, idx_test,
               n_perturbations, att_lr,  epochs=200,
               verbose=False, reduction='mean', **kwargs):
        """
        Generate perturbations on the input graph
        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        # ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)

        victim_model.eval()
        for t in tqdm(range(epochs), desc='Perturb Adj'):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj, device=self.device)
            output = victim_model(ori_features, adj_norm) # forward of gcn need to normalize adj first
            self.loss = self._loss(output[idx_target], labels[idx_target])

            # New: add regularization term for spectral distance
            eigen_mse = 0
            eigen_norm = self.norm = torch.norm(ori_e)
            if self.regularization_weight != 0:
                e, v = torch.symeig(adj_norm, eigenvectors=True)
                # eigen_mse = F.mse_loss(ori_e, e, reduction=reduction)
                eigen_mse = torch.norm(ori_e, e)
            reg_loss = eigen_mse / eigen_norm * self.regularization_weight

            if verbose and t%20 == 0:
                loss_target, acc_target = calc_acc(output, labels, idx_target)
                print ('-- Epoch {}, '.format(t), 
                       'class loss = {:.4f} | '.format(self.loss.item()),
                       'reg loss = {:.8f} | '.format(reg_loss),
                       'eigen_mse = {:.8f} | '.format(eigen_mse),
                       'eigen_norm = {:.4f} | '.format(eigen_norm),
                       'acc = {}'.format(acc_target))

                
            self.loss += reg_loss
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
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        adj_norm = utils.normalize_adj_tensor(self.modified_adj, device=self.device)
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
                adj_norm = utils.normalize_adj_tensor(modified_adj, device=self.device)
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

    def projection(self, n_perturbations):
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
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


class MinMaxSpectral(SpectralAttack):
    """MinMax attack for graph data.

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    Examples
    --------

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MinMax
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = MinMax(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, 
                 model=None, 
                 nnodes=None, 
                 loss_type='CE', 
                 feature_shape=None,
                 attack_structure=True, 
                 attack_features=False,
                 regularization_weight=5.0,
                 device='cpu'):
        
        super(MinMaxSpectral, self).__init__(model, 
                                             nnodes, 
                                             loss_type, 
                                             feature_shape, 
                                             attack_structure, 
                                             attack_features,
                                             regularization_weight, 
                                             device=device)
        

    def attack(self, ori_features, ori_adj, labels, idx_target, idx_train, idx_test,
               n_perturbations, att_lr, epochs=200, 
               verbose=False, reduction='mean', **kwargs):
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs

        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        # ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)

        # optimizer
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)

        victim_model.eval()
        for t in tqdm(range(epochs)):
            # update victim model
            victim_model.train()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj, device=self.device)
            output = victim_model(ori_features, adj_norm)
            loss = self._loss(output[idx_target], labels[idx_target])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # generate pgd attack
            victim_model.eval()
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj, device=self.device)
            output = victim_model(ori_features, adj_norm)

            self.loss = self._loss(output[idx_target], labels[idx_target])

            # New: add regularization term for spectral distance
            eigen_mse = 0
            eigen_norm = self.norm = torch.norm(ori_e)
            if self.regularization_weight != 0:
                e, v = torch.symeig(adj_norm, eigenvectors=True)
                # eigen_mse = F.mse_loss(ori_e, e, reduction=reduction)
                eigen_mse = torch.norm(ori_e, e)
            reg_loss = eigen_mse / eigen_norm * self.regularization_weight

            if verbose and t%20 == 0:
                loss_target, acc_target = calc_acc(output, labels, idx_target)
                print ('-- Epoch {}, '.format(t), 
                       'class loss = {:.4f} | '.format(self.loss.item()),
                       'reg loss = {:.8f} | '.format(reg_loss),
                       'mse/norm = {:8f} | '.format(eigen_mse / eigen_norm),
                       'eigen_mse = {:.8f} | '.format(eigen_mse),
                       'eigen_norm = {:.4f} | '.format(eigen_norm),
                       'acc = {}'.format(acc_target))
                
            self.loss += reg_loss
            adj_grad = torch.autograd.grad(self.loss, self.adj_changes)[0]

            # adj_grad = self.adj_changes.grad

            if self.loss_type == 'CE':
                # lr = 200 / np.sqrt(t+1)
                lr = att_lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                # lr = 0.1 / np.sqrt(t+1)
                lr = att_lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            # self.adj_changes.grad.zero_()
            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels, idx_target, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

        # for sanity check
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        adj_norm = utils.normalize_adj_tensor(self.modified_adj, device=self.device)
        e, v = torch.symeig(adj_norm, eigenvectors=True)

        self.adj = ori_adj.detach()
        self.labels = labels.detach()
        self.ori_e = ori_e
        self.ori_v = ori_v
        self.e = e
        self.v = v
