"""
Paper: Graph Structural Attack by Spectral Disctance
"""

from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
from deeprobust.graph.data import Dataset
from deeprobust.graph.myutils import calc_acc, save_all

class PGDAttack(BaseAttack):
    """
    Spectral attack for graph data
    """
    
    def __init__(self, 
                 model=None, 
                 nnodes=None, 
                 loss_type='CE', 
                 feature_shape=None,
                 attack_structure=True, 
                 attack_features=False,
                 loss_weight=1.0,
                 regularization_weight=0.0,
                 device='cpu'):
        
        super(PGDAttack, self).__init__(model, 
                                             nnodes, 
                                             attack_structure,
                                             attack_features,
                                             device)

        assert attack_structure or attack_features, 'attack_feature or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_adj = None
        self.modified_features = None
        self.loss_weight = loss_weight
        self.regularization_weight = regularization_weight

        if attack_features:
            assert True, 'Current Spectral Attack does not support attack feature'

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))
            torch.nn.init.uniform_(self.adj_changes, 0.0, 0.001)
            # self.adj_changes.data.fill_(0)

        self.complementary = None


    def set_model(self, model):
        self.surrogate = model


    def attack(self, ori_features, ori_adj, labels, idx_target, n_perturbations, att_lr,  epochs=200, 
               distance_type='l2',
               sample_type='sample',
               opt_type='max',
               verbose=True, **kwargs):
        """
        Generate perturbations on the input graph
        """

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        # ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)

        l, r, m = 0, 0 , 0
        victim_model.eval()
        # for t in tqdm(range(epochs), desc='Perturb Adj'):
        for t in tqdm(range(epochs)):
            modified_adj = self.get_modified_adj(ori_adj)
            adj_norm = utils.normalize_adj_tensor(modified_adj, device=self.device)
            output = victim_model(ori_features, adj_norm) # forward of gcn need to normalize adj first
            task_loss = self._loss(output[idx_target], labels[idx_target])

            # spectral distance term for spectral distance
            eigen_mse = torch.tensor(0)
            eigen_self = torch.tensor(0)
            eigen_gf = torch.tensor(0)
            eigen_norm = self.norm = torch.norm(ori_e)
            if self.regularization_weight != 0:
                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                # modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = utils.normalize_adj_tensor(modified_adj_noise, device=self.device)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_mse = torch.norm(ori_e-e)
                eigen_self = torch.norm(e)
                
                # low-rank loss in GF-attack
                idx = torch.argsort(e)[:128]
                mask = torch.zeros_like(e).bool()
                mask[idx] = True
                eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, ori_features), p=2), 2)
            
            reg_loss = 0
            if distance_type == 'l2':
                reg_loss = eigen_mse / eigen_norm
            elif distance_type == 'normDiv':
                reg_loss = eigen_self / eigen_norm
            elif distance_type == 'gf':
                reg_loss = eigen_gf
            else:
                exit(f'unknown distance metric: {distance_type}')
            
            if verbose and t%20 == 0:
                loss_target, acc_target = calc_acc(output, labels, idx_target)
                print ('-- Epoch {}, '.format(t), 
                       'ptb budget/true = {:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()),
                       'l/r/m = {:.4f}/{:.4f}/{:.4f}'.format(l, r, m),
                       'class loss = {:.4f} | '.format(task_loss.item()),
                       'reg loss = {:.4f} | '.format(reg_loss.item()),
                       'mse_norm = {:4f} | '.format(eigen_norm),
                       'eigen_mse = {:.4f} | '.format(eigen_mse),
                       'eigen_self = {:.4f} | '.format(eigen_self),
                       'acc/mis = {:.4f}/{:.4f}'.format(acc_target, 1-acc_target))

            self.loss = self.loss_weight * task_loss + self.regularization_weight * reg_loss
            
            adj_grad = torch.autograd.grad(self.loss, self.adj_changes)[0]

            if self.loss_type == 'CE':
                lr = att_lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = att_lr / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)
                
            # return self.adj_changes.cpu().detach().numpy()
                
            if verbose and t%20 == 0:
                print('budget/true={:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()))
            
            if sample_type == 'sample':
                l, r, m = self.projection(n_perturbations)
            elif sample_type == 'greedy':
                self.greedy(n_perturbations)
            elif sample_type == 'greedy2':
                self.greedy2(n_perturbations)
            elif sample_type == 'greedy3':
                self.greedy3(n_perturbations)
            else:
                exit(f"unkown sample type {sample_type}")
                
            if verbose and t%20 == 0:
                print('budget/true={:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()))
                
        if sample_type == 'sample':
            self.random_sample(ori_adj, ori_features, labels, idx_target, n_perturbations)
        elif sample_type == 'greedy':
            self.greedy(n_perturbations)
        elif sample_type == 'greedy2':
            self.greedy2(n_perturbations)
        elif sample_type == 'greedy3':
            self.greedy3(n_perturbations)
        else:
            exit(f"unkown sample type {sample_type}")
              
        print("final ptb budget/true= {:.1f}/{:.1f}".format(n_perturbations, self.adj_changes.sum()))
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


    def greedy(self, n_perturbations):
        s = self.adj_changes.cpu().detach().numpy()
        # l = min(s)
        # r = max(s)
        # noise = np.random.normal((l+r)/2, 0.1*(r-l), s.shape)
        # s += noise
        
        s_vec = np.squeeze(np.reshape(s, (1,-1)))
        # max_index = (-np.absolute(s_vec)).argsort()[:n_perturbations]
        max_index = (-s_vec).argsort()[:n_perturbations]
        
        mask = np.zeros_like(s_vec)
        mask[max_index]=1.0
        
        best_s = np.reshape(mask, s.shape)
        
        self.adj_changes.data.copy_(torch.clamp(torch.tensor(best_s), min=0, max=1))
        
    def greedy3(self, n_perturbations):
        s = self.adj_changes.cpu().detach().numpy()
        s_vec = np.squeeze(np.reshape(s, (1,-1)))
        # max_index = (-np.absolute(s_vec)).argsort()[:n_perturbations]
        max_index = (s_vec).argsort()[:n_perturbations]
        
        mask = np.zeros_like(s_vec)
        mask[max_index]=1.0
        
        best_s = np.reshape(mask, s.shape)
        
        self.adj_changes.data.copy_(torch.clamp(torch.tensor(best_s), min=0, max=1))
        
    def greedy2(self, n_perturbations):
        s = self.adj_changes.cpu().detach().numpy()
        l = min(s)
        r = max(s)
        noise = np.random.normal((l+r)/2, 0.4*(r-l), s.shape)
        s += noise
        
        s_vec = np.squeeze(np.reshape(s, (1,-1)))
        max_index = (-np.absolute(s_vec)).argsort()[:n_perturbations]
        
        mask = np.zeros_like(s_vec)
        mask[max_index]=1.0
        
        best_s = np.reshape(mask, s.shape)
        
        self.adj_changes.data.copy_(torch.clamp(torch.tensor(best_s), min=0, max=1))
        

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
                # if sampled.sum() > n_perturbations:
                #     indices = np.transpose(np.nonzero(sampled))
                #     candidate_idx = [m for m in range(indices.shape[0])]
                #     chosen_idx = np.random.choice(candidate_idx, n_perturbations, replace=False)
                #     chosen_indices = indices[chosen_idx, :]
                #     sampled = np.zeros_like(sampled)
                #     for idx in chosen_indices:
                #         sampled[idx] = 1
                    
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


    def add_random_noise(self, ori_adj):
        noise = 1e-4 * torch.rand(self.nnodes, self.nnodes).to(self.device)
        return (noise + torch.transpose(noise, 0, 1))/2.0 + ori_adj


    def projection2(self, n_perturbations):
        s = self.adj_changes.cpu().detach().numpy()
        n = np.squeeze(np.reshape(s, (1,-1))).shape[0]
        self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=n_perturbations/n))
        return 0, 0, 0


    def projection(self, n_perturbations):
        l, r, m = 0, 0, 0
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            l = left.cpu().detach()
            r = right.cpu().detach()
            m = miu.cpu().detach()
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data-miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))
        
        return l, r, m
            
            
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
                b = miu
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu


    # def calc_utility(self, ori_features, ori_adj, labels, idx_target):
    #     victim_model = self.surrogate
    #     self.sparse_features = sp.issparse(ori_features)
    #     ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
    #     ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)
        
    #     victim_model.eval()
    #     s = self.adj_changes.cpu().detach().numpy()
    #     s_vec = np.squeeze(np.reshape(s, (1,-1)))
    #     utility = []
    #     for i in tqdm(range(len(s_vec))):
    #         mask = np.zeros_like(s_vec)
    #         mask[i] = 1.0
    #         best_s = np.reshape(mask, s.shape)
    #         self.adj_changes.data.copy_(torch.tensor(best_s))
            
    #         modified_adj = self.get_modified_adj(ori_adj)
    #         adj_norm = utils.normalize_adj_tensor(modified_adj, device=self.device)
    #         output = victim_model(ori_features, adj_norm) # forward of gcn need to normalize adj first
    #         task_loss = self._loss(output[idx_target], labels[idx_target])
            
    #         # spectral distance term for spectral distance
    #         eigen_mse = 0
    #         eigen_self = 0
    #         eigen_norm = self.norm = torch.norm(ori_e)
            
    #         # add noise to make the graph asymmetric
    #         modified_adj_noise = modified_adj
    #         # modified_adj_noise = self.add_random_noise(modified_adj)
    #         adj_norm_noise = utils.normalize_adj_tensor(modified_adj_noise, device=self.device)
    #         e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
    #         eigen_mse = torch.norm(ori_e-e)
    #         eigen_self = torch.norm(e)
            
    #         loss_target, acc_target = calc_acc(output, labels, idx_target)
            
    #         utility.append((task_loss.item(), acc_target, (eigen_mse/eigen_norm).item(), (eigen_self/eigen_norm).item()))
            
    #         if i%100 == 0:
    #             print(utility[-1])
            
    #     return utility


class MinMax(PGDAttack):

    def __init__(self, 
                 model=None, 
                 nnodes=None, 
                 loss_type='CE', 
                 feature_shape=None,
                 attack_structure=True, 
                 attack_features=False,
                 loss_weight=1.0,
                 regularization_weight=5.0,
                 device='cpu'):
        
        super(MinMax, self).__init__(model, 
                                             nnodes, 
                                             loss_type, 
                                             feature_shape, 
                                             attack_structure, 
                                             attack_features,
                                             loss_weight,
                                             regularization_weight, 
                                             device=device)
        

    def attack(self, ori_features, ori_adj, labels, idx_target, n_perturbations, att_lr, epochs=200, 
               distance_type='l2',
               sample_type='sample',
               opt_type='max',
               **kwargs):

        victim_model = self.surrogate

        self.sparse_features = sp.issparse(ori_features)
        # ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj, device=self.device)
        ori_e, ori_v = torch.symeig(ori_adj_norm, eigenvectors=True)

        # optimizer
        optimizer = optim.Adam(victim_model.parameters(), lr=0.01)

        l, r, m = 0, 0, 0
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

            task_loss = self._loss(output[idx_target], labels[idx_target])

            # spectral distance term for spectral distance
            eigen_mse = torch.tensor(0)
            eigen_self = torch.tensor(0)
            eigen_gf = torch.tensor(0)
            eigen_norm = self.norm = torch.norm(ori_e)
            if self.regularization_weight != 0:
                # add noise to make the graph asymmetric
                modified_adj_noise = modified_adj
                # modified_adj_noise = self.add_random_noise(modified_adj)
                adj_norm_noise = utils.normalize_adj_tensor(modified_adj_noise, device=self.device)
                e, v = torch.symeig(adj_norm_noise, eigenvectors=True)
                eigen_mse = torch.norm(ori_e-e)
                eigen_self = torch.norm(e)
                
                # low-rank loss in GF-attack
                idx = torch.argsort(e)[:128]
                mask = torch.zeros_like(e).bool()
                mask[idx] = True
                eigen_gf = torch.pow(torch.norm(e*mask, p=2), 2) * torch.pow(torch.norm(torch.matmul(v.detach()*mask, ori_features), p=2), 2)
            
            reg_loss = 0
            if distance_type == 'l2':
                reg_loss = eigen_mse / eigen_norm
            elif distance_type == 'normDiv':
                reg_loss = eigen_self / eigen_norm
            elif distance_type == 'gf':
                reg_loss = eigen_gf
            else:
                exit(f'unknown distance metric: {distance_type}')

            if t%20 == 0:
                loss_target, acc_target = calc_acc(output, labels, idx_target)
                print ('-- Epoch {}, '.format(t), 
                       'ptb budget/true = {:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()),
                       'l/r/m = {:.4f}/{:.4f}/{:.4f}'.format(l, r, m),
                       'class loss = {:.4f} | '.format(task_loss.item()),
                       'reg loss = {:.4f} | '.format(reg_loss.item()),
                       'mse_norm = {:4f} | '.format(eigen_norm),
                       'eigen_mse = {:.4f} | '.format(eigen_mse),
                       'eigen_self = {:.4f} | '.format(eigen_self),
                       'acc/mis = {:.4f}/{:.4f}'.format(acc_target, 1-acc_target))
                
            self.loss = self.loss_weight * task_loss + self.regularization_weight * reg_loss
            if opt_type == 'min':
                self.loss = -self.loss

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

            if t%20 == 0:
                print('budget/true={:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()))
            
            # self.adj_changes.grad.zero_()
            if sample_type == 'sample':
                l, r, m = self.projection(n_perturbations)
            elif sample_type == 'greedy':
                self.greedy(n_perturbations)
            else:
                exit(f"unkown sample type {sample_type}")
                
            if t%20 == 0:
                print('budget/true={:.1f}/{:.1f}'.format(n_perturbations, torch.clamp(self.adj_changes, 0, 1).sum()))

        if sample_type == 'sample':
            self.random_sample(ori_adj, ori_features, labels, idx_target, n_perturbations)
        elif sample_type == 'greedy':
            self.greedy(n_perturbations)
        else:
            exit(f"unkown sample type {sample_type}")
        
        print("final ptb budget/true= {:.1f}/{:.1f}".format(n_perturbations, self.adj_changes.sum()))
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
