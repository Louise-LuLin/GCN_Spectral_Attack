"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
from torch_geometric.nn import GATConv

from spac.graph import utils
from spac.data_loader import Dataset

class GAT(nn.Module):
    """ 2 Layer Graph Attention Network based on pytorch geometric.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    heads: int
        number of attention heads
    output_heads: int
        number of attention output heads
    dropout : float
        dropout rate for GAT
    lr : float
        learning rate for GAT
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GAT weights.
    device: str
        'cpu' or 'cuda'.

    """

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass

        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        print (x.shape, edge_index.shape)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def fit(self, geodata, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        """Train the GAT model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        geodata :
            pytorch geometric Data object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # self.device = self.conv1.weight.device
        if initialize:
            self.initialize()

        self.data = geodata.to(self.device)

        # use patience to control whether to use early stopping on validation
        if patience > 1000:
            self._train_without_val(train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(train_iters, patience, verbose)
            else:
                self._train_with_val(train_iters, patience, verbose)

    def _train_without_val(self, train_iters, verbose):
        """No early stopping
        """
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        self.output = self.forward(self.data)

    def _train_with_val(self, train_iters, patience, verbose):
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, dropout=0.0):
        """Evaluate GAT performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        self.dropout = dropout

        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self, geodata=None, dropout=0.0):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GAT
        """

        self.eval()
        self.dropout = dropout

        self.data = self.data if geodata==None else geodata.to(self.device)
        
        return self.forward(self.data)


if __name__ == "__main__":
    data = Dataset(root='./tmp/', name='cora', setting='gcn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    gat = GAT(nfeat=features.shape[1],
          nhid=8, heads=8,
          nclass=labels.max().item() + 1,
          dropout=0.5, device='cpu')
    gat = gat.to('cpu')
    gat.fit(data.geodata, verbose=True) # train with earlystopping
    gat.test()
    print(gat.predict())

