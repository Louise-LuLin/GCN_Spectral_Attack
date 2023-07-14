"""
Inherit from DeepRobust: https://github.com/DSE-MSU/DeepRobust

"""

from .gcn import GCN, GraphConvolution
import warnings
try:
    from .gat import GAT
    from .sgc import SGC
except ImportError as e:
    print(e)
    warnings.warn("Please install pytorch geometric if you " +
            "would like to use the datasets from pytorch " +
            "geometric. See details in https://pytorch-geom" +
            "etric.readthedocs.io/en/latest/notes/installation.html")

__all__ = ['GCN', 'GraphConvolution', 'GAT', 'SGC']
