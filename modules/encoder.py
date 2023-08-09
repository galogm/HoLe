import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn


def preprocess_graph(
    adj: sp.csr_matrix,
    layer: int,
    norm: str = "sym",
    renorm: bool = True,
) -> torch.Tensor:
    """Generalized Laplacian Smoothing Filter

    Args:
        adj (sp.csr_matrix): 2D sparse adj.
        layer (int):numbers of linear layers
        norm (str):normalize mode of Laplacian matrix
        renorm (bool): If with the renormalization trick

    Returns:
        adjs (sp.csr_matrix):Laplacian Smoothing Filter
    """
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == "sym":
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = (adj_.dot(degree_mat_inv_sqrt).transpose().dot(
            degree_mat_inv_sqrt).tocoo())
        laplacian = ident - adj_normalized
    elif norm == "left":
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.0).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [2 / 3] * (layer)

    adjs = []
    for i in reg:
        adjs.append(ident - (i * laplacian))
    return adjs


def scale(z):
    """Feature Scale
    Args:
        z (torch.Tensor):hidden embedding

    Returns:
        z_scaled (torch.Tensor):scaled embedding
    """
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std
    return z_scaled


class LinTrans(nn.Module):
    """Linear Transform Model

    Args:
        layers (int):number of linear layers.
        dims (list):Number of units in hidden layers.
    """

    def __init__(self, layers, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        """Forward Propagation

        Args:
            x (torch.Tensor):feature embedding

        Returns:
            out (torch.Tensor):hiddin embedding
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        out = scale(out)
        out = F.normalize(out)
        return out
