"""
Decoder layers
"""
# pylint: disable=invalid-name
import torch
import torch.nn.functional as F
from torch import nn


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout: float = 0.0, act=lambda x: x):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class SampleDecoder(nn.Module):
    """Decoder Model , inner dot

    Args:
        activation (object, optional): activation of Decoder.
    """

    def __init__(self, act=torch.sigmoid):
        super().__init__()
        self.act = act

    def forward(self, zx, zy):
        """Forward Propagation

        Args:
            zx (torch.Tensor):Sample node embedding for x-axis
            zy (torch.Tensor): Sample node embedding for y-axis

        Returns:
            sim (torch.Tensor):prediction of adj
        """
        sim = (zx * zy).sum(1)
        sim = self.act(sim)

        return sim
