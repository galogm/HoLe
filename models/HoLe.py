"""Homophily-enhanced Structure Learning for Graph Clustering
"""
import copy
import math
import time
from typing import Callable
from typing import List

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn

from modules import InnerProductDecoder
from modules import LinTrans
from modules import preprocess_graph
from modules import SampleDecoder
from utils import eliminate_zeros
from utils import set_seed


class HoLe(nn.Module):
    """HoLe"""

    def __init__(
        self,
        in_feats: int,
        hidden_units: List,
        n_clusters: int,
        n_lin_layers: int = 1,
        n_gnn_layers: int = 10,
        n_cls_layers: int = 1,
        lr: float = 0.001,
        n_epochs: int = 400,
        n_pretrain_epochs: int = 400,
        norm: str = "sym",
        renorm: bool = True,
        tb_filename: str = "hole",
        warmup_filename: str = "hole_warmup",
        inner_act: Callable = lambda x: x,
        udp=10,
        reset=False,
        regularization=0,
        seed: int = 4096,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_gnn_layers = n_gnn_layers
        self.n_lin_layers = n_lin_layers
        self.n_cls_layers = n_cls_layers
        self.hidden_units = hidden_units
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_pretrain_epochs = n_pretrain_epochs
        self.norm = norm
        self.renorm = renorm
        self.device = None
        self.sm_fea_s = None
        self.adj_label = None
        self.lbls = None
        self.tb_filename = tb_filename
        self.warmup_filename = warmup_filename
        self.udp = udp
        self.reset = reset
        self.labels = None
        self.adj_sum_raw = None
        self.adj_orig = None
        self.regularization = regularization
        set_seed(seed)

        self.dims = [in_feats] + hidden_units
        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.cluster_layer = nn.Parameter(
            torch.Tensor(self.n_clusters, hidden_units[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sample_decoder = SampleDecoder(act=lambda x: x)
        self.inner_product_decoder = InnerProductDecoder(act=inner_act)

        self.best_model = copy.deepcopy(self)

    def reset_weights(self):
        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def bce_loss(preds, labels, norm=1.0, pos_weight=None):
        return norm * F.binary_cross_entropy_with_logits(
            preds,
            labels,
            pos_weight=pos_weight,
        )

    @staticmethod
    def get_fd_loss(z: torch.Tensor, norm: int = 1) -> torch.Tensor:
        """Feature decorrelation loss.

        Args:
            z (torch.Tensor): embedding matrix.
            norm (int, optional): coefficient. Defaults to 1.

        Returns:
            torch.Tensor: loss.
        """
        norm_ff = z / (z**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(2.0)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = norm * F.cross_entropy(coef_mat, a)
        return L_fd

    @staticmethod
    def target_distribution(q):
        """get target distribution P

        Args:
            q (torch.Tensor): Soft assignments

        Returns:
            torch.Tensor: target distribution P
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_cluster_center(self, z=None):
        if z is None:
            z = self.get_embedding()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _ = kmeans.fit_predict(z.data.cpu().numpy())
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(
            self.device)

    def get_Q(self, z):
        """get soft clustering assignment distribution

        Args:
            z (torch.Tensor): node embedding

        Returns:
            torch.Tensor: Soft assignments
        """
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def _pretrain(self):
        best_loss = 1e9

        for epoch in range(self.n_pretrain_epochs):
            self.train()
            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)
            preds = self.inner_product_decoder(z).view(-1)
            loss = self.bce_loss(
                preds,
                self.lbls,
                norm=self.norm_weights,
                pos_weight=self.pos_weight,
            )

            (loss + self.get_fd_loss(z, self.regularization)).backward()

            self.optimizer.step()
            cur_loss = loss.item()

            print(f"Cluster Epoch: {epoch}, embeds_loss={cur_loss}"
                  f"time={time.time() - t:.5f}")

            if cur_loss < best_loss:
                best_loss = cur_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def _train(self):
        best_loss = 1e9

        self.get_cluster_center()
        for epoch in range(self.n_epochs):
            self.train()
            if epoch % self.udp == 0 or epoch == self.n_epochs - 1:
                with torch.no_grad():
                    z_detached = self.get_embedding()
                    Q = self.get_Q(z_detached)
                    q = Q.detach().data.cpu().numpy().argmax(1)

            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)
            preds = self.inner_product_decoder(z).view(-1)
            loss = self.bce_loss(preds, self.lbls, self.norm_weights,
                                 self.pos_weight)

            q = self.get_Q(z)
            p = self.target_distribution(Q.detach())
            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")

            (loss + kl_loss + self.regularization * self.bce_loss(
                self.inner_product_decoder(q).view(-1), preds)).backward()

            self.optimizer.step()

            cur_loss = loss.item()

            print(f"Cluster Epoch: {epoch}, embeds_loss={cur_loss:.5f},"
                  f"kl_loss={kl_loss.item()},"
                  f"time={time.time() - t:.5f}")

            if cur_loss < best_loss:
                best_loss = cur_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def update_adj(
        self,
        adj,
        idx=0,
        ratio=0.2,
        edge_ratio_raw=1,
        del_ratio=0.005,
    ):
        adj_cp = copy.deepcopy(adj).tolil()
        n_nodes = adj.shape[0]

        self.to(self.device)
        ratio = ratio + 0.01 * idx
        edge_ratio = edge_ratio_raw * (1 + idx)
        if del_ratio * (1 - 0.05 * idx) > 0:
            del_ratio = del_ratio * (1 - 0.05 * idx)
        else:
            del_ratio = del_ratio * 0.05

        z_detached = self.get_embedding()
        with torch.no_grad():
            soft = self.get_Q(z_detached)
        preds = soft.argmax(dim=1).view(-1).cpu().numpy()

        preds_soft = soft[list(range(adj_cp.shape[0])), preds].cpu().numpy()

        top_k_idx = np.array([], dtype="int64")
        for i in range(self.n_clusters):
            c_idx = (preds == i).nonzero()[0]
            top_k_idx = np.concatenate((
                top_k_idx,
                c_idx[preds_soft[c_idx].argsort()[::-1]
                      [:int(len(c_idx) * ratio)]],
            ))

        top_k_preds = preds[top_k_idx]

        if edge_ratio != 0:
            adj_tensor = torch.IntTensor(self.adj_orig.toarray())
            for c in range(self.n_clusters):
                mask = top_k_preds == c

                n_nodes_c = mask.sum()
                cluster_c = top_k_idx[mask]
                adj_c = adj_tensor.index_select(
                    0, torch.LongTensor(cluster_c)).index_select(
                        1, torch.LongTensor(cluster_c))

                add_num = math.ceil(edge_ratio * self.adj_sum_raw * n_nodes_c /
                                    n_nodes)
                if add_num == 0:
                    add_num = math.ceil(self.adj_sum_raw * n_nodes_c / n_nodes)

                z_detached = self.get_embedding()
                cluster_embeds = z_detached[cluster_c]
                sim = torch.matmul(cluster_embeds, cluster_embeds.t())

                sim = (
                    sim *
                    (torch.eye(sim.shape[0], dtype=int) ^ 1).to(self.device) +
                    torch.eye(sim.shape[0], dtype=int).to(self.device) * -100)

                sim = sim * (adj_c ^ 1).to(self.device) + adj_c.to(
                    self.device) * -100

                sim = sim.view(-1)
                top = sim.sort(descending=True)
                top_sort = top.indices.cpu().numpy()

                sim_top_k = top_sort[:add_num * 2]
                xind = sim_top_k // n_nodes_c
                yind = sim_top_k % n_nodes_c
                u = cluster_c[xind]
                v = cluster_c[yind]

                adj_cp[u, v] = adj_cp[v, u] = 1

            print(f"add edges in all: {adj_cp.sum() - adj.sum()}")

        edge_num = adj_cp.sum()

        del_num = math.ceil(del_ratio * edge_num)
        if del_num != 0:
            eds = adj_cp.toarray().reshape(-1).astype(bool)

            mask = eds

            negs_inds = np.nonzero(mask)[0]

            z_detached = self.get_embedding()
            cosine = np.matmul(
                z_detached.cpu().numpy(),
                np.transpose(z_detached.cpu().numpy()),
            ).reshape([-1])
            negs = cosine[mask]
            del_num = min(del_num, len(negs))

            add_num_actual = adj_cp.sum() - adj.sum()
            if del_num > add_num_actual:
                print(
                    f"Ooops, too few edges added: {add_num_actual}, but {del_num} needs to be deleted."
                )
                del_num = int(add_num_actual)

            del_edges = negs_inds[np.argpartition(negs,
                                                  del_num - 1)[:del_num - 1]]

            u_del = del_edges // z_detached.shape[0]
            v_del = del_edges % z_detached.shape[0]

            adj_cp[u_del, v_del] = adj_cp[v_del,
                                          u_del] = np.array([0] * len(u_del))

            isolated_nodes = (np.asarray(adj_cp.sum(1)).flatten().astype(bool)
                              ^ True).nonzero()[0]
            if len(isolated_nodes):
                for node_zero in isolated_nodes:
                    with torch.no_grad():
                        max_sim_node_0 = np.argpartition(
                            self.sample_decoder(z_detached[node_zero],
                                                z_detached).cpu().numpy(),
                            n_nodes - 2,
                        )[-2:][0]
                        adj_cp[node_zero,
                               max_sim_node_0] = adj_cp[max_sim_node_0,
                                                        node_zero] = 1

                        neighbors = self.adj_orig[node_zero, :].nonzero()[1]
                        if len(neighbors):
                            max_sim_node_1 = neighbors[self.sample_decoder(
                                z_detached[node_zero],
                                z_detached[neighbors]).argmax()]
                            adj_cp[node_zero,
                                   max_sim_node_1] = adj_cp[max_sim_node_1,
                                                            node_zero] = 1
        return adj_cp

    def update_features(self, adj):
        """Check whether adj matrix needs to remove self-loops
        """
        sm_fea_s = sp.csr_matrix(self.features).toarray()

        adj_cp = copy.deepcopy(adj)
        self.adj_label = adj_cp

        adj_norm_s = preprocess_graph(
            adj_cp,
            self.n_gnn_layers,
            norm=self.norm,
            renorm=self.renorm,
        )
        adj_csr = adj_norm_s[0] if len(adj_norm_s) > 0 else adj_cp

        print("Laplacian Smoothing...")
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(self.device)

        self.pos_weight = torch.FloatTensor([
            (float(adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) /
             adj_csr.sum())
        ]).to(self.device)
        self.norm_weights = (adj_csr.shape[0] * adj_csr.shape[0] / float(
            (adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) * 2))

        self.lbls = torch.FloatTensor(adj_csr.todense()).view(-1).to(
            self.device)

    def fit(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        add_edge_ratio=0.2,
        del_edge_ratio=0.1,
        node_ratio=0.2,
        gsl_epochs=10,
        labels=None,
        adj_sum_raw=None,
        load=False,
        dump=True,
    ) -> None:
        """Fitting

        Args:
            adj (sp.csr_matrix): 2D sparse adj.
            features (torch.Tensor): features.
        """
        self.device = device
        self.features = graph.ndata["feat"]
        adj = self.adj_orig = graph.adj_external(scipy_fmt="csr")
        self.n_nodes = self.features.shape[0]
        self.labels = labels
        self.adj_sum_raw = adj_sum_raw

        adj = eliminate_zeros(adj)

        self.to(self.device)

        self.update_features(adj)

        adj = self.adj_label

        from utils.utils import check_modelfile_exists

        if load and check_modelfile_exists(self.warmup_filename):
            from utils.utils import load_model

            self, self.optimizer, _, _ = load_model(
                self.warmup_filename,
                self,
                self.optimizer,
                self.device,
            )

            self.to(self.device)
            print(f"model loaded from {self.warmup_filename} to {self.device}")
        else:
            self._pretrain()
            if dump:
                from utils.utils import save_model

                save_model(
                    self.warmup_filename,
                    self,
                    self.optimizer,
                    None,
                    None,
                )
                print(f"dump to {self.warmup_filename}")

        if gsl_epochs != 0:
            self._train()

            with torch.no_grad():
                z_detached = self.get_embedding()
                Q = self.get_Q(z_detached)
                q = Q.detach().data.cpu().numpy().argmax(1)

        adj_pre = copy.deepcopy(adj)

        for gls_ep in range(1, gsl_epochs + 1):
            print(f"==============GSL epoch:{gls_ep} ===========")

            adj_new = eliminate_zeros(
                self.update_adj(
                    adj_pre,
                    idx=gls_ep,
                    ratio=node_ratio,
                    del_ratio=del_edge_ratio,
                    edge_ratio_raw=add_edge_ratio,
                ))

            self.update_features(adj=adj_new)

            self._train()

            adj_pre = copy.deepcopy(self.adj_label)

            if self.reset:
                self.reset_weights()
                self.to(self.device)

    def get_embedding(self, best=True):
        with torch.no_grad():
            mu = (self.best_model.encoder(self.sm_fea_s)
                  if best else self.encoder(self.sm_fea_s))
        return mu.detach()
