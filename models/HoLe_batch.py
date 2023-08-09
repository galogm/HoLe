"""Homophily-enhanced Graph Structure Learning for Graph Clustering
"""
import copy
import gc
import random
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
from utils import sparse_mx_to_torch_sparse_tensor
from utils import torch_sparse_to_dgl_graph


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
        self.pseudo_label = None
        self.hard_mask = None
        self.gsl_embedding = 0
        self.tb_filename = tb_filename
        self.warmup_filename = warmup_filename
        self.udp = udp
        self.reset = reset
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
    def get_fd_loss(z, norm=1):
        norm_ff = z / (z**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(2.0)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = norm * F.cross_entropy(coef_mat, a)
        return L_fd

    def get_cluster_center(self):
        z = self.get_embedding()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _ = kmeans.fit_predict(z.data.cpu().numpy())
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(
            self.device)

    def _re_pretrain(self):
        best_loss = 1e9

        for epoch in range(self.n_pretrain_epochs):
            self.train()

            self.optimizer.zero_grad()
            t = time.time()

            z = self.encoder(self.sm_fea_s)

            index = list(range(len(z)))
            split_list = data_split(index, 10000)

            ins_loss = 0

            for batch in split_list:
                z_batch = z[batch]
                A_batch = torch.FloatTensor(
                    self.lbls[batch, :][:, batch].toarray()).to(self.device)
                preds = self.inner_product_decoder(z_batch).view(-1)

                pos_weight_b = torch.FloatTensor([
                    (float(A_batch.shape[0] * A_batch.shape[0] - A_batch.sum())
                     / A_batch.sum())
                ]).to(self.device)
                norm_weights_b = (A_batch.shape[0] * A_batch.shape[0] / float(
                    (A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) * 2))
                loss = self.bce_loss(preds, A_batch.view(-1), norm_weights_b,
                                     pos_weight_b)
                ins_loss += loss.item()

                (loss).backward(retain_graph=True)
                self.optimizer.step()

            torch.cuda.empty_cache()
            gc.collect()

            print(f"Cluster Epoch: {epoch}, ins_loss={ins_loss}, "
                  f"time={time.time() - t:.5f}")

            if ins_loss < best_loss:
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def _re_ce_train(self, ):
        best_loss = 1e9

        self.get_cluster_center()

        for epoch in range(self.n_epochs):
            self.train()

            if epoch % self.udp == 0 or epoch == self.n_epochs - 1:
                with torch.no_grad():
                    z_detached = self.get_embedding()
                    Q = self.get_Q(z_detached)
                    q = Q.detach().data.cpu().numpy().argmax(1)

            t = time.time()

            z = self.encoder(self.sm_fea_s)

            index = list(range(len(z)))
            split_list = data_split(index, 10000)

            ins_loss = 0

            for batch in split_list:
                z_batch = z[batch]
                A_batch = torch.FloatTensor(
                    self.lbls[batch, :][:, batch].toarray()).to(self.device)
                preds = self.inner_product_decoder(z_batch).view(-1)

                q = self.get_Q(z_batch)
                p = target_distribution(Q[batch].detach())
                kl_loss = F.kl_div(q.log(), p, reduction="batchmean")

                pos_weight_b = torch.FloatTensor([
                    (float(A_batch.shape[0] * A_batch.shape[0] - A_batch.sum())
                     / A_batch.sum())
                ]).to(self.device)
                norm_weights_b = (A_batch.shape[0] * A_batch.shape[0] / float(
                    (A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) * 2))
                loss = self.bce_loss(preds, A_batch.view(-1), norm_weights_b,
                                     pos_weight_b)
                ins_loss += loss.item()

                self.optimizer.zero_grad()
                (loss + kl_loss).backward(retain_graph=True)
                self.optimizer.step()

            print(f"Cluster Epoch: {epoch}, ins_loss={ins_loss},"
                  f"kl_loss={kl_loss.item()},"
                  f"time={time.time() - t:.5f}")

            if ins_loss < best_loss:
                best_loss = ins_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

            torch.cuda.empty_cache()
            gc.collect()

    def get_pseudo_label(self, node_rate=0.2):
        with torch.no_grad():
            z_detached = self.get_embedding()
            Q = self.get_Q(z_detached)
            soft = Q.detach()
        hard = soft.argmax(dim=1).view(-1).cpu().numpy()

        hard_mask = np.array([False for _ in range(len(hard))], dtype=np.bool)
        for c in range(self.n_clusters):
            add_num = int(node_rate * soft.shape[0])

            c_col = soft[:, c].detach().cpu().numpy()
            c_col_idx = c_col.argsort()[::-1][:add_num]
            top_c_idx = c_col_idx
            hard[top_c_idx] = c

            hard_mask[top_c_idx] = True

            print(f"class {c},num={len(top_c_idx)}")

        hard[~hard_mask] = -1
        self.pseudo_label = hard
        self.hard_mask = hard_mask

    def _train(self, node_ratio=0.2):
        self.get_pseudo_label(node_ratio)
        self._re_ce_train()

    def _pretrain(self, ):
        self._re_pretrain()

    def add_edge(self, edges, label, n_nodes, add_edge_rate, z):
        u, v = edges[0].numpy(), edges[1].numpy()
        if add_edge_rate == 0:
            row = u.tolist()
            col = v.tolist()
            data = [1 for _ in range(len(row))]
            adj_csr = sp.csr_matrix((data, (row, col)),
                                    shape=(n_nodes, n_nodes))

            adj_csr[adj_csr > 1] = 1

            adj_csr = eliminate_zeros(adj_csr)

            print(f"after add edge, final edges num={adj_csr.toarray().sum()}")
            return adj_csr

        lbl_np = label
        final_u, final_v = [], []
        for i in np.unique(lbl_np):
            if i == -1:
                continue
            print(f"************ label == {i} ************")
            same_class_bool = np.array(lbl_np == i, dtype=np.bool)
            same_class_idx = np.nonzero(same_class_bool)[0]
            nodes = same_class_idx

            n_nodes_c = len(nodes)
            add_num = int(add_edge_rate * n_nodes_c**2)
            cluster_embeds = z[nodes]

            sims = []
            CHUNK = 1024 * 4

            CTS = len(cluster_embeds) // CHUNK
            if len(cluster_embeds) % CHUNK != 0:
                CTS += 1
            for j in range(CTS):
                a = j * CHUNK
                b = (j + 1) * CHUNK
                b = min(b, len(cluster_embeds))

                cts = torch.matmul(cluster_embeds,
                                   cluster_embeds[a:b].T).view(-1)
                sims.append(cts.cpu())
            sims = torch.cat(sims).cpu()
            sim_top_k = sims.sort(descending=True).indices.numpy()[:add_num]
            xind = sim_top_k // n_nodes_c
            yind = sim_top_k % n_nodes_c

            new_u = nodes[xind]
            new_v = nodes[yind]
            final_u.extend(new_u.tolist())
            final_v.extend(new_v.tolist())

        rw = final_u + final_v
        cl = final_v + final_u
        data0 = [1 for _ in range(len(rw))]
        extra_adj = sp.csr_matrix((data0, (rw, cl)), shape=(n_nodes, n_nodes))

        extra_adj[extra_adj > 1] = 1

        extra_adj = eliminate_zeros(extra_adj)

        row = u.tolist() + final_u + final_v
        col = v.tolist() + final_v + final_u
        data = [1 for _ in range(len(row))]
        adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        adj_csr[adj_csr > 1] = 1

        adj_csr = eliminate_zeros(adj_csr)

        print(f"after add edge,final edges num={adj_csr.sum()}")
        return adj_csr

    def del_edge(self, edges, label, n_nodes, del_edge_rate=0.25):
        u, v = edges[0].numpy(), edges[1].numpy()
        lbl_np = label
        if del_edge_rate == 0:
            row = u.tolist()
            col = v.tolist()
            data = [1 for _ in range(len(row))]
            adj_csr = sp.csr_matrix((data, (row, col)),
                                    shape=(n_nodes, n_nodes))

            adj_csr[adj_csr > 1] = 1

            adj_csr = eliminate_zeros(adj_csr)

            print(f"after del edge, final edges num={adj_csr.sum()}")
            return adj_csr

        inter_class_bool = np.array(
            (lbl_np[u] != -1) & (lbl_np[v] != -1) & (lbl_np[u] != lbl_np[v]),
            dtype=np.bool,
        )
        inter_class_idx = np.nonzero(inter_class_bool)[0]
        inter_class_edge_len = len(inter_class_idx)

        all_other_edges_idx = np.nonzero(~inter_class_bool)[0].tolist()

        np.random.shuffle(inter_class_idx)

        remain_edge_len = inter_class_edge_len - int(
            inter_class_edge_len * del_edge_rate)

        inter_class_idx = inter_class_idx[:remain_edge_len].tolist()
        final_edges_idx = all_other_edges_idx + inter_class_idx

        new_u = u[final_edges_idx].tolist()
        new_v = v[final_edges_idx].tolist()

        row = new_u
        col = new_v
        data = [1 for _ in range(len(row))]
        adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        adj_csr[adj_csr > 1] = 1

        adj_csr = eliminate_zeros(adj_csr)

        print(f"after del edge, final edges num={adj_csr.sum()}")

        return adj_csr

    def update_features(self, adj):
        adj_norm_s = preprocess_graph(
            adj,
            self.n_gnn_layers,
            norm=self.norm,
            renorm=self.renorm,
        )
        adj_csr = adj_norm_s[0]
        sm_fea_s = sp.csr_matrix(self.features).toarray()

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

        self.lbls = adj_csr

    def fit(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        add_edge_ratio=0.04,
        node_ratio=0.2,
        del_edge_ratio=0,
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
        self.labels = labels
        adj = graph.adj_external(scipy_fmt="csr")
        edges = graph.edges()
        n_nodes = self.features.shape[0]

        adj = eliminate_zeros(adj)
        self.adj = adj
        self.to(self.device)

        self.update_features(adj)

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
        self.get_pseudo_label(node_ratio)

        adj_pre = copy.deepcopy(adj)

        for gls_ep in range(gsl_epochs):
            torch.cuda.empty_cache()
            gc.collect()

            print(f"==============GSL epoch:{gls_ep} ===========")
            with torch.no_grad():
                z_detached = self.get_embedding()

            adj_new = self.add_edge(edges, self.pseudo_label, n_nodes,
                                    add_edge_ratio, z_detached)

            graph_new = torch_sparse_to_dgl_graph(
                sparse_mx_to_torch_sparse_tensor(adj_new))
            graph_new.ndata["feat"] = graph.ndata["feat"]
            graph_new.ndata["label"] = graph.ndata["label"]
            graph_new.ndata["emb"] = z_detached.cpu()
            edges = graph_new.edges()

            adj_new = self.del_edge(edges, self.pseudo_label, n_nodes,
                                    del_edge_ratio)

            graph_new = torch_sparse_to_dgl_graph(
                sparse_mx_to_torch_sparse_tensor(adj_new))
            graph_new.ndata["feat"] = graph.ndata["feat"]
            graph_new.ndata["label"] = graph.ndata["label"]
            graph_new.ndata["emb"] = z_detached.cpu()
            edges = graph_new.edges()

            self.update_features(adj=adj_new)
            self._train(node_ratio)

            if self.reset:
                self.reset_weights()
                self.to(self.device)

    def get_embedding(self):
        with torch.no_grad():
            mu = self.best_model.encoder(self.sm_fea_s)
        return mu.detach()

    def get_Q(self, z):
        """get soft clustering assignment distribution

        Args:
            z (torch.Tensor): node embedding

        Returns:
            torch.Tensor: Soft assignments
        """
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.cpu().unsqueeze(1) - self.cluster_layer.cpu(), 2), 2) /
                   1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def data_split(full_list, n_sample):
    offset = n_sample
    random.shuffle(full_list)
    len_all = len(full_list)
    index_now = 0
    split_list = []
    while index_now < len_all:
        if index_now + offset > len_all:
            split_list.append(full_list[index_now:len_all])
        else:
            split_list.append(full_list[index_now:index_now + offset])
        index_now += offset
    return split_list


def target_distribution(q):
    """get target distribution P

    Args:
        q (torch.Tensor): Soft assignments

    Returns:
        torch.Tensor: target distribution P
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
