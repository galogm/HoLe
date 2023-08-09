import csv
import os
import random
import time
from pathlib import Path
from pathlib import PurePath
from typing import Tuple

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


def sk_clustering(
    X: torch.Tensor,
    n_clusters: int,
    name: str = "kmeans",
) -> np.ndarray:
    """sklearn clustering.

    Args:
        X (torch.Tensor): data embeddings.
        n_clusters (int): num of clusters.
        name (str, optional): type name. Defaults to 'kmeans'.

    Raises:
        NotImplementedError: clustering method not implemented.

    Returns:
        np.ndarray: cluster assignments.
    """
    if name == "kmeans":
        model = KMeans(n_clusters=n_clusters)
        label_pred = model.fit(X).labels_
        return label_pred

    if name == "spectral":
        model = SpectralClustering(n_clusters=n_clusters,
                                   affinity="precomputed")
        label_pred = model.fit(X).labels_
        return label_pred

    raise NotImplementedError


def make_parent_dirs(target_path: PurePath) -> None:
    """make all the parent dirs of the target path.

    Args:
        target_path (PurePath): target path.
    """
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)


def refresh_file(target_path: str = None) -> None:
    """clear target path

    Args:
        target_path (str): file path
    """
    if target_path is not None:
        target_path: PurePath = Path(target_path)
        if target_path.exists():
            target_path.unlink()

        make_parent_dirs(target_path)
        target_path.touch()


def save_model(
    model_filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_epoch: int,
    loss: float,
) -> None:
    """Save model, optimizer, current_epoch, loss to ``.checkpoints/${model_filename}.pt``.

    Args:
        model_filename (str): filename to save model.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.
        current_epoch (int): current epoch.
        loss (float): loss.
    """
    model_path = get_modelfile_path(model_filename)
    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        model_path,
    )


def load_model(
    model_filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """Load model from ``.checkpoints/${model_filename}.pt``.

    Args:
        model_filename (str): filename to load model.
        model (torch.nn.Module): model.
        optimizer (torch.optim.Optimizer): optimizer.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
            [model, optimizer, epoch, loss]
    """

    model_path = get_modelfile_path(model_filename)
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss


def get_modelfile_path(model_filename: str) -> PurePath:
    model_path: PurePath = Path(f".checkpoints/{model_filename}.pt")
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    return model_path


def check_modelfile_exists(model_filename: str) -> bool:
    if get_modelfile_path(model_filename).exists():
        return True
    return False


def get_str_time():
    output_file = "time_" + time.strftime("%m%d%H%M%S", time.localtime())
    return output_file


def node_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    """Calculate node homophily.

    Args:
        adj (sp.spmatrix): adjacent matrix.
        labels (torch.Tensor): labels.

    Returns:
        float: node homophily.
    """
    adj_coo = adj.tocoo()
    adj_coo.data = ((
        labels[adj_coo.col] == labels[adj_coo.row]).cpu().numpy().astype(int))
    return (np.asarray(adj_coo.sum(1)).flatten() /
            np.asarray(adj.sum(1)).flatten()).mean()


def edge_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    """Calculate edge homophily.

    Args:
        adj (sp.spmatrix): adjacent matrix.
        labels (torch.Tensor): labels.

    Returns:
        float: edge homophily.
    """
    return (
        (labels[adj.tocoo().col] == labels[adj.tocoo().row]).cpu().numpy() *
        adj.data).sum() / adj.sum()


def eliminate_zeros(adj: sp.spmatrix) -> sp.spmatrix:
    """Remove self-loops and edges with value of zero.

    Args:
        adj (sp.spmatrix): adjacent matrix.

    Returns:
        sp.spmatrix: adjacent matrix.
    """
    adj = adj - sp.dia_matrix(
        (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    return adj


def csv2file(
    target_path: str,
    thead: Tuple[str] = None,
    tbody: Tuple = None,
    refresh: bool = False,
    is_dict: bool = False,
) -> None:
    """save csv to target_path

    Args:
        target_path (str): target path
        thead (Tuple[str], optional): csv table header, only written into the file when\
            it is not None and file is empty. Defaults to None.
        tbody (Tuple, optional): csv table content. Defaults to None.
        refresh (bool, optional): whether to clean the file first. Defaults to False.
    """
    target_path: PurePath = Path(target_path)
    if refresh:
        refresh_file(target_path)

    make_parent_dirs(target_path)

    with open(target_path, "a+", newline="", encoding="utf-8") as csvfile:
        csv_write = csv.writer(csvfile)
        if os.stat(target_path).st_size == 0 and thead is not None:
            csv_write.writerow(thead)
        if tbody is not None:
            if is_dict:
                dict_writer = csv.DictWriter(csvfile,
                                             fieldnames=tbody[0].keys())
                for elem in tbody:
                    dict_writer.writerow(elem)
            else:
                csv_write.writerow(tbody)


def set_seed(seed: int = 4096) -> None:
    """Set random seed.

    NOTE:!!! conv and neighborSampler of dgl is somehow nondeterministic !!!

    Set according to the pytorch doc: https://pytorch.org/docs/1.9.0/notes/randomness.html
    cudatoolkit doc: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    dgl issue: https://github.com/dmlc/dgl/issues/3302

    Args:
        seed (int, optional): random seed. Defaults to 4096.
    """
    if seed is not False:
        os.environ["PYTHONHASHSEED"] = str(seed)
        # required by torch: Deterministic behavior was enabled with either
        # `torch.use_deterministic_algorithms(True)` or
        # `at::Context::setDeterministicAlgorithms(true)`,
        # but this operation is not deterministic because it uses CuBLAS and you have
        # CUDA >= 10.2. To enable deterministic behavior in this case,
        # you must set an environment variable before running your PyTorch application:
        # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
        # For more information, go to
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        # NOTE: dgl.seed will occupy cuda:0 no matter which gpu is set if seed is set before device
        # see the issue：https://github.com/dmlc/dgl/issues/3054
        dgl.seed(seed)


def set_device(gpu: str = "0") -> torch.device:
    """Set torch device.

    Args:
        gpu (str): args.gpu. Defaults to '0'.

    Returns:
        torch.device: torch device. `device(type='cuda: x')` or `device(type='cpu')`.
    """
    max_device = torch.cuda.device_count() - 1
    if gpu == "none":
        print("Use CPU.")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if not gpu.isnumeric():
            raise ValueError(
                f"args.gpu:{gpu} is not a single number for gpu setting."
                f"Multiple GPUs parallelism is not supported.")

        if int(gpu) <= max_device:
            print(f"GPU available. Use cuda:{gpu}.")
            device = torch.device(f"cuda:{gpu}")
            torch.cuda.set_device(device)
        else:
            print(
                f"cuda:{gpu} is not in available devices [0, {max_device}]. Use CPU instead."
            )
            device = torch.device("cpu")
    else:
        print("GPU is not available. Use CPU instead.")
        device = torch.device("cpu")
    return device


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor

    Args:
        sparse_mx (<class 'scipy.sparse'>): sparse matrix

    Returns:
        (torch.Tensor): torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    """Convert a torch sparse tensor matrix to dgl graph

    Args:
        torch_sparse_mx (torch.Tensor): torch sparse tensor

    Returns:
        (dgl.graph): dgl graph
    """
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0])
    dgl_graph.edata["w"] = values.detach()
    return dgl_graph
