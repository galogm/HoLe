"""Main
"""
# pylint: disable=line-too-long,invalid-name,
import argparse
import random

import scipy.sparse as sp
import torch
from graph_datasets import load_data

from models import HoLe
from models import HoLe_batch
from utils import check_modelfile_exists
from utils import csv2file
from utils import evaluation
from utils import get_str_time
from utils import set_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="HoLe",
        description=
        "Homophily-enhanced Structure Learning for Graph Clustering",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset used in the experiment",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    args = parser.parse_args()

    final_params = {}

    dim = 500
    n_lin_layers = 1
    dump = True
    device = set_device(str(args.gpu_id))
    lr = {
        "Cora": 0.001,
        "Citeseer": 0.001,
        "ACM": 0.001,
        "Pubmed": 0.001,
        "BlogCatalog": 0.001,
        "Flickr": 0.001,
        "Reddit": 2e-5,
    }
    n_gnn_layers = {
        "Cora": [8],
        "Citeseer": [3],
        "ACM": [3],
        "Pubmed": [35],
        "BlogCatalog": [1],
        "Flickr": [1],
        "Reddit": [3],
    }
    pre_epochs = {
        "Cora": [150],
        "Citeseer": [150],
        "ACM": [200],
        "Pubmed": [50],
        "BlogCatalog": [150],
        "Flickr": [300],
        "Reddit": [3],
    }
    epochs = {
        "Cora": 50,
        "Citeseer": 150,
        "ACM": 150,
        "Pubmed": 200,
        "BlogCatalog": 150,
        "Flickr": 150,
        "Squirrel": 150,
        "Reddit": 3,
    }
    inner_act = {
        "Cora": lambda x: x,
        "Citeseer": torch.sigmoid,
        "ACM": lambda x: x,
        "Pubmed": lambda x: x,
        "BlogCatalog": lambda x: x,
        "Flickr": lambda x: x,
        "Squirrel": lambda x: x,
        "Reddit": lambda x: x,
    }
    udp = {
        "Cora": 10,
        "Citeseer": 40,
        "ACM": 40,
        "Pubmed": 10,
        "BlogCatalog": 40,
        "Flickr": 40,
        "Squirrel": 40,
        "Reddit": 40,
    }
    node_ratios = {
        "Cora": [1],
        "Citeseer": [0.3],
        "ACM": [0.3],
        "Pubmed": [0.5],
        "BlogCatalog": [1],
        "Flickr": [0.3],
        "Squirrel": [0.3],
        "Reddit": [0.01],
    }
    add_edge_ratio = {
        "Cora": 0.5,
        "Citeseer": 0.5,
        "ACM": 0.5,
        "Pubmed": 0.5,
        "BlogCatalog": 0.5,
        "Flickr": 0.5,
        "Reddit": 0.005,
    }
    del_edge_ratios = {
        "Cora": [0.01],
        "Citeseer": [0.005],
        "ACM": [0.005],
        "Pubmed": [0.005],
        "BlogCatalog": [0.005],
        "Flickr": [0.005],
        "Reddit": [0.02],
    }
    gsl_epochs_list = {
        "Cora": [5],
        "Citeseer": [5],
        "ACM": [10],
        "Pubmed": [3],
        "BlogCatalog": [10],
        "Flickr": [10],
        "Reddit": [1],
    }
    regularization = {
        "Cora": 1,
        "Citeseer": 0,
        "ACM": 0,
        "Pubmed": 1,
        "BlogCatalog": 0,
        "Flickr": 0,
        "Reddit": 0,
    }
    source = {
        "Cora": "dgl",
        "Citeseer": "dgl",
        "ACM": "sdcn",
        "Pubmed": "dgl",
        "BlogCatalog": "cola",
        "Flickr": "cola",
        "Reddit": "dgl",
    }

    datasets = [args.dataset]

    for ds in datasets:
        if ds == "Reddit":
            hole = HoLe_batch
        else:
            hole = HoLe

        for gsl_epochs in gsl_epochs_list[ds]:
            runs = 10

            for n_gnn_layer in n_gnn_layers[ds]:
                for pre_epoch in pre_epochs[ds]:
                    for del_edge_ratio in del_edge_ratios[ds]:
                        for node_ratio in node_ratios[ds]:
                            final_params["dim"] = dim
                            final_params["n_gnn_layers"] = n_gnn_layer
                            final_params["n_lin_layers"] = n_lin_layers
                            final_params["lr"] = lr[ds]
                            final_params["pre_epochs"] = pre_epoch
                            final_params["epochs"] = epochs[ds]
                            final_params["udp"] = udp[ds]
                            final_params["inner_act"] = inner_act[ds]
                            final_params["add_edge_ratio"] = add_edge_ratio[ds]
                            final_params["node_ratio"] = node_ratio
                            final_params["del_edge_ratio"] = del_edge_ratio
                            final_params["gsl_epochs"] = gsl_epochs

                            time_name = get_str_time()
                            save_file = f"results/hole/hole_{ds}_gnn_{n_gnn_layer}_gsl_{gsl_epochs}_{time_name[:9]}.csv"

                            graph, labels, n_clusters = load_data(
                                dataset_name=ds,
                                source=source[ds],
                                verbosity=2,
                            )
                            features = graph.ndata["feat"]
                            if ds in ("Cora", "Pubmed"):
                                graph.ndata["feat"][(features -
                                                     0.0) > 0.0] = 1.0
                            adj_csr = graph.adj_external(scipy_fmt="csr")
                            adj_sum_raw = adj_csr.sum()

                            edges = graph.edges()
                            features_lil = sp.lil_matrix(features)

                            final_params["dataset"] = ds

                            warmup_filename = f"hole_{ds}_run_gnn_{n_gnn_layer}"

                            if not check_modelfile_exists(warmup_filename):
                                print("warmup first")
                                model = hole(
                                    hidden_units=[dim],
                                    in_feats=features.shape[1],
                                    n_clusters=n_clusters,
                                    n_gnn_layers=n_gnn_layer,
                                    n_lin_layers=n_lin_layers,
                                    lr=lr[ds],
                                    n_pretrain_epochs=pre_epoch,
                                    n_epochs=epochs[ds],
                                    norm="sym",
                                    renorm=True,
                                    tb_filename=
                                    f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                    warmup_filename=warmup_filename,
                                    inner_act=inner_act[ds],
                                    udp=udp[ds],
                                    regularization=regularization[ds],
                                )

                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio[ds],
                                    node_ratio=node_ratio,
                                    del_edge_ratio=del_edge_ratio,
                                    gsl_epochs=0,
                                    labels=labels,
                                    adj_sum_raw=adj_sum_raw,
                                    load=False,
                                    dump=dump,
                                )

                            seed_list = [
                                random.randint(0, 999999) for _ in range(runs)
                            ]
                            for run_id in range(runs):
                                final_params["run_id"] = run_id
                                seed = seed_list[run_id]
                                final_params["seed"] = seed
                                # Citeseer needs reset to overcome over-fitting
                                reset = ds == "Citeseer"

                                model = hole(
                                    hidden_units=[dim],
                                    in_feats=features.shape[1],
                                    n_clusters=n_clusters,
                                    n_gnn_layers=n_gnn_layer,
                                    n_lin_layers=n_lin_layers,
                                    lr=lr[ds],
                                    n_pretrain_epochs=pre_epoch,
                                    n_epochs=epochs[ds],
                                    norm="sym",
                                    renorm=True,
                                    tb_filename=
                                    f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_gsl_{gsl_epochs}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                    warmup_filename=warmup_filename,
                                    inner_act=inner_act[ds],
                                    udp=udp[ds],
                                    reset=reset,
                                    regularization=regularization[ds],
                                    seed=seed,
                                )

                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio[ds],
                                    node_ratio=node_ratio,
                                    del_edge_ratio=del_edge_ratio,
                                    gsl_epochs=gsl_epochs,
                                    labels=labels,
                                    adj_sum_raw=adj_sum_raw,
                                    load=True,
                                    dump=dump,
                                )

                                with torch.no_grad():
                                    z_detached = model.get_embedding()
                                    Q = model.get_Q(z_detached)
                                    q = Q.detach().data.cpu().numpy().argmax(1)
                                (
                                    ARI_score,
                                    NMI_score,
                                    AMI_score,
                                    ACC_score,
                                    Micro_F1_score,
                                    Macro_F1_score,
                                    purity,
                                ) = evaluation(labels, q)

                                print("\n"
                                      f"ARI:{ARI_score}\n"
                                      f"NMI:{ NMI_score}\n"
                                      f"AMI:{ AMI_score}\n"
                                      f"ACC:{ACC_score}\n"
                                      f"Micro F1:{Micro_F1_score}\n"
                                      f"Macro F1:{Macro_F1_score}\n"
                                      f"purity_score:{purity}\n")

                                final_params["qARI"] = ARI_score
                                final_params["qNMI"] = NMI_score
                                final_params["qACC"] = ACC_score
                                final_params["qMicroF1"] = Micro_F1_score
                                final_params["qMacroF1"] = Macro_F1_score
                                final_params["qPurity"] = Macro_F1_score

                                if save_file is not None:
                                    csv2file(
                                        target_path=save_file,
                                        thead=list(final_params.keys()),
                                        tbody=list(final_params.values()),
                                    )
                                    print(f"write to {save_file}")

                                print(final_params)
