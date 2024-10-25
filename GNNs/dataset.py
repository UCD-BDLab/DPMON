import re

import torch
import pandas as pd
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json

from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
from torch_geometric.data import Data
from utils import precompute_dist_data

## adjust path to work indepently
DATASET_DIR = (
    "C://Users//ramosv//Desktop//NetCo//BioInformedSubjectRepresentation//GNNs"
)

# DATASET_DIR = 'C:\\Users\\13033\\Desktop\\CU Denver\\NetCO\\GNNs\\data\\COPD\\SparsifiedNetworks'


def get_tg_dataset(args, dataset_name, use_cache=True):
    # try:
    dataset = load_tg_dataset(dataset_name)
    # except Exception as E:
    #     print("Error while Attempting to Read Dataset [%s]" % E)

    # TODO: Review this Code for PGNN (Where Shortest Path & Dists are Needed
    # Precompute the Shortest Path
    if not os.path.isdir("datasets"):
        os.mkdir("datasets")
    if not os.path.isdir("datasets/cache"):
        os.mkdir("datasets/cache")
    f1_name = "datasets/cache/" + dataset_name + str(args.approximate) + "_dists.dat"
    f2_name = (
        "datasets/cache/" + dataset_name + str(args.approximate) + "_dists_removed.dat"
    )

    if use_cache and os.path.isfile(f1_name):
        with open(f1_name, "rb") as f1:
            dists_list = pickle.load(f1)
        # print("Distances List in Case of Using Cachce %s" % dists_list)
        dataset.dists = torch.from_numpy(dists_list).float()

        # for i, data in enumerate(dataset):
        #     data.dists = torch.from_numpy(dists_list[i]).float()
        #     data_list.append(data)
    else:
        dists_list = []
        dists = precompute_dist_data(
            dataset.edge_index.numpy(), dataset.edge_attr.numpy(), dataset.num_nodes
        )
        # print("Distances after Precompute %s" % dists)
        dists_list.append(dists)
        dataset.dists = torch.from_numpy(dists).float()

        with open(f1_name, "wb") as f1, open(f2_name, "wb") as f2:
            pickle.dump(dists_list, f1)
    return dataset


def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    from torch_geometric.utils import add_remaining_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    fill_value = 1.0 if not improved else 2.0
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )

    row, col = edge_index
    # print("Row %s\nCol %s" %(row, col))
    # print("Edge Weight from Inside the Normalization Function %s" % edge_weight)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    # print("Deg %s" % deg)
    deg_inv_sqrt = deg.pow(-0.5)
    # print("Deg Inverse Sqrt %s" % deg_inv_sqrt)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    # print("After Adjust %s" % deg_inv_sqrt)

    # print("Deg Inverst Sqrt Row %s" % deg_inv_sqrt[row])
    # print("Deg Inverst Sqrt Col %s" % deg_inv_sqrt[col])
    # print("Returned Value %s" % deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col])
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def nx_to_tg_data(graph, features, nodes_labels):
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # TODO: Investigate if you can Increase the Resolution of Features (For now it's 4 Decimal Values)
    x = np.zeros(features.shape)
    graph_nodes = list(graph.nodes)
    for m in range(features.shape[0]):
        x[graph_nodes[m]] = features[m]
    x = torch.from_numpy(x).float()

    # Edges Indexes
    edge_index = np.array(list(graph.edges))
    edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
    edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

    # Edges Weights
    # TODO: Need to understand if edges weights are used as part of the adjacency matrix or not!
    edge_weight = np.array(list(nx.get_edge_attributes(graph, "weight").values()))
    # edge_weight = np.array([0.17, 0.55, 0.56, 0.20, 0.52, 0.71, 0.43, 0.73, 0.62, 0.25, 0.18, 0.08, 0.01,  0.47, 0.77,
    #                          0.65, 0.33, 0.86, 0.92, 0.57, 0.82, 0.26, 0.02, 0.62, 0.19, 0.05, 0.26, 1., 0.71, 0.66,
    #                          0.67, 0.75, 0.90, 0.83, 0.65, 0.96, 0.740, 0.67, 0.64])
    edge_weight = np.concatenate((edge_weight, edge_weight), axis=0)
    edge_weight = torch.from_numpy(edge_weight).float()

    # print("***** My Data ******")
    # print("Edge Index %s" % edge_index)
    # print("Edge Weight %s" % edge_weight)

    # TODO: Augmenting the Node Identity!?
    def identity_fun(graph, **kwargs):
        if "feature_dim" not in kwargs:
            raise ValueError("Argument feature_dim not supplied")
        return compute_identity(
            graph.edge_index, graph.num_nodes, kwargs["feature_dim"]
        )

    def compute_identity(edge_index, n, k):
        # print("Augmenting the Node Identity as a Feature")
        n = len(graph.nodes)
        k = 10
        # Normalizing Edg Weight by the Degree of Nodes Connected by that Edge
        id, value = norm(edge_index=edge_index, num_nodes=n, edge_weight=edge_weight)
        # print("ID %s\nValue%s" % (id, value))
        adj_sparse = torch.sparse.FloatTensor(id, value, torch.Size([n, n]))
        # print("Adjacency Sparse %s" % adj_sparse)
        adj = adj_sparse.to_dense()
        # print("Adjacency Dense %s" % adj)
        diag_all = [torch.diag(adj)]
        # print("Initial Diag All %s" % diag_all)
        adj_power = adj
        # print("Adj Power Before %s" % adj_power)
        for i in range(1, k):
            # print("**********************i %s" % i)
            adj_power = adj_power @ adj
            # print("Adj Power %s" % adj_power)
            diag_all.append(torch.diag(adj_power))
        diag_all = torch.stack(diag_all, dim=1)
        # print("Diag All %s" % diag_all)
        return diag_all

    # TODO: Need to Understand The Difference based on task_type
    # if as_label:
    #     repr_method = 'balanced' if 'classification' in \
    #                                 cfg.dataset.task_type else 'original'
    # else:
    #     repr_method = cfg.dataset.augment_feature_repr
    # TODO: Maybe in the Future for Comparison, Conside Different Augmentation Techniques to See if it Makes any Difference

    feature_dict = {
        # 'node_degree': degree_fun,
        # 'node_betweenness_centrality': centrality_fun,
        # 'node_path_len': path_len_fun,
        # 'node_pagerank': pagerank_fun,
        # 'node_clustering_coefficient': clustering_coefficient_fun,
        "node_identity": identity_fun,
        # 'node_const': const_fun,
        # 'node_onehot': onehot_fun,
        # 'edge_path_len': edge_path_len_fun,
        # 'graph_laplacian_spectrum': graph_laplacian_spectrum_fun,
        # 'graph_path_len': graph_path_len_fun,
        # 'graph_clustering_coefficient': graph_clustering_fun
    }

    # print("Original Features Before Augmenting %s" % features)
    # TODO: Need to Understand this feature_dims (This is the Augment Feature Dims)
    # feature_dims = 10
    # actual_feat_dims = []
    # for key, dim in zip(features, feature_dims):
    #     print("Key %s\t Dims %s" % (key, dim))
    #     feat_fun = feature_dict[key]
    #     # key = key + '_label' if as_label else key
    #     if key not in dataset[0]:
    #         # compute (raw) features
    #         dataset.apply_transform(feat_fun,
    #                                 update_graph=False,
    #                                 update_tensor=False,
    #                                 as_label=as_label,
    #                                 feature_dim=dim)
    #         # feat = dataset[0][key]
    #         if repr_method == 'original':
    #             # use the original feature as is
    #             # this ignores the specified config feature_dims
    #             dataset.apply_transform(FeatureAugment._orig_features,
    #                                     update_graph=True,
    #                                     update_tensor=False,
    #                                     key=key)
    #         elif repr_method == 'position':
    #             # positional encoding similar to that of transformer
    #             scale = dim / 2 / FeatureAugment._get_max_value(
    #                 dataset, key)
    #             dataset.apply_transform(FeatureAugment._position_features,
    #                                     update_graph=True,
    #                                     update_tensor=False,
    #                                     key=key,
    #                                     feature_dim=dim,
    #                                     scale=scale)
    #         else:
    #             # Bin edges for one-hot (repr_method = balanced, bounded,
    #             # equal_width)
    #             # use all features in dataset for computing bin edges
    #             bin_edges = self._get_bin_edges(dataset, key, dim,
    #                                             repr_method)
    #             # apply binning
    #             dataset.apply_transform(FeatureAugment._bin_features,
    #                                     update_graph=True,
    #                                     update_tensor=False,
    #                                     key=key,
    #                                     bin_edges=bin_edges,
    #                                     feature_dim=len(bin_edges),
    #                                     as_label=as_label)
    #     actual_feat_dims.append(dataset[0].get_num_dims(key,
    #                                                     as_label=as_label))

    node_identity = compute_identity(edge_index, len(graph.nodes), 10)
    # TODO: Need to Fix the Size (It's not Static (3))
    new_features = np.zeros((len(graph.nodes), len(features[0]) + 10))
    # print("**** Node Identity %s" % node_identity)
    for idx, feature in enumerate(features):
        # print("Feature %s and Node Identity %s" % (feature, np.array(node_identity[idx])))
        # print("Appended %s" % np.append(feature, np.array(node_identity[idx])))
        new_features[idx] = np.append(feature, np.array(node_identity[idx]))
    # print("Features After Augmenting %s" % new_features)

    # TODO: Investigate if you can Increase the Resolution of Features (For now it's 4 Decimal Values)
    # x = np.zeros(new_features.shape)
    # graph_nodes = list(graph.nodes)
    # for m in range(new_features.shape[0]):
    #     x[graph_nodes[m]] = new_features[m]
    # x = torch.from_numpy(x).float()

    nodes_labels = torch.from_numpy(np.array(nodes_labels)).float()
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=nodes_labels)

    return data


def load_graph(dataset_str):
    graph_adj_file = os.path.join(DATASET_DIR, dataset_str)
    print("Loading graph from {}".format(graph_adj_file))
    graph_adj = pd.read_csv(graph_adj_file, index_col=0).to_numpy()
    nodes_names = pd.read_csv(graph_adj_file, index_col=0).index.tolist()

    # if 'fev1' in dataset_str:
    if True:
        # fev1_X_R.csv is a Reduced Version of the Original Dataset with 18 Features
        # original_dataset = pd.read_csv(os.path.join(DATASET_DIR, 'fev1_X_R.csv'), index_col=0).reset_index(drop='index')
        original_dataset = pd.read_csv(
            os.path.join(DATASET_DIR, "fev1_X.csv"), index_col=0
        ).reset_index(drop="index")
        dataset_associated_phenotype = pd.read_csv(
            os.path.join(DATASET_DIR, "fev1_Y.csv"), index_col=0
        ).reset_index(drop="index")

        nodes_correlation_with_phenotype = original_dataset.corrwith(
            dataset_associated_phenotype["FEV1pp_utah"]
        )
        complete_original_dataset = pd.read_csv(
            os.path.join(DATASET_DIR, "fev1_clinical_variables.csv"), index_col=0
        ).reset_index(drop="index")

        # Names of Clinical Variables
        # clinical_variables_cols = ['gender', 'age_visit', 'race', 'smoking_status', 'BMI', 'Waist_CM',
        #                            'Chronic_Bronchitis', 'PRM_pct_airtrapping_Thirona', 'PRM_pct_emphysema_Thirona',
        #                            'PRM_pct_normal_Thirona', 'Pi10_Thirona', 'AWT_seg_Thirona', 'WallAreaPct_seg_Thirona',
        #                            'DLco_GLI_tr_pp', 'FRC_TLC_ratio_Thirona', 'distwalked', 'SF36_PCS_score',
        #                            'SF36_MCS_score', 'comorbidities']

        clinical_variables_cols = [
            "gender",
            "age_visit",
            "Chronic_Bronchitis",
            "PRM_pct_emphysema_Thirona",
            "PRM_pct_normal_Thirona",
            "Pi10_Thirona",
            "comorbidities",
        ]
    else:
        raise NotImplementedError

    # graph = nx.from_numpy_matrix(graph_adj)
    graph = nx.from_numpy_array(graph_adj)

    nodes_features = []
    for node_name in nodes_names:
        node_features = []
        for clinical_variable in clinical_variables_cols:
            node_features.append(
                abs(
                    original_dataset[node_name].corr(
                        complete_original_dataset[clinical_variable].astype("float64")
                    )
                )
            )
        nodes_features.append(node_features)

    features = np.array(nodes_features)
    # print("Nodes Features %s" % nodes_features)
    # print("Features before Zero %s" % features)
    # features = np.zeros(features.shape)
    # print("Features after Zero %s" % features)
    # Using correlation as Node Labels
    nodes_labels = [abs(x) for x in nodes_correlation_with_phenotype.values.tolist()]

    # Using mean_shap_value as Node Labels
    mean_shap_value_dic = {
        "Troponin T": 3.1540397551414956,
        "Growth/differentiation factor 15": 1.9930830586144692,
        "C-reactive protein": 1.763424268170919,
        "Alpha-(1,3)-fucosyltransferase 5": 1.679293941860064,
        "Apolipoprotein A-I": 1.3323696457951508,
        "Kallistatin": 1.2860353083707738,
        "phosphocholine": 1.2688204892023618,
        "Trefoil factor 3": 1.2650743554639183,
        "(N(1) + N(8))-acetylspermidine": 1.2500571559436615,
        "Protein S100-A4": 1.1766156392606308,
        "adrenate (22:4n6)": 1.1621175408547413,
        "myristoleoylcarnitine (C14:1)*": 1.153006025513969,
        "Complement component C9": 1.1464053614768615,
        "5-acetylamino-6-amino-3-methyluracil": 1.1222009354180207,
        "Carbonic anhydrase 6": 1.1206090344978248,
        "X - 12026": 1.1185282545919488,
        "ergothioneine": 1.107927620690626,
        "5-hydroxyhexanoate": 1.0314057804500296,
        "Epidermal growth factor receptor": 1.0220716192448691,
        "C-C motif chemokine 14": 1.0036318628240863,
        "N-terminal pro-BNP": 0.9953282553330273,
        "N2,N2-dimethylguanosine": 0.9666407989495159,
        "Beta-2-microglobulin": 0.9265955080614319,
        "C-glycosyltryptophan": 0.8626149019726225,
        "Tumor necrosis factor receptor superfamily member 1A": 0.8601066752634039,
        "Cystatin-C": 0.8167309052019965,
        "X - 12117": 0.799549589625069,
    }

    nodes_labels = []
    for node_name in nodes_names:
        nodes_labels.append(mean_shap_value_dic[node_name])

    # NEW: Attempting to Use Nodes Importance (based on SHAP Values) as a Label
    # features_ordered = ['TroponinT', 'Growthdifferentiationfactor15', 'Creactiveprotein', 'Alpha13fucosyltransferase5',
    #                     'ApolipoproteinAI', 'Kallistatin', 'phosphocholine', 'Trefoilfactor3', 'N1N8acetylspermidine',
    #                     'ProteinS100A4', 'adrenate224n6', 'myristoleoylcarnitineC141', 'ComplementcomponentC9',
    #                     '5acetylamino6amino3methyluracil', 'Carbonicanhydrase6', 'X12026', 'ergothioneine',
    #                     '5hydroxyhexanoate', 'Epidermalgrowthfactorreceptor', 'CCmotifchemokine14', 'NterminalproBNP',
    #                     'N2N2dimethylguanosine', 'Beta2microglobulin', 'Cglycosyltryptophan',
    #                     'Tumornecrosisfactorreceptorsuperfamilymember1A', 'CystatinC', 'X12117']
    #
    # features_importance = {feature: 28 - rank for rank, feature in enumerate(features_ordered, start=1)}

    # nodes_labels = []
    # for node_name in nodes_names:
    #     node_name = re.sub('[^A-Za-z0-9_]+', '', node_name)
    #     # print("Node Name after update %s" % node_name)
    #     nodes_labels.append(features_importance.get(node_name))

    print(nodes_labels)

    # print("Nodes Labels %s" % nodes_labels)
    # TODO: Consider normalization when working with features
    # features_all = (features_all - np.mean(features_all, axis=-1, keepdims=True)) / np.std(features_all, axis=-1, keepdims=True)
    return graph, features, nodes_labels


def load_tg_dataset(name):
    graph, features, nodes_labels = load_graph(name)
    return nx_to_tg_data(graph, features, nodes_labels)
