import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as pca
import matplotlib.pyplot as plt


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        # dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [
        pool.apply_async(
            single_source_shortest_path_length_range,
            args=(
                graph,
                nodes[
                    int(len(nodes) / num_workers * i) : int(
                        len(nodes) / num_workers * (i + 1)
                    )
                ],
                cutoff,
            ),
        )
        for i in range(num_workers)
    ]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, edge_weight, num_nodes):
    """
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    """
    # TODO: Fix the computation of the distance (need to add weights of edges).
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()

    for i, edge in enumerate(edge_list):
        edge.append(edge_weight[i])

    graph.add_weighted_edges_from(edge_list)
    n = num_nodes
    dists_array = np.zeros((n, n))

    dists_dict = all_pairs_shortest_path_length_parallel(graph)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                dists_array[node_i, node_j] = 1 / (dist + 1)
    # TODO: Double Check if you need to consider weights of edges! Because we want position, we shouldn't care about weights
    # maybe the inverse of the weight? or not consider weights at all? Need to check all options here
    return dists_array


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device="cpu"):

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num // anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2 ** (i + 1) - 1
        anchors = np.random.choice(
            data.num_nodes,
            size=(layer_num, anchor_num_per_size, anchor_size),
            replace=True,
        )
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros(
        (layer_num, anchor_num, data.num_nodes), dtype=int
    )

    anchorset_id = get_random_anchorset(data.num_nodes, c=1)
    # print("Anchor Set ID %s" % anchorset_id)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
    # print("Distance Matrix %s\n" % data.dists_max)


# you can write code to reduce the dimensionality of the embeddings and visualize them in 2 dimensions.
# Ideally, we'd want the nodes in the graph with similar labels to have similar embeddings.
def reduce_dimensionality(data):

    # Implement PCA to reduce dimensionality
    # Fist Scale the data

    # Split the data into X (features) and Y (class)
    X = data.iloc[:, 0 : len(data.columns) - 1].values
    y = data.iloc[:, len(data.columns)].values

    # Standard Scaler function substracts the mean and divides it by the Std
    X_std = StandardScaler().fit_transform(X)

    # Construct the Coveriance Matrix of X
    cov_mat = np.cov(X_std.T)
    print(f"Covariance Matrix: {cov_mat}")

    # Perform Eigenvalue,Eigenvector Decomposition of the Covariance Matrix
    L, W = np.linalg.eig(cov_mat)
    # L = Lambda = the eigenvalues: L[0], L[1], L[2], ...
    # W = the eigenvectors,
    # W[:,0] is the eigen vector corresponding to the eigenvalue L[0]
    # W[:,1] is the eigen vector corresponding to the eigenvalue L[1]
    # ...
    # and so on.

    print("Eigenvectors \n%s" % W)
    print("\nEigenvalues \n%s" % L)

    # Make a list of (eigenvalue, eigenvector) tuples

    eig_pairs = [(np.abs(L[i]), W[:, i]) for i in range(len(L))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print("Eigenvalues in descending order:")
    for i in eig_pairs:
        print(i[0])

    # Next compute the explained variance

    tot = sum(L)
    var_exp = [(i / tot) * 100 for i in sorted(L, reverse=True)]
    var_exp

    plt.figure()
    plt.bar(np.arange(4), var_exp)
    plt.xticks(np.arange(4), ("PC1", "PC2", "PC3", "PC4"))
    plt.show()

    # Constructing the Projection Matrix
    # It will be used to transform the data onto the new feature subspace.
    eig_pairs[0][0]  # eigen value of PC1
    eig_pairs[0][1]  # eigenvector of PC1
    eig_pairs[1][0]  # eigenvalue of PC2
    eig_pairs[1][1]  # eigenvector of PC2

    # Projection Matrix, P
    P = np.hstack((eig_pairs[0][1].reshape(4, 1), (eig_pairs[1][1].reshape(4, 1))))
    print(W)
    print(P)  # 4x2 dim projection matrix

    # Now project dataset onto the new feature space through the projection matrix
    # New_X = XP
    New_X = X_std.dot(P)
    X_std.shape
    New_X.shape
    New_X
    # New_X should have reduced the dimensionality of the dataset

    # OR just use sklearn
    # sklearn_pca = pca(n_components=2)
    # X_new = sklearn_pca.fit_transform(X_std)
    # X_new.shape
    # X_new
