import re
import torch
import os.path
from args import *
import numpy as np
import pandas as pd
import networkx as nx
import torch.utils.data
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit


def get_omics_dataset():
    # Extract the MultiOmics Dataset Dir
    args = make_args()
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        raise ValueError('No Dataset Directory Specified!')

    omics_dataset_name = ''
    phenotype_file_name = ''
    for _, _, files in os.walk(dataset_dir):
        for file in files:
            if re.compile('.*omics_data\.csv').match(file):
                omics_dataset_name = file
            elif re.compile('.*gold\.csv').match(file):
                phenotype_file_name = file
    if omics_dataset_name == '' or phenotype_file_name == '':
        raise FileNotFoundError

    omics_dataset = pd.read_csv(os.path.join(dataset_dir, omics_dataset_name))
    phenotype = pd.read_csv(os.path.join(dataset_dir,
                                         phenotype_file_name))  # TODO: Consider Consolidating Classes 0, 1 & 2 -> 1, 3 & 4 -> 2
    omics_dataset_complete = omics_dataset.merge(phenotype, how='left', left_index=True, right_index=True)
    # Replacing PRISM (-1) with 5 for Prediction to Work!
    omics_dataset_complete['finalgold_visit'] = np.where(omics_dataset_complete['finalgold_visit'] == -1, 5, omics_dataset_complete['finalgold_visit'])
    return omics_dataset_complete


def get_omics_datasets(omics_dataset):
    args = make_args()
    dataset_dir = args.dataset_dir
    omics_networks_l = []
    omics_datasets = []
    for dir_name, _, files in os.walk(dataset_dir):
        for file in files:
            if re.compile('.*Network\d+-\d+\.csv').match(file):
                omics_networks_l.append(os.path.join(dir_name, file))

    if not omics_networks_l:
        raise FileNotFoundError

    for omics_network_f in omics_networks_l:
        omics_network_adj = pd.read_csv(omics_network_f, index_col=0)
        omics_network_nodes_names = omics_network_adj.index.tolist()
        omics_datasets.append(omics_dataset[omics_network_nodes_names + ['finalgold_visit']])

    return omics_datasets


def get_omics_networks_tg():
    args = make_args()
    dataset_dir = args.dataset_dir
    omics_networks_l = []
    clinical_dataset_f = ''
    for dir_name, _, files in os.walk(dataset_dir):
        for file in files:
            if re.compile('.*Network\d+-\d+\.csv').match(file):
                omics_networks_l.append(os.path.join(dir_name, file))
            elif re.compile('.*clinical_data\.csv').match(file):
                clinical_dataset_f = file
    if clinical_dataset_f == '' or omics_networks_l == []:
        raise FileNotFoundError

    # Filtering the Clinical Dataset to Relevant Columns [List Provided by Katherine]
    clinical_dataset = pd.read_csv(os.path.join(dataset_dir, clinical_dataset_f), low_memory=False)
    clinical_dataset_comorbidities = ['Angina', 'CongestHeartFail', 'CoronaryArtery', 'HeartAttack', 'PeriphVascular',
                                      'Stroke', 'TIA', 'Diabetes', 'Osteoporosis', 'HighBloodPres', 'HighCholest',
                                      'CognitiveDisorder', 'MacularDegen', 'KidneyDisease', 'LiverDisease']
    clinical_dataset = clinical_dataset.assign(
        comorbidities=clinical_dataset[clinical_dataset_comorbidities].sum(axis=1))
    clinical_dataset_cols = ['gender', 'age_visit', 'Chronic_Bronchitis', 'PRM_pct_emphysema_Thirona',
                             'PRM_pct_normal_Thirona', 'Pi10_Thirona', 'comorbidities']

    clinical_dataset = clinical_dataset[clinical_dataset_cols]
    omics_dataset = get_omics_dataset()

    omics_networks_tg = []
    for omics_network_f in omics_networks_l:
        omics_network_adj = pd.read_csv(omics_network_f, index_col=0)
        omics_network = nx.from_numpy_array(omics_network_adj.to_numpy())
        omics_network_nodes_names = omics_network_adj.index.tolist()

        omics_network_nodes_features = []
        for omics_network_node_name in omics_network_nodes_names:
            omics_network_node_features = []
            for clinical_variable in clinical_dataset_cols:
                omics_network_node_features.append(
                    abs(omics_dataset[omics_network_node_name].corr(
                        clinical_dataset[clinical_variable].astype('float64'))))
            omics_network_nodes_features.append(omics_network_node_features)

        omics_network_nodes_features = np.array(omics_network_nodes_features)
        omics_network.remove_edges_from(nx.selfloop_edges(omics_network))  # Removing Selfloop Edges

        x = np.zeros(omics_network_nodes_features.shape)
        graph_nodes = list(omics_network.nodes)
        for m in range(omics_network_nodes_features.shape[0]):
            x[graph_nodes[m]] = omics_network_nodes_features[m]
        x = torch.from_numpy(x).float()

        # Edges Indexes
        edge_index = np.array(list(omics_network.edges))
        edge_index = np.concatenate((edge_index, edge_index[:, ::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)

        # Edges Weights
        edge_weight = np.array(list(nx.get_edge_attributes(omics_network, 'weight').values()))
        edge_weight = np.concatenate((edge_weight, edge_weight), axis=0)
        edge_weight = torch.from_numpy(edge_weight).float()

        # In an End-to-End Pipeline Setup; the Weights for the GNN Params are Updated using the Prediction Loss Function.
        # The y Value is set to 0s and is not used as Part of the Training
        omics_network_tg = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                                y=torch.zeros(len(omics_network_nodes_names)))

        num_val_nodes = 10
        num_test_nodes = 12
        # For Small Networks
        if len(omics_network.nodes()) < 30:
            num_val_nodes = 5
            num_test_nodes = 5

        transform = RandomNodeSplit(num_val=num_val_nodes, num_test=num_test_nodes)
        omics_networks_tg.append(transform(omics_network_tg))
    return omics_networks_tg


def get_dataset():
    # Subjects Multiomics Dataset
    omics_dataset = get_omics_dataset()
    # Different Slices of the Multiomics Dataset based on Different Networks
    omics_datasets = get_omics_datasets(omics_dataset)
    # Multiomics Networks
    omics_networks_tg = get_omics_networks_tg()
    return omics_datasets, omics_networks_tg
