import os
import pickle
import random
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
#import orca
from torch_scatter import scatter_add

from common import utils

AUGMENT_METHOD = "concat"
FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = [], []
#FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = ["identity"], [4]
#FEATURE_AUGMENT = ["motif_counts"]
#FEATURE_AUGMENT_DIMS = [73]
#FEATURE_AUGMENT_DIMS = [15]

def norm(edge_index, num_nodes, edge_weight=None, improved=False,
         dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def compute_identity(edge_index, n, k):
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                             device=edge_index.device)
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, 1, n)
    adj_sparse = torch.sparse.FloatTensor(edge_index, edge_weight,
        torch.Size([n, n]))
    adj = adj_sparse.to_dense()

    deg = torch.diag(torch.sum(adj, -1))
    deg_inv_sqrt = deg.pow(-0.5)
    adj = deg_inv_sqrt @ adj @ deg_inv_sqrt 

    diag_all = [torch.diag(adj)]
    adj_power = adj
    for i in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all

class FeatureAugment(nn.Module):
    def __init__(self):
        super(FeatureAugment, self).__init__()

        def degree_fun(graph, feature_dim):
            graph.node_degree = self._one_hot_tensor(
                [d for _, d in graph.G.degree()],
                one_hot_dim=feature_dim)
            return graph

        def centrality_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            centrality = nx.betweenness_centrality(graph.G)
            graph.betweenness_centrality = torch.tensor(
                [centrality[x] for x in
                nodes]).unsqueeze(1)
            return graph

        def path_len_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            graph.path_len = self._one_hot_tensor(
                [np.mean(list(nx.shortest_path_length(graph.G,
                    source=x).values())) for x in nodes],
                one_hot_dim=feature_dim)
            return graph

        def pagerank_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            pagerank = nx.pagerank(graph.G)
            graph.pagerank = torch.tensor([pagerank[x] for x in
                nodes]).unsqueeze(1)
            return graph

        def identity_fun(graph, feature_dim):
            graph.identity = compute_identity(
                graph.edge_index, graph.num_nodes, feature_dim)
            return graph

        def clustering_coefficient_fun(graph, feature_dim):
            node_cc = list(nx.clustering(graph.G).values())
            if feature_dim == 1:
                graph.node_clustering_coefficient = torch.tensor(
                        node_cc, dtype=torch.float).unsqueeze(1)
            else:
                graph.node_clustering_coefficient = FeatureAugment._bin_features(
                        node_cc, feature_dim=feature_dim)

        def motif_counts_fun(graph, feature_dim):
            assert feature_dim % 73 == 0
            counts = orca.orbit_counts("node", 5, graph.G)
            counts = [[np.log(c) if c > 0 else -1.0 for c in l] for l in counts]
            counts = torch.tensor(counts).type(torch.float)
            #counts = FeatureAugment._wave_features(counts,
            #    feature_dim=feature_dim // 73)
            graph.motif_counts = counts
            return graph

        def node_features_base_fun(graph, feature_dim):
            for v in graph.G.nodes:
                if "node_feature" not in graph.G.nodes[v]:
                    graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
            return graph

        self.node_features_base_fun = node_features_base_fun

        self.node_feature_funs = {"node_degree": degree_fun,
            "betweenness_centrality": centrality_fun,
            "path_len": path_len_fun,
            "pagerank": pagerank_fun,
            'node_clustering_coefficient': clustering_coefficient_fun,
            "motif_counts": motif_counts_fun,
            "identity": identity_fun}

    def register_feature_fun(name, feature_fun):
        self.node_feature_funs[name] = feature_fun

    @staticmethod
    def _wave_features(list_scalars, feature_dim=4, scale=10000):
        pos = np.array(list_scalars)
        if len(pos.shape) == 1:
            pos = pos[:,np.newaxis]
        batch_size, n_feats = pos.shape
        pos = pos.reshape(-1)
        
        rng = np.arange(0, feature_dim // 2).astype(
            np.float) / (feature_dim // 2)
        sins = np.sin(pos[:,np.newaxis] / scale**rng[np.newaxis,:])
        coss = np.cos(pos[:,np.newaxis] / scale**rng[np.newaxis,:])
        m = np.concatenate((coss, sins), axis=-1)
        m = m.reshape(batch_size, -1).astype(np.float)
        m = torch.from_numpy(m).type(torch.float)
        return m

    @staticmethod
    def _bin_features(list_scalars, feature_dim=2):
        arr = np.array(list_scalars)
        min_val, max_val = np.min(arr), np.max(arr)
        bins = np.linspace(min_val, max_val, num=feature_dim)
        feat = np.digitize(arr, bins) - 1
        assert np.min(feat) == 0
        assert np.max(feat) == feature_dim - 1
        return FeatureAugment._one_hot_tensor(feat, one_hot_dim=feature_dim)

    @staticmethod
    def _one_hot_tensor(list_scalars, one_hot_dim=1):
        if not isinstance(list_scalars, list) and not list_scalars.ndim == 1:
            raise ValueError("input to _one_hot_tensor must be 1-D list")
        vals = torch.LongTensor(list_scalars).view(-1,1)
        vals = vals - min(vals)
        vals = torch.min(vals, torch.tensor(one_hot_dim - 1))
        vals = torch.max(vals, torch.tensor(0))
        one_hot = torch.zeros(len(list_scalars), one_hot_dim)
        one_hot.scatter_(1, vals, 1.0)
        return one_hot

    def augment(self, dataset):
        dataset = dataset.apply_transform(self.node_features_base_fun,
            feature_dim=1)
        for key, dim in zip(FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS):
            dataset = dataset.apply_transform(self.node_feature_funs[key], 
                feature_dim=dim)
        return dataset

class Preprocess(nn.Module):
    def __init__(self, dim_in):
        super(Preprocess, self).__init__()
        self.dim_in = dim_in
        if AUGMENT_METHOD == 'add':
            self.module_dict = {
                    key: nn.Linear(aug_dim, dim_in)
                    for key, aug_dim in zip(FEATURE_AUGMENT, 
                                            FEATURE_AUGMENT_DIMS)
                    }

    @property
    def dim_out(self):
        if AUGMENT_METHOD == 'concat':
            return self.dim_in + sum(
                    [aug_dim for aug_dim in FEATURE_AUGMENT_DIMS])
        elif AUGMENT_METHOD == 'add':
            return dim_in
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                    AUGMENT_METHOD))

    def forward(self, batch):
        if AUGMENT_METHOD == 'concat':
            feature_list = [batch.node_feature]
            for key in FEATURE_AUGMENT:
                feature_list.append(batch[key])
            batch.node_feature = torch.cat(feature_list, dim=-1)
        elif AUGMENT_METHOD == 'add':
            for key in FEATURE_AUGMENT:
                batch.node_feature = batch.node_feature + self.module_dict[key](
                        batch[key])
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                    AUGMENT_METHOD))
        return batch
