"""Defines all graph embedding models"""
from functools import reduce
import random

from itertools import permutations, product
from scipy.sparse import csr_matrix
import dgl
import torch.sparse as tsp

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from sklearn.preprocessing import normalize

from common import utils
from common import feature_preprocess

# adapted from https://github.com/leichen2018/GNN-Substructure-Counting/tree/main/synthetic
def Elist(graph):
    E_list = []
    for i in range(graph.number_of_nodes()):
        E_list.append(graph.successors(i).numpy())
    return E_list

def seq_generate_easy(E_list, start_node, length = 4, full_permutation = False):
    if full_permutation:
        all_perm_full = permutations(E_list[start_node])
        all_perm = [[start_node] + list(p[:length]) for p in all_perm_full]
        return all_perm
    else:
        all_perm = permutations(E_list[start_node], min(length, len(E_list[start_node])))
        return [[start_node] + list(p) for p in all_perm]

def seq_to_sp_indx(graph, one_perm, subtensor_length):
    dim_dict = {node:i for i, node in enumerate(one_perm)}

    node_to_length_indx_row = [i + i * subtensor_length for i in range(len(one_perm))]
    node_to_length_indx_col = one_perm

    product_one_perm = list(product(one_perm, one_perm))
    query_edge_id_src, query_edge_id_end = [edge[0] for edge in product_one_perm], [edge[1] for edge in product_one_perm]
    query_edge_result = graph.edge_ids(query_edge_id_src, query_edge_id_end, return_uv = True)

    edge_to_length_indx_row = [int(dim_dict[src.item()] * subtensor_length + dim_dict[end.item()]) for src, end, _ in zip(*query_edge_result)]
    edge_to_length_indx_col = [int(edge_id.item()) for edge_id in query_edge_result[2]]

    return [np.array(node_to_length_indx_row), np.array(node_to_length_indx_col), np.array(edge_to_length_indx_row), np.array(edge_to_length_indx_col)]

def lrp_helper(graph, subtensor_length = 4, full_permutation = False):
    num_of_nodes = graph.number_of_nodes()
    graph_Elist = Elist(graph)

    egonet_seq_graph = []

    for i in range(num_of_nodes):
        # this_node_perms = seq_generate(graph_Elist, i, 1, split_level = False)
        this_node_perms = seq_generate_easy(graph_Elist, start_node = i, length = subtensor_length - 1, full_permutation = full_permutation)
        this_node_egonet_seq = []

        for perm in this_node_perms:
            this_node_egonet_seq.append(seq_to_sp_indx(graph, perm, subtensor_length))
        this_node_egonet_seq = np.array(this_node_egonet_seq)
        egonet_seq_graph.append(this_node_egonet_seq)

    egonet_seq_graph = np.array(egonet_seq_graph)

    return egonet_seq_graph
def np_sparse_to_pt_sparse(matrix):
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def build_perm_pooling_sp_matrix(split_list, pooling = "sum"):
    dim0, dim1 = len(split_list), sum(split_list)
    col = np.arange(dim1)
    row = np.array([i for i, count in enumerate(split_list) for j in range(count)])
    data = np.ones((dim1, ))
    pooling_sp_matrix = csr_matrix((data, (row, col)), shape = (dim0, dim1))

    if pooling == "mean":
        pooling_sp_matrix = normalize(pooling_sp_matrix, norm='l1', axis=1)
    
    return np_sparse_to_pt_sparse(pooling_sp_matrix)

def build_batch_graph_node_to_perm_times_length(graphs, lrp_egonet):
    '''
        graphs: list of DGLGraph
        lrp_egonet: list of egonet of graph, dim: #graphs x #nodes x #perms x
                     (sparse index for length x #nodes or #edges)
    '''
    list_num_nodes_in_graphs = [g.number_of_nodes() for g in graphs]
    sum_num_nodes_before_graphs = [sum(list_num_nodes_in_graphs[:i]) for i in range(len(graphs))]
    list_num_edges_in_graphs = [g.number_of_edges() for g in graphs]
    sum_num_edges_before_graphs = [sum(list_num_edges_in_graphs[:i]) for i in range(len(graphs))]

    node_to_perm_length_indx_row = []
    node_to_perm_length_indx_col = []
    edge_to_perm_length_indx_row = []
    edge_to_perm_length_indx_col = []

    sum_row_number = 0

    for i, g_egonet in enumerate(lrp_egonet):
        for n_egonet in g_egonet:
            for perm in n_egonet:
                node_to_perm_length_indx_col.extend(perm[1] + sum_num_nodes_before_graphs[i])
                node_to_perm_length_indx_row.extend(perm[0] + sum_row_number)

                edge_to_perm_length_indx_col.extend(perm[3] + sum_num_edges_before_graphs[i])
                edge_to_perm_length_indx_row.extend(perm[2] + sum_row_number)

                sum_row_number += 16

    node_to_perm_length_size_row = sum_row_number
    node_to_perm_length_size_col = sum(list_num_nodes_in_graphs)
    edge_to_perm_length_size_row = sum_row_number
    edge_to_perm_length_size_col = sum(list_num_edges_in_graphs)

    # return node_to_perm_length_indx_row, node_to_perm_length_indx_col
    # return edge_to_perm_length_indx_row, edge_to_perm_length_indx_col
    data1 = np.ones((len(node_to_perm_length_indx_col, )))
    node_to_perm_length_sp_matrix = csr_matrix((data1, (node_to_perm_length_indx_row, node_to_perm_length_indx_col)), shape = (node_to_perm_length_size_row, node_to_perm_length_size_col))

    data2 = np.ones((len(edge_to_perm_length_indx_col, )))
    edge_to_perm_length_sp_matrix = csr_matrix((data2, (edge_to_perm_length_indx_row, edge_to_perm_length_indx_col)), shape = (edge_to_perm_length_size_row, edge_to_perm_length_size_col))

    return node_to_perm_length_sp_matrix, edge_to_perm_length_sp_matrix

def collate_lrp_dgl_light(samples):
    graphs, lrp_egonets, labels = map(list, zip(*samples))
    n_to_pl, e_to_pl = build_batch_graph_node_to_perm_times_length(graphs, lrp_egonets)
    batched_graph = dgl.batch(graphs)
    return batched_graph, [len(node) for g in lrp_egonets for node in g], [n_to_pl, e_to_pl], torch.stack(labels)

class LRP_layer(nn.Module):
    def __init__(self,
                 lrp_length = 16,
                 lrp_e_dim = 2,
                 lrp_in_dim = 2,
                 lrp_out_dim = 128):
        super(LRP_layer, self).__init__()

        coeffs_values_3 = lambda i, j, k: torch.randn([i, j, k])
        coeffs_values_4 = lambda i, j, k, l: torch.randn([i, j, k, l])
        self.weights = nn.Parameter(coeffs_values_3(lrp_in_dim, lrp_out_dim, lrp_length))

        self.bias = nn.Parameter(torch.zeros(1, lrp_out_dim))

        self.degnet_0, self.degnet_1 = nn.Linear(1, 2 * lrp_out_dim), nn.Linear(2 * lrp_out_dim, lrp_out_dim)

        self.lrp_length = lrp_length
        self.lrp_e_dim = lrp_e_dim
        self.lrp_in_dim = lrp_in_dim
        self.lrp_out_dim = lrp_out_dim
    
    def forward(self, nfeat, pooling_matrix, degs, n_to_perm_length_sp_matrix):
        #nfeat = graph.ndata['h']

        nfeat = tsp.mm(n_to_perm_length_sp_matrix, nfeat)# + tsp.mm(e_to_perm_length_sp_matrix, efeat)

        nfeat = nfeat.transpose(0, 1).view(self.lrp_in_dim, -1, self.lrp_length).permute(1, 2, 0)

        nfeat = F.relu(torch.einsum('dab,bca->dc', nfeat, self.weights) + self.bias)
        nfeat = tsp.mm(pooling_matrix, nfeat)

        factor_degs = self.degnet_1(F.relu(self.degnet_0(degs.unsqueeze(1))))#.squeeze()

        nfeat = F.relu(torch.einsum('ab,ab->ab', nfeat, factor_degs))

        #graph.ndata['h'] = nfeat
        return nfeat


class LRP(nn.Module):
    def __init__(self, input_dim,
                 output_dim,
                 lrp_length = 16,
                 #lrp_in_dim = 2,
                 hid_dim = 128,
                 num_layers = 1,
                 bn = False,
                 mlp = False
                 ):
        super(LRP, self).__init__()
        lrp_in_dim = input_dim
        
        self.lrp_list = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lrp_list.append(LRP_layer(lrp_length = lrp_length,
                                                lrp_e_dim = lrp_in_dim,
                                                lrp_in_dim = lrp_in_dim,
                                                lrp_out_dim = hid_dim 
                                                ))
            else:
                self.lrp_list.append(LRP_layer(lrp_length = lrp_length,
                                                lrp_e_dim = 2,
                                                lrp_in_dim = hid_dim,
                                                lrp_out_dim = hid_dim 
                                                ))

        self.final_predict = nn.Linear(hid_dim, output_dim)

        self.bn = bn
        self.mlp = mlp

        if bn:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hid_dim) for i in range(num_layers)])

        if mlp:
            self.mlp_layers = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for i in range(num_layers)])

        if len(feature_preprocess.FEATURE_AUGMENT) > 0:
            self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

    def forward_asdf(self, graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix):
        graph.ndata['h'] = graph.ndata['feat']
        for lrp_layer, mlp_layer, bn in zip(self.lrp_list, self.mlp_layers, self.bn_layers):
            graph = lrp_layer(graph, pooling_matrix, degs, n_to_perm_length_sp_matrix, e_to_perm_length_sp_matrix)
            graph.ndata['h'] = bn(graph.ndata['h'])
            graph.ndata['h'] = F.relu(mlp_layer(graph.ndata['h']))
        graph.ndata['h'] = self.final_predict(graph.ndata['h'])
        output = dgl.sum_nodes(graph, 'h')
        return output

    def forward(self, data):
        #if data.x is None:
        #    data.x = torch.ones((data.num_nodes, 1), device=utils.get_device())

        #x = self.pre_mp(x)
        if self.feat_preprocess is not None:
            if not hasattr(data, "preprocessed"):
                data = self.feat_preprocess(data)
                data.preprocessed = True
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        def nx_to_graph_entry(g):
            dgl_g = dgl.DGLGraph()
            dgl_g.from_networkx(g, node_attrs=["node_feature"])
            return dgl_g, lrp_helper(dgl_g), torch.tensor(0)
        loader = DataLoader([nx_to_graph_entry(g) for g in data.G],
            batch_size=len(data.G),
            shuffle=False, collate_fn=collate_lrp_dgl_light)
        batch, split_list, sp_matrices, _ = next(iter(loader))

        device = torch.device("cuda")
        mean_pooling_matrix = build_perm_pooling_sp_matrix(
            split_list, "mean").to(device)

        n_to_perm_length_sp_matrix = np_sparse_to_pt_sparse(
            sp_matrices[0]).to(device)
        degs = batch.in_degrees(list(range(batch.number_of_nodes()))).type(torch.FloatTensor).to(device)

        h = batch.ndata["node_feature"].to(device)
        for lrp_layer, mlp_layer, bn in zip(self.lrp_list, self.mlp_layers, self.bn_layers):
            h = lrp_layer(h, mean_pooling_matrix, degs,
                n_to_perm_length_sp_matrix)
            h = bn(h)
            #print(h.shape)
            h = F.relu(mlp_layer(h))
        #graph.ndata['h'] = self.final_predict(graph.ndata['h'])
        h = self.final_predict(h)
        batch.ndata['h'] = h
        output = dgl.sum_nodes(batch, 'h')
        #output = pyg_nn.global_add_pool(h, batch)
        return output


# LRP -> concat -> MLP graph classification baseline
class BaselineLRP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineLRP, self).__init__()
        self.emb_model = LRP(input_dim, hidden_dim, bn=True, mlp=True)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

# GNN -> concat -> MLP graph classification baseline
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

# neural tensor network
class BaselineNTN(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineNTN, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, emb_motif, emb_motif_mod):
        bilin_out = self.bilinear(emb_motif, emb_motif_mod)
        lin_out = self.linear(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = self.linear2(F.relu(bilin_out + lin_out))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred#.argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)

class BoxEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(BoxEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False
        self.regularization = 0
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs
        #return torch.exp(emb_as), torch.exp(emb_bs)

    def get_lr(self, embs):
        half = embs.shape[-1] // 2
        embs_l, embs_d = embs[:,:half], embs[:,half:]
        embs_r = embs_l + torch.abs(embs_d)
        return -embs_l, embs_r

    def predict(self, pred):
        """Predict if a is a subgraph of b (batched)

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred
        #half = emb_as.shape[-1] // 2
        #emb_as_l, emb_as_r = -emb_as[:,:half], emb_as[:,half:]
        #emb_bs_l, emb_bs_r = -emb_bs[:,:half], emb_bs[:,half:]
        emb_as_l, emb_as_r = self.get_lr(emb_as)
        emb_bs_l, emb_bs_r = self.get_lr(emb_bs)

        #return (torch.sum(emb_as + 1e-2 - emb_bs >= 0, dim=1) ==
        #    emb_as.shape[0]).type(torch.long)
        #print(emb_as - emb_bs)
        return ((torch.sum(emb_as_l - emb_bs_l, dim=1) 
             + (torch.sum(emb_bs_r - emb_as_r , dim=1))))

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for box emb.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        #half = emb_as.shape[-1] // 2
        #emb_as_l, emb_as_r = -emb_as[:,:half], emb_as[:,half:]
        #emb_bs_l, emb_bs_r = -emb_bs[:,:half], emb_bs[:,half:]
        emb_as_l, emb_as_r = self.get_lr(emb_as)
        emb_bs_l, emb_bs_r = self.get_lr(emb_bs)
        e = torch.sum(torch.max(torch.zeros_like(emb_as_l,
            device=utils.get_device()), emb_bs_l - emb_as_l)**2, dim=1)
        e[labels == 0] = torch.max(torch.tensor(0.0,
            device=utils.get_device()), self.margin - e)[labels == 0]
        e2 = torch.sum(torch.max(torch.zeros_like(emb_as_r,
            device=utils.get_device()), emb_bs_r - emb_as_r)**2, dim=1)
        e2[labels == 0] = torch.max(torch.tensor(0.0,
            device=utils.get_device()), self.margin - e2)[labels == 0]

        reg = torch.sum(emb_as**2) + torch.sum(emb_bs**2)   # TODO: exact reg
        relation_loss = torch.sum(e) + torch.sum(e2) + self.regularization*reg
        return relation_loss

# Order embedder model -- contains a graph embedding model `emb_model`
class OrderEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict if a is a subgraph of b (batched)

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for order emb.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: subgraph labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as,
            device=utils.get_device()), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0,
            device=utils.get_device()), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss

class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        if len(feature_preprocess.FEATURE_AUGMENT) > 0:
            self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
            input_dim = self.feat_preprocess.dim_out
        else:
            self.feat_preprocess = None

        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
            args.conv_type == "PNA" else hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3*hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        if args.conv_type == "PNA":
            post_input_dim *= 3
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            #return lambda i, h: pyg_nn.GINConv(nn.Sequential(
            #    nn.Linear(i, h), nn.ReLU()))
            return lambda i, h: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)
                ))
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return SAGEConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        #if data.x is None:
        #    data.x = torch.ones((data.num_nodes, 1), device=utils.get_device())

        #x = self.pre_mp(x)
        if self.feat_preprocess is not None:
            if not hasattr(data, "preprocessed"):
                data = self.feat_preprocess(data)
                data.preprocessed = True
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs_sum) if self.conv_type=="PNA" else
            len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                        self.convs_mean[i](curr_emb, edge_index),
                        self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                        self.convs_mean[i](emb, edge_index),
                        self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)
        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        #emb = self.batch_norm(emb)   # TODO: test
        #out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
            out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        #edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        #return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)
        #aggr_out = torch.matmul(aggr_out, self.weight)

        #if self.bias is not None:
        #    aggr_out = aggr_out + self.bias

        #if self.normalize:
        #    aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# pytorch geom GINConv + weighted edges
class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
            edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
            edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

