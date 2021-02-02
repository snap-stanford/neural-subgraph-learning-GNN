from itertools import combinations
import networkx as nx
from copy import deepcopy
from random import choice
import numpy as np
import torch
from torch.utils.data import Dataset

def findSubgraphGT(graph, subgraph):
    #print(graph.edges, graph.nodes, subgraph.edges, subgraph.nodes)
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, subgraph,
        node_match=lambda a, b: a["label"] == b["label"])
    return matcher.subgraph_isomorphisms_iter()

def toGT(adj, feat):
    G = nx.Graph()
    
    G.add_edges_from(adj.nonzero().detach().cpu().numpy())
    for v in G.nodes:
        G.nodes[v]["label"] = tuple(feat[v,:].tolist())
    return G

#def findSubgraphGT(graph, subgraph):
#    vm = gtt.subgraph_isomorphism(subgraph, graph, vertex_label=(
#	subgraph.vp['label'], graph.vp['label']), max_n=0)
#    return vm
#
#def toGT(adj, feat):
#    G = Graph()
#    G.add_edge_list(np.transpose(adj.nonzero()))
#    labels = G.new_vertex_property('vector<int>')
#    for idx in G.get_vertices():
#        labels[idx] = feat[idx,:].tolist()
#    G.vertex_properties['label'] = labels
#    return G

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def regen(self):
        raise NotImplementedError

    def set_curriculum_size(self, size):
        raise NotImplementedError


NODE_PROB_LARGE = 0.1
MAX_SIZE_LARGE = 4

NODE_PROB_SMALL = 0.4
MAX_SIZE_SMALL = 100

node_add_prob = NODE_PROB_LARGE
max_size = MAX_SIZE_LARGE

class RandomBasisDataset(BaseDataset):
    def __init__(self, basis_graphs, num_queries, gpu=False, induced=False, init_size=None, query_size=4, phase='all',
                 sample_neighborhoods=False):
        super().__init__()

        self.bases = basis_graphs
        self.num_queries = num_queries
        self.basis_mapping = {}
        self.gpu = gpu
        self.size = init_size if init_size is not None else min(len(self.bases), num_queries)
        self.query_hops = query_size
        self.induced = induced
        self.sample_neighborhoods = sample_neighborhoods
        self.hard_negative_ratio = 0.
        self.phase = phase
        self.regen()

    def set_phase(self, phase):
        self.phase = phase

    def regen(self):
        self.queries = []
        self.neighborhoods = []
        self.centers = []
        self.neighborhood_centers = []
        self.basis_idxs = []
        self.query_sizes = []
        self.neighborhood_sizes = []
        self.max_query_size = 0
        self.max_neighborhood_size = 0

        i = 0
        while i < self.num_queries:
            graph_idx = np.random.choice(self.size, 1)[0]
            self.basis_mapping[i] = graph_idx

            self.basis_idxs.append(graph_idx)
            n_hops = self.query_hops
            graph = self.bases[graph_idx]
            start = np.random.choice(len(graph.nodes))
            if self.sample_neighborhoods:
                neighborhood = self.get_neighborhood(start, graph, n_hops=4,
                        node_add_prob=node_add_prob,
                        max_size=max_size)
                query_start = nx.center(neighborhood)[0]
                query = self.get_neighborhood(query_start, neighborhood, n_hops=n_hops)
                self.centers.append(np.argwhere(np.array(list(query)) == query_start).squeeze().item())
            else:
                query = self.get_neighborhood(start, graph, n_hops=n_hops)
                if self.induced:
                    query = self.get_neighborhood(start, graph, n_hops=n_hops, edge_add_prob=1.0)

                neighborhood = self.get_neighborhood(start, graph, n_hops=4, node_add_prob=1.0, edge_add_prob=1.0)

                self.centers.append(np.argwhere(np.array(list(query)) == start).squeeze().item())
            # remove neighborhoods and queries that are trivial
            if query.number_of_edges() <= 2 or neighborhood.number_of_edges() <= 3:
                continue

            self.queries.append(query)
            self.max_query_size = max(self.max_query_size, query.number_of_nodes())
            self.query_sizes.append(query.number_of_nodes())

            self.neighborhood_centers.append(np.argwhere(np.array(list(neighborhood)) == start).squeeze().item())
            self.neighborhoods.append(neighborhood)
            self.max_neighborhood_size = max(self.max_neighborhood_size, neighborhood.number_of_nodes())
            self.neighborhood_sizes.append(neighborhood.number_of_nodes())

            i += 1

        for idx, query in enumerate(self.queries):
            query_adj, query_feat = self.extract_representation(query, query=True)
            self.queries[idx] = (torch.clamp(query_adj, 0, 1), query_feat)

            neighborhood = self.neighborhoods[idx]
            neighborhood_adj, neighborhood_feat = self.extract_representation(neighborhood)
            self.neighborhoods[idx] = (torch.clamp(neighborhood_adj, 0, 1), neighborhood_feat)

    def to_numpy_matrix(self, G, edge_type=False):
        adj = nx.to_numpy_matrix(G).astype(int)
        if edge_type:
            n_vals = adj.max() + 1
            # this creates the 3D adj: n x n x (edge_types+1). The +1 is due to entry 0 (no edge)
            # The edges types are from 1, 2, ...
            adj_categorical = np.eye(n_vals)[adj]
            # remove the dim corresponding to edge type 0 ( no edge )
            adj_categorical = adj_categorical[:, :, 1:]
            # move the edge type dimension to the first dim
            return adj_categorical.transpose(2, 0, 1)
        return adj

    def extract_representation(self, G, query=False):
        adj = self.to_numpy_matrix(G, edge_type=False).astype(np.float32)
        feat = np.array([G.nodes[u]['feat'] for u in G.nodes()]).astype(np.float32)

        pad_size = self.max_query_size if query else self.max_neighborhood_size
        adj_padded = torch.from_numpy(np.pad(adj, ((0, pad_size - adj.shape[0]),), mode='constant')[np.newaxis, :, :])
        feat_padded = torch.from_numpy(
                np.pad(feat, ((0, pad_size - feat.shape[0]), (0, 0)), mode='constant'))

        if self.gpu:
            adj_padded = adj_padded.cuda()
            feat_padded = feat_padded.cuda()

        return adj_padded, feat_padded

    def set_curriculum_size(self, size):
        self.size = min(min(size, len(self.bases)), self.num_queries)

    def set_query_hops(self, hops):
        self.query_hops = hops

    def __len__(self):
        return self.size

    @staticmethod
    def add_edge_GT(G, nodes):
        first_node = G.vertex(choice(nodes))
        possible_nodes = set(nodes)
        neighbours = list(first_node.out_neighbors()) + [first_node]
        possible_nodes.difference_update(neighbours)
        possible_nodes = list(possible_nodes)
        if len(possible_nodes) > 0:
            second_node = G.vertex(choice(list(possible_nodes)))
            G.add_edge(first_node, second_node)
        return G

    def __getitem__(self, idx):
        query_adj, query_feat = self.queries[idx]
        query_adj = deepcopy(query_adj)
        center = self.centers[idx]
        q_size = self.query_sizes[idx]

        label = torch.tensor([np.random.rand() < 0.25]).long()
        if label.item() == 1:
            neighborhood_adj, neighborhood_feat = self.neighborhoods[idx]
            if np.random.rand() < 0.5 or self.phase == 'center':
                neighborhood_center = self.neighborhood_centers[idx]
            else:
                n_size = self.neighborhood_sizes[idx]
                q = toGT(query_adj[0, :q_size, :q_size].cpu(), query_feat[:q_size, :].cpu())
                n = toGT(neighborhood_adj[0, :n_size, :n_size].cpu(), neighborhood_feat[:n_size, :].cpu())
                #print("Q", q.edges)
                #print("T", n.edges)

                mapping = findSubgraphGT(n, q)
                #print(mapping)
                mapping = next(iter(mapping))

                #print(mapping, self.query_sizes[idx])
                neighborhood_center, center = choice(
                    list(mapping.items()))
                #print(neighborhood_center, center)
                #input()
                #center = np.random.choice(self.query_sizes[idx])
                #neighborhood_center = mapping[center]

        else:
            if np.random.rand() < self.hard_negative_ratio:
                # Hard negatives
                neighborhood_adj, neighborhood_feat = self.neighborhoods[idx]
                n_size = self.neighborhood_sizes[idx]
                neighborhood_center = self.neighborhood_centers[idx]

                if self.induced:
                    edges = query_adj[0, :q_size, :q_size].nonzero()
                    edges = [edge for edge in edges if edge[0] > edge[1]]
                    remove_num = int(np.random.rand() * 0.2 * len(edges))
                    remove_edges = np.random.choice(edges, remove_num, replace=False)
                    for edge in remove_edges:
                        query_adj[edge[0], edge[1]] = 0
                        query_adj[edge[1], edge[0]] = 1
                else:
                    q = toGT(query_adj[0, :q_size, :q_size].cpu(), query_feat[:q_size, :].cpu())
                    n = toGT(neighborhood_adj[0, :n_size, :n_size].cpu(), neighborhood_feat[:n_size, :].cpu())

                    nodes = q.get_vertices()
                    found = False
                    for _ in range(5):  # 5 tries
                        add_num = np.random.randint(1, 10)
                        for p in range(add_num):
                            q_test = self.add_edge_GT(q, nodes)
                        if len(findSubgraphGT(n, q_test)) == 0:
                            found = True
                            adj = adjacency(q_test).toarray()
                            query_adj = torch.zeros_like(query_adj)
                            query_adj[0, :adj.shape[0], :adj.shape[1]] = torch.from_numpy(adj)
                            break

                    if not found:
                        label = torch.tensor([1])
            else:
                q = toGT(query_adj[0, :q_size, :q_size].cpu(), query_feat[:q_size, :].cpu())
                if (self.size > 1 and np.random.rand() < 0.4) or self.phase == 'center':
                    # Use another neighborhood node as negative
                    negatives = [j for j in range(self.num_queries) if j != idx]
                    found = False
                    for tries in range(5):
                        neg_idx = np.random.choice(negatives)

                        neighborhood_adj, neighborhood_feat = self.neighborhoods[neg_idx]
                        #neighborhood_center = self.neighborhood_centers[neg_idx]
                        neighborhood_center = np.random.choice(self.neighborhood_sizes[neg_idx])
                        n_size = self.neighborhood_sizes[neg_idx]

                        n = toGT(neighborhood_adj[0, :n_size, :n_size].cpu(), neighborhood_feat[:n_size, :].cpu())

                        success = True
                        for match in findSubgraphGT(n, q):
                            if neighborhood_center in match and \
                                match[neighborhood_center] == center:
                                success = False
                                break
                        if success:
                            found = True
                            break

                else:
                    # Use noncenter node from the true neighborhood as negative
                    negative_centers = [j for j in range(self.neighborhood_sizes[idx]) if j != self.neighborhood_centers[idx]]
                    neighborhood_center = np.random.choice(negative_centers)
                    neighborhood_adj, neighborhood_feat = self.neighborhoods[idx]
                    n_size = self.neighborhood_sizes[idx]

                    n = toGT(neighborhood_adj[0, :n_size, :n_size].cpu(), neighborhood_feat[:n_size, :].cpu())

                    found = True
                    for _ in range(5):
                        i = 0
                        for match in findSubgraphGT(n, q):
                            i += 1
                            #print(neighborhood_center, match.keys())
                            if neighborhood_center in match.keys():
                                found = False
                                break
                        if found:
                            break
                        neighborhood_center = np.random.choice(negative_centers)
                if not found:
                    label = torch.tensor([1])

        idx = torch.tensor(idx)
        if self.gpu:
            label = label.cuda()
            idx = idx.cuda()
        return query_adj, query_feat, center, neighborhood_adj, neighborhood_feat, \
               neighborhood_center, label, idx

    def get_neighborhood(self, start, graph, n_hops=4, node_add_prob=0.6, edge_add_prob=0.8, max_size=None):
        G = nx.Graph()
        G.add_node(start)
        queue = [(start, 0)]
        while queue:
            node, layer = queue.pop(0)
            G.nodes[node]['feat'] = graph.nodes[node]['feat']
            if max_size is not None and nx.number_of_nodes(G) > max_size:
                continue
            if layer >= n_hops:
                continue
            for neighbor in graph.neighbors(node):
                if G.number_of_nodes() > 2 and np.random.rand() > node_add_prob:
                    continue
                G.add_node(neighbor)
                G.add_edge(node, neighbor)
                queue.append((neighbor, layer + 1))

        for node1, node2 in combinations(G.nodes, 2):
            if graph.has_edge(node1, node2) and not G.has_edge(node1, node2):
                if np.random.rand() <= edge_add_prob:
                    G.add_edge(node1, node2)

        if self.induced:
            G = graph.subgraph(G.nodes)
        return G

    def visualize(self, writer, prefix='', savepath=None, num_plots=np.inf):
        for i, query_tuple in enumerate(self.queries):
            if i >= num_plots:
                return
            query = nx.from_numpy_matrix(query_tuple[0].cpu().numpy().squeeze(0))
            if len(query.nodes) > 1:
                query.remove_nodes_from(list(nx.isolates(query)))
            neighborhood = nx.from_numpy_matrix(self.neighborhoods[i][0].cpu().numpy().squeeze(0))
            if len(neighborhood.nodes) > 1:
                neighborhood.remove_nodes_from(list(nx.isolates(neighborhood)))
            self.plot_graph(query, prefix + 'query_' + str(i), writer, savepath)
            self.plot_graph(neighborhood, prefix + 'neighborhood_' + str(i), writer, savepath)

