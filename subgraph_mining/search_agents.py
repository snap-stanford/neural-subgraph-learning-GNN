import argparse
import csv
from itertools import combinations
import time
import os

from deepsnap.batch import Batch
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import TUDataset, PPI
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from decoder.config import parse_decoder
from encoder.config import parse_encoder

import matplotlib.pyplot as plt

import random
from scipy.io import mmread
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import defaultdict
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA

class SearchAgent:
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        neighs, embs, method="greedy", node_anchored=False,
        analyze=False, rank_method="counts", model_type="order",
        out_batch_size=20):
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.neighs = neighs
        self.embs = embs
        self.method = method
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.rank_method = rank_method
        self.model_type = model_type
        self.out_batch_size = out_batch_size
        print("RANK METHOD:", rank_method)

    def run_search(self, n_trials=1000):
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))

        search_ctx = self.init_search(n_trials)
        while not self.is_search_done(search_ctx):
            search_ctx = self.step(search_ctx, n_trials)
        return self.finish_search(search_ctx)

class MCTSSearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        neighs, embs, method="greedy", node_anchored=False,
        analyze=False, rank_method="counts", model_type="order",
        out_batch_size=20, c_uct=0.7):
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            neighs, embs, method=method, node_anchored=False,
            analyze=False, rank_method=rank_method, model_type=model_type,
            out_batch_size=out_batch_size)
        self.c_uct = c_uct
        self.max_size = self.min_pattern_size

    def init_search(self, n_trials):
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()

    def is_search_done(self, ctx):
        return self.max_size == self.max_pattern_size + 1

    # returns whether there are at least n nodes reachable from start_node in graph
    def has_min_reachable_nodes(self, graph, start_node, n):
        for depth_limit in range(n+1):
            edges = nx.bfs_edges(graph, start_node, depth_limit=depth_limit)
            nodes = set([v for u, v in edges])
            if len(nodes) + 1 >= n:
                return True
        return False

    def step(self, ctx, n_trials):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print(self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        for simulation_n in tqdm(range(n_trials //
            (self.max_pattern_size+1-self.min_pattern_size))):
            # pick seed node
            best_graph_idx, best_start_node, best_score = None, None, -float("inf")
            for cand_graph_idx, cand_start_node in self.visited_seed_nodes:
                state = cand_graph_idx, cand_start_node
                my_visit_counts = sum(self.visit_counts[state].values())
                q_score = (sum(self.cum_action_values[state].values()) /
                    (my_visit_counts or 1))
                uct_score = self.c_uct * np.sqrt(np.log(simulation_n or 1) /
                    (my_visit_counts or 1))
                node_score = q_score + uct_score
                #print(cand_graph_idx, cand_start_node, q_score, uct_score)
                if node_score > best_score:
                    best_score = node_score
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node
            # if existing seed beats choosing a new seed
            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                assert best_start_node in self.dataset[graph_idx].nodes
                graph = self.dataset[graph_idx]
                #print("old seed")
            else:
                #print(simulation_n, "new seed")
                found = False
                while not found:
                    graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
                    graph = self.dataset[graph_idx]
                    start_node = random.choice(list(graph.nodes))
                    # don't pick isolated nodes or small islands
                    if self.has_min_reachable_nodes(graph, start_node,
                        self.min_pattern_size):
                        found = True
                self.visited_seed_nodes.add((graph_idx, start_node))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            neigh_g = nx.Graph()
            neigh_g.add_node(start_node, anchor=1)
            cur_state = graph_idx, start_node
            state_list = [cur_state]
            #print(len(frontier), len(neigh))
            while frontier and len(neigh) < self.max_size:
                #print(len(neigh))
                cand_neighs, anchors = [], []
                for cand_node in frontier:
                    cand_neigh = graph.subgraph(neigh + [cand_node])
                    cand_neighs.append(cand_neigh)
                    if self.node_anchored:
                        anchors.append(neigh[0])
                cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                    cand_neighs, anchors=anchors if self.node_anchored else None))
                best_v_score, best_node_score, best_node = 0, -float("inf"), None
                for cand_node, cand_emb in zip(frontier, cand_embs):
                    score, n_embs = 0, 0
                    for emb_batch in self.embs:
                        score += torch.sum(self.model.predict((
                            emb_batch.to(utils.get_device()), cand_emb))).item()
                        n_embs += len(emb_batch)
                    v_score = -np.log(score/n_embs + 1) + 1
                    # get wl hash of next state
                    neigh_g = graph.subgraph(neigh + [cand_node]).copy()
                    neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                    for v in neigh_g.nodes:
                        neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                    next_state = utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)
                    # compute node score
                    parent_visit_counts = sum(self.visit_counts[cur_state].values())
                    my_visit_counts = sum(self.visit_counts[next_state].values())
                    q_score = (sum(self.cum_action_values[next_state].values()) /
                        (my_visit_counts or 1))
                    uct_score = self.c_uct * np.sqrt(np.log(parent_visit_counts or
                        1) / (my_visit_counts or 1))
                    node_score = q_score + uct_score
                    #print(q_score, uct_score, len(neigh))
                    if node_score > best_node_score:
                        best_node_score = node_score
                        best_v_score = v_score
                        best_node = cand_node
                frontier = list(((set(frontier) |
                    set(graph.neighbors(best_node))) - visited) -
                    set([best_node]))
                visited.add(best_node)
                neigh.append(best_node)

                # update visit counts, wl cache
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                prev_state = cur_state
                cur_state = utils.wl_hash(neigh_g, node_anchored=self.node_anchored)
                state_list.append(cur_state)
                self.wl_hash_to_graphs[cur_state].append(neigh_g)

            # backprop value
            for i in range(0, len(state_list) - 1):
                #print(best_v_score)
                self.cum_action_values[state_list[i]][
                    state_list[i+1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i+1]] += 1
        self.max_size += 1

    def finish_search(self, ctx):
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count
        
        print(list(counts.keys()))
        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            for wl_hash, count in sorted(counts[pattern_size].items(), key=lambda
                x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash]))
                print(pattern_size, count)
        return cand_patterns_uniq

class GreedySearchAgent(SearchAgent):
    def init_search(self, n_trials):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        for trial in range(n_trials):
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            #graph_idx = random.randint(0, len(dataset)-1)
            graph = self.dataset[graph_idx]
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            beams.append((0, neigh, frontier, visited, graph_idx))
        return beams

    def is_search_done(self, beams):
        return len(beams) == 0

    def step(self, beams, n_trials):
        #beam_size = int(method.split("-")[-1]) if "-" in method else 1
        #while len(neigh) < max_pattern_size and frontier:
        new_beams = []
        print(len(set(b[-1] for b in beams)))
        for _, neigh, frontier, visited, graph_idx in tqdm(beams):
            graph = self.dataset[graph_idx]
            if len(neigh) >= self.max_pattern_size or not frontier: continue
            #for cand_node in random.sample(frontier, len(frontier)):
            cand_neighs, anchors = [], []
            for cand_node in frontier:
                cand_neigh = graph.subgraph(neigh + [cand_node])
                cand_neighs.append(cand_neigh)
                if self.node_anchored:
                    anchors.append(neigh[0])
            cand_embs = self.model.emb_model(utils.batch_nx_graphs(
                cand_neighs, anchors=anchors if self.node_anchored else None))
            best_score, best_node = float("inf"), None
            for cand_node, cand_emb in zip(frontier, cand_embs):
                score, n_embs = 0, 0
                for emb_batch in self.embs:
                    n_embs += len(emb_batch)
                    if self.model_type == "order":
                        score -= torch.sum(torch.argmax(
                            self.model.clf_model(self.model.predict((
                            emb_batch.to(utils.get_device()),
                            cand_emb)).unsqueeze(1)), axis=1)).item()
                    elif self.model_type == "mlp":
                        score += torch.sum(self.model(
                            emb_batch.to(utils.get_device()),
                            cand_emb.unsqueeze(0).expand(len(emb_batch), -1)
                            )[:,0]).item()
                    else:
                        print("unrecognized model type")
                #score = np.log(score/n_embs + 1)
                if score < best_score:
                    best_score = score
                    best_node = cand_node
            new_frontier = list(((set(frontier) |
                set(graph.neighbors(best_node))) - visited) - set([best_node]))
            new_beams.append((
            #new_beams[graph_idx, tuple(sorted(neigh+[cand_node]))] = (
                best_score, neigh + [best_node],
                new_frontier, visited | set([best_node]), graph_idx))
        #new_beams = list(new_beams.values())
        new_beams = list(sorted(new_beams, key=lambda x: x[0]))[:n_trials]
        for score, neigh, frontier, visited, graph_idx in new_beams:#[:1]:
            graph = self.dataset[graph_idx]
            #print(frontier, len(frontier))
            # add to record
            neigh_g = graph.subgraph(neigh).copy()
            neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
            for v in neigh_g.nodes:
                neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
            self.cand_patterns[len(neigh_g)].append((score, neigh_g))
            if self.rank_method in ["counts", "hybrid"]:
                self.counts[len(neigh_g)][utils.wl_hash(neigh_g,
                    node_anchored=self.node_anchored)].append(neigh_g)
        beams = new_beams
        return beams

    def finish_search(self, ctx):
        if self.analyze:
            with open("results/analyze.p", "wb") as f:
                pickle.dump((cand_patterns, neighs), f)

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            if self.rank_method == "hybrid":
                cur_rank_method = "margin" if len(max(
                    self.counts[pattern_size].values(), key=len)) < 3 else "counts"
            else:
                cur_rank_method = self.rank_method
            print(pattern_size, cur_rank_method)

            if cur_rank_method == "margin":
                wl_hashes = set()
                cands = cand_patterns[pattern_size]
                cand_patterns_uniq_size = []
                for pattern in sorted(cands, key=lambda x: x[0]):
                    wl_hash = utils.wl_hash(pattern[1], node_anchored=node_anchored)
                    if wl_hash not in wl_hashes:
                        wl_hashes.add(wl_hash)
                        cand_patterns_uniq_size.append(pattern[1])
                        if len(cand_patterns_uniq_size) >= out_batch_size:
                            cand_patterns_uniq += cand_patterns_uniq_size
                            break
            elif cur_rank_method == "counts":
                for _, neighs in list(sorted(self.counts[pattern_size].items(),
                    key=lambda x: len(x[1]), reverse=True))[:self.out_batch_size]:
                    print(pattern_size, len(neighs))
                    #print(len(neighs))
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("Unrecognized rank method")
        return cand_patterns_uniq
