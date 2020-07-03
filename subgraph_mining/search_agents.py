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

import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

from common import data
from common import models
from common import utils
from common import combined_syn
from subgraph_mining.config import parse_decoder
from subgraph_matching.config import parse_encoder

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
    """ Class for search strategies to identify frequent subgraphs in embedding space.

    The problem is formulated as a search. The first action chooses a seed node to grow from.
    Subsequent actions chooses a node in dataset to connect to the existing subgraph pattern,
    increasing the pattern size by 1.

    See paper for rationale and algorithm details.
    """
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20):
        """ Subgraph pattern search by walking in embedding space.

        Args:
            min_pattern_size: minimum size of frequent subgraphs to be identified.
            max_pattern_size: maximum size of frequent subgraphs to be identified.
            model: the trained subgraph matching model (PyTorch nn.Module).
            dataset: the DeepSNAP dataset for which to mine the frequent subgraph pattern.
            embs: embeddings of sampled node neighborhoods (see paper).
            node_anchored: an option to specify whether to identify node_anchored subgraph patterns.
                node_anchored search procedure has to use a node_anchored model (specified in subgraph
                matching config.py).
            analyze: whether to enable analysis visualization.
            model_type: type of the subgraph matching model (requires to be consistent with the model parameter).
            out_batch_size: the number of frequent subgraphs output by the mining algorithm for each size.
                They are predicted to be the out_batch_size most frequent subgraphs in the dataset.
        """
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.model = model
        self.dataset = dataset
        self.embs = embs
        self.node_anchored = node_anchored
        self.analyze = analyze
        self.model_type = model_type
        self.out_batch_size = out_batch_size

    def run_search(self, n_trials=1000): 
        self.cand_patterns = defaultdict(list)
        self.counts = defaultdict(lambda: defaultdict(list))
        self.n_trials = n_trials

        self.init_search()
        while not self.is_search_done():
            self.step()
        return self.finish_search()

    def init_search():
        raise NotImplementedError

    def step(self):
        """ Abstract method for executing a search step.
        Every step adds a new node to the subgraph pattern.
        Run_search calls step at least min_pattern_size times to generate a pattern of at least this
        size. To be inherited by concrete search strategy implementations.
        """
        raise NotImplementedError

class MCTSSearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, model_type="order",
        out_batch_size=20, c_uct=0.7):
        """ MCTS implementation of the subgraph pattern search.
        Uses MCTS strategy to search for the most common pattern.

        Args:
            c_uct: the exploration constant used in UCT criteria (See paper).
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.c_uct = c_uct
        assert not analyze

    def init_search(self):
        self.wl_hash_to_graphs = defaultdict(list)
        self.cum_action_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(float))
        self.visited_seed_nodes = set()
        self.max_size = self.min_pattern_size

    def is_search_done(self):
        return self.max_size == self.max_pattern_size + 1

    # returns whether there are at least n nodes reachable from start_node in graph
    def has_min_reachable_nodes(self, graph, start_node, n):
        for depth_limit in range(n+1):
            edges = nx.bfs_edges(graph, start_node, depth_limit=depth_limit)
            nodes = set([v for u, v in edges])
            if len(nodes) + 1 >= n:
                return True
        return False

    def step(self):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        print("Size", self.max_size)
        print(len(self.visited_seed_nodes), "distinct seeds")
        for simulation_n in tqdm(range(self.n_trials //
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
                if node_score > best_score:
                    best_score = node_score
                    best_graph_idx = cand_graph_idx
                    best_start_node = cand_start_node
            # if existing seed beats choosing a new seed
            if best_score >= self.c_uct * np.sqrt(np.log(simulation_n or 1)):
                graph_idx, start_node = best_graph_idx, best_start_node
                assert best_start_node in self.dataset[graph_idx].nodes
                graph = self.dataset[graph_idx]
            else:
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
            while frontier and len(neigh) < self.max_size:
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
                self.cum_action_values[state_list[i]][
                    state_list[i+1]] += best_v_score
                self.visit_counts[state_list[i]][state_list[i+1]] += 1
        self.max_size += 1

    def finish_search(self):
        counts = defaultdict(lambda: defaultdict(int))
        for _, v in self.visit_counts.items():
            for s2, count in v.items():
                counts[len(random.choice(self.wl_hash_to_graphs[s2]))][s2] += count

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            for wl_hash, count in sorted(counts[pattern_size].items(), key=lambda
                x: x[1], reverse=True)[:self.out_batch_size]:
                cand_patterns_uniq.append(random.choice(
                    self.wl_hash_to_graphs[wl_hash]))
                print("- outputting", count, "motifs of size", pattern_size)
        return cand_patterns_uniq

class GreedySearchAgent(SearchAgent):
    def __init__(self, min_pattern_size, max_pattern_size, model, dataset,
        embs, node_anchored=False, analyze=False, rank_method="counts",
        model_type="order", out_batch_size=20, n_beams=1):
        """Greedy implementation of the subgraph pattern search.
        At every step, the algorithm chooses greedily the next node to grow while the pattern
        remains predicted to be frequent. The criteria to choose the next action depends
        on the score predicted by the subgraph matching model 
        (the actual score is determined by the rank_method argument).

        Args:
            rank_method: greedy search heuristic requires a score to rank the
                possible next actions. 
                If rank_method=='counts', counts of the pattern in search tree is used;
                if rank_method=='margin', margin score of the pattern predicted by the matching model is
                    used.
                if rank_method=='hybrid', it considers both the count and margin to rank the actions.
        """
        super().__init__(min_pattern_size, max_pattern_size, model, dataset,
            embs, node_anchored=node_anchored, analyze=analyze,
            model_type=model_type, out_batch_size=out_batch_size)
        self.rank_method = rank_method
        self.n_beams = n_beams
        print("Rank Method:", rank_method)

    def init_search(self):
        ps = np.array([len(g) for g in self.dataset], dtype=np.float)
        ps /= np.sum(ps)
        graph_dist = stats.rv_discrete(values=(np.arange(len(self.dataset)), ps))

        beams = []
        for trial in range(self.n_trials):
            graph_idx = np.arange(len(self.dataset))[graph_dist.rvs()]
            graph = self.dataset[graph_idx]
            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])
            beams.append([(0, neigh, frontier, visited, graph_idx)])
        self.beam_sets = beams
        self.analyze_embs = []

    def is_search_done(self):
        return len(self.beam_sets) == 0

    def step(self):
        new_beam_sets = []
        print("seeds come from", len(set(b[0][-1] for b in self.beam_sets)),
            "distinct graphs")
        analyze_embs_cur = []
        for beam_set in tqdm(self.beam_sets):
            new_beams = []
            for _, neigh, frontier, visited, graph_idx in beam_set:
                graph = self.dataset[graph_idx]
                if len(neigh) >= self.max_pattern_size or not frontier: continue
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
                    if score < best_score:
                        best_score = score
                        best_node = cand_node
                    new_frontier = list(((set(frontier) |
                        set(graph.neighbors(cand_node))) - visited) -
                        set([cand_node]))
                    new_beams.append((
                        score, neigh + [cand_node],
                        new_frontier, visited | set([cand_node]), graph_idx))
            new_beams = list(sorted(new_beams, key=lambda x:
                x[0]))[:self.n_beams]
            for score, neigh, frontier, visited, graph_idx in new_beams[:1]:
                graph = self.dataset[graph_idx]
                # add to record
                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))
                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0
                self.cand_patterns[len(neigh_g)].append((score, neigh_g))
                if self.rank_method in ["counts", "hybrid"]:
                    self.counts[len(neigh_g)][utils.wl_hash(neigh_g,
                        node_anchored=self.node_anchored)].append(neigh_g)
                if self.analyze and len(neigh) >= 3:
                    emb = self.model.emb_model(utils.batch_nx_graphs(
                        [neigh_g], anchors=[neigh[0]] if self.node_anchored
                        else None)).squeeze(0)
                    analyze_embs_cur.append(emb.detach().cpu().numpy())
            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)
        self.beam_sets = new_beam_sets
        self.analyze_embs.append(analyze_embs_cur)

    def finish_search(self):
        if self.analyze:
            print("Saving analysis info in results/analyze.p")
            with open("results/analyze.p", "wb") as f:
                pickle.dump((self.cand_patterns, self.analyze_embs), f)
            xs, ys = [], []
            for embs_ls in self.analyze_embs:
                for emb in embs_ls:
                    xs.append(emb[0])
                    ys.append(emb[1])
            print("Saving analysis plot in results/analyze.png")
            plt.scatter(xs, ys, color="red", label="motif")
            plt.legend()
            plt.savefig("plots/analyze.png")
            plt.close()

        cand_patterns_uniq = []
        for pattern_size in range(self.min_pattern_size, self.max_pattern_size+1):
            if self.rank_method == "hybrid":
                cur_rank_method = "margin" if len(max(
                    self.counts[pattern_size].values(), key=len)) < 3 else "counts"
            else:
                cur_rank_method = self.rank_method

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
                    cand_patterns_uniq.append(random.choice(neighs))
            else:
                print("Unrecognized rank method")
        return cand_patterns_uniq
