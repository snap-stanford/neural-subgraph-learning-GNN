from subgraph_matching import alignment
import argparse
from itertools import permutations
import pickle
from queue import PriorityQueue
import os
import random
import time
from tqdm import tqdm

from deepsnap.batch import Batch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from common import data
from common import models
from common import utils
from subgraph_matching.config import parse_encoder
from subgraph_matching.test import validation
from subgraph_matching.train import build_model

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report

from scipy.optimize import linear_sum_assignment

from baselines.fastPFP.fastPFP import fastPFP_faster, greedy_assignment, loss

import sys
sys.path.append("./baselines/ASAP/")
from ASAP import ASAP_main_G

import subprocess
from subprocess import STDOUT, check_output

import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def make_feat_mat(graph):
    m = [graph.nodes[v]["feat"] if "feat" in graph.nodes[v] else [1]
        for v in graph.nodes]
    return np.array(m)

def run_isorank(target, query, name):
    with open("data{}.inp".format(name), "w") as f:
        f.write(".\n")
        f.write("-\n")
        f.write("2\n")
        f.write("A{}\n".format(name))   # target
        f.write("B{}\n".format(name))   # query

    with open("A{}.tab".format(name), "w") as f:
        f.write("INTERACTOR_A\tINTERACTOR_B\n")
        for u, v in target.edges:
            f.write("a{}{}\ta{}{}\n".format(name, u, name, v))
            f.write("a{}{}\ta{}{}\n".format(name, v, name, u))

    with open("B{}.tab".format(name), "w") as f:
        f.write("INTERACTOR_A\tINTERACTOR_B\n")
        for u, v in query.edges:
            f.write("b{}{}\tb{}{}\n".format(name, u, name, v))
            f.write("b{}{}\tb{}{}\n".format(name, v, name, u))

    for n1, g1 in [("A", target), ("B", query)]:
        for n2, g2 in [("A", target), ("B", query)]:
            if (n1, n2) == ("B", "A"): continue
            with open("{}{}-{}{}.evals".format(n1, name, n2, name), "w") as f:
                for u in g1.nodes:
                    for v in g2.nodes:
                        s = (np.dot(g1.nodes[u]["feat"], g2.nodes[v]["feat"])
                            if "feat" in g1.nodes[u] else 1)
                        f.write("{}{}{} {}{}{} {}\n".format(n1.lower(), name,
                            u, n2.lower(), name, v, s))

    subprocess.run(["../isorank-n-v3-64/isorank-n-v3-64", "--K", "10",
        "--thresh", "1e-4", "--alpha", "0.9", "--maxveclen", "1000000",
        "--prefix", name, "data{}.inp".format(name)])

    with open("{}_match-score.txt".format(name), "r") as f:
        mat = np.zeros((len(target), len(query)))
        for line in f:
            u, v, score = line.strip().split(" ")
            score = float(score)
            mat[int(u[len(name)+1:]), int(v[len(name)+1:])] = score

    #row_ind, col_ind = linear_sum_assignment(-mat)
    #score = -mat[row_ind, col_ind].mean()
    score = np.mean(mat)
    #P = greedy_assignment(mat)
    #P = (X == X.max(1)[:, None])
    #score = loss(A, B, X, C=C, D=D, lam=1.0) / (len(target) *
    #    len(query))
    #print(score)
    #scores.append(-score)
    #mat = P
    ##mat = greedy_assignment(X)
    #row_ind, col_ind = np.nonzero(mat)
    #B = nx.to_numpy_array(query)
    #D = make_feat_mat(query)
    #adj_t = nx.to_numpy_array(target.subgraph(row_ind))
    #feat_t = make_feat_mat(target.subgraph(row_ind))
    #print(np.mean(adj_t == B), np.mean(feat_t @ D.T))
    #score = (np.mean(adj_t == B) + np.mean(feat_t @ D.T))

    return mat, score

def main():
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    parser = argparse.ArgumentParser(description='Alignment arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    parser.add_argument('--query_path', type=str, help='path of query graph',
        default="")
    parser.add_argument('--target_path', type=str, help='path of target graph',
        default="")
    parser.add_argument('--baseline', type=str, default="")
    parser.add_argument('--use_whole_targets', action="store_true")
    parser.add_argument('--agg_func', type=str, default="hungarian")
    args = parser.parse_args()
    args.test = True

    if args.baseline == "":
        model = build_model(args)

    print("USE WHOLE TARGETS:", args.use_whole_targets)
    assert args.use_whole_targets == (args.dataset in ["cox2", "enzymes", "msrc"])
    #data_source = data.PerturbTargetDataSource(args.dataset,
    #    node_anchored=False, use_whole_targets=args.use_whole_targets,
    #    use_feats=True,
    #    target_larger=False)
    if args.dataset in ["imdb-binary", "WN"]:  # too slow for random basis
        data_source = data.DiskDataSource(args.dataset, node_anchored=False)
    else:
        data_source = data.RandomBasisDataSource(args.dataset,
            edge_induced=args.edge_induced)
        #data_source = data.DiskImbalancedDataSource(args.dataset,
        #    node_anchored=False, use_whole_targets=True, use_feats=True,
        #    target_larger=True)
    #data_source = data.RandomBasisDataSource(args.dataset,
    #    node_anchored=True)
    n_samp = 64*100
    if args.method_type == "lrp":
        n_samp = 64*100
    if args.dataset in ["imdb-binary", "WN"]:
        n_samp = 3000

    loaders = data_source.gen_data_loaders(n_samp, 64, train=False)
    record_data = []
    labels, scores = [], []
    preds = []
    all_times = []
    start_time = time.time()
    if args.baseline:
        baseline_filter, baseline_order, baseline_engine = args.baseline.split(
            "-")
    for batch_target, batch_neg_target, batch_neg_query in tqdm(zip(*loaders)):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        for ls_a, ls_b, label in [(pos_a, pos_b, 1), (neg_a, neg_b, 0)]:
            if not ls_a: continue
            for target, query in zip(ls_a.G, ls_b.G):
                # make files
                target_fn = "/tmp/target-{}-{}".format(args.dataset,
                    args.baseline)
                query_fn = "/tmp/query-{}-{}".format(args.dataset,
                    args.baseline)
                with open(target_fn, "w") as f:
                    f.write("t {} {}\n".format(len(target), len(target.edges)))
                    for v in target.nodes:
                        f.write("v {} {} {}\n".format(v, 0, target.degree[v]))
                    for u, v in target.edges:
                        f.write("e {} {}\n".format(u, v))
                with open(query_fn, "w") as f:
                    f.write("t {} {}\n".format(len(query), len(query.edges)))
                    for v in query.nodes:
                        f.write("v {} {} {}\n".format(v, 0, query.degree[v]))
                    for u, v in query.edges:
                        f.write("e {} {}\n".format(u, v))
                start_time_query = time.time()

                if args.baseline:
                    if baseline_engine == "VF2":
                        try:
                            with timeout(seconds=60*10):
                                matcher = nx.algorithms.isomorphism.GraphMatcher(target, query)
                                matcher.subgraph_is_isomorphic()
                        except:
                            print("TIMEOUT")
                    else:
                        try:
                            check_output(["./baselines/SubgraphMatching/build/matching/SubgraphMatching.out",
                                "-d", target_fn,
                                "-q", query_fn,
                                "-filter", baseline_filter,
                                "-order", baseline_order,
                                "-engine", baseline_engine,
                                "-num", "1"], stderr=STDOUT, timeout=60*5)
                        except:
                            print("EXCEPTION")
                else:
                    if args.method_type == "lrp":
                        mat = 0
                        batch = utils.batch_nx_graphs([query])
                        emb_q = model.emb_model(batch)
                        batch = utils.batch_nx_graphs([target])
                        emb_t = model.emb_model(batch)
                        pred = model(emb_t, emb_q)
                        score = model.predict(pred)[0,1].item()
                    else:
                        mat = alignment.gen_alignment_matrix(model, query, target,
                            method_type=args.method_type)
                end_time_query = time.time()
                print(end_time_query - start_time_query)
                all_times.append(end_time_query - start_time_query)
                record_data.append((end_time_query - start_time_query,
                    len(query), len(target)))

                print(len(target), len(query))
                #assert len(target) >= len(query)
                #center = nx.center(query)[0]
                #scores.append(-np.mean(mat[center]))
                #row_ind, col_ind = linear_sum_assignment(mat)
                #score = -mat[row_ind, col_ind].sum()
                #scores.append(score)
    end_time = time.time()
    print("RUNTIME:", end_time - start_time)

    with open("data/runtime-expt-{}-{}.p".format(
        args.dataset, args.method_type if
        args.baseline == "" else args.baseline), "wb") as f:
        pickle.dump(record_data, f)
    #np.save("results/alignment.npy", mat)
    #print("Saved alignment matrix in results/alignment.npy")

    #plt.imshow(mat, interpolation="nearest")
    #plt.savefig("plots/alignment.png")
    #print("Saved alignment matrix plot in plots/alignment.png")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()


