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

import subprocess

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
    parser.add_argument('--agg_func', type=str, default="mean")
    args = parser.parse_args()
    args.test = True

    model = build_model(args)

    print("USE WHOLE TARGETS:", args.use_whole_targets)
    assert args.use_whole_targets == (args.dataset in ["cox2", "enzymes", "msrc"])
    data_source = data.PerturbTargetDataSource(args.dataset,
        node_anchored=False, use_whole_targets=args.use_whole_targets,
        use_feats=True,
        target_larger=False)
        #data_source = data.DiskImbalancedDataSource(args.dataset,
        #    node_anchored=False, use_whole_targets=True, use_feats=True,
        #    target_larger=True)
    #data_source = data.RandomBasisDataSource(args.dataset,
    #    node_anchored=True)
    loaders = data_source.gen_data_loaders(64*100, 64, train=False)
    record_data = []
    labels, scores = [], []
    preds = []
    start_time = time.time()
    for batch_target, batch_neg_target, batch_neg_query in tqdm(zip(*loaders)):
        pos_a, pos_b, neg_a, neg_b = data_source.gen_batch(batch_target,
            batch_neg_target, batch_neg_query, False)
        for ls_a, ls_b, label in [(pos_a, pos_b, 1), (neg_a, neg_b, 0)]:
            if not ls_a: continue
            for target, query in zip(ls_a.G, ls_b.G):
                start_time_query = time.time()
                if args.baseline == "":
                    mat = alignment.gen_alignment_matrix(model, query, target,
                        method_type=args.method_type)
                    if args.agg_func == "mean":
                        score = -np.mean(mat)
                    elif args.agg_func == "min":
                        score = -np.min(mat)
                    elif args.agg_func == "hungarian":
                        row_ind, col_ind = linear_sum_assignment(mat)
                        score = -mat[row_ind, col_ind].sum()
                    elif args.agg_func == "hungarian-mean":
                        row_ind, col_ind = linear_sum_assignment(mat)
                        score = -mat[row_ind, col_ind].mean()

                    scores.append(score)
                elif args.baseline == "pfp":
                    A = nx.to_numpy_array(target)
                    B = nx.to_numpy_array(query)
                    C = make_feat_mat(target)
                    D = make_feat_mat(query)
                    X = fastPFP_faster(A, B, C=C, D=D, lam=1.0, alpha=0.5,
                        threshold1=1.0e-4, threshold2=1.0e-4, verbose=False)
                    P = greedy_assignment(X)
                    #P = (X == X.max(1)[:, None])
                    #score = loss(A, B, X, C=C, D=D, lam=1.0) / (len(target) *
                    #    len(query))
                    #print(score)
                    #scores.append(-score)
                    mat = P
                    #mat = greedy_assignment(X)
                    row_ind, col_ind = np.nonzero(mat)
                    adj_t = nx.to_numpy_array(target.subgraph(row_ind))
                    feat_t = make_feat_mat(target.subgraph(row_ind))
                    score = np.mean(adj_t == B) + np.mean(feat_t @ D.T)
                    scores.append(score)
                elif args.baseline == "isorank":
                    mat, score = run_isorank(target, query, args.dataset)
                    scores.append(score)
                end_time_query = time.time()

                print(len(target), len(query))
                assert len(target) >= len(query)
                #center = nx.center(query)[0]
                #scores.append(-np.mean(mat[center]))
                #row_ind, col_ind = linear_sum_assignment(mat)
                #score = -mat[row_ind, col_ind].sum()
                #scores.append(score)
                preds.append(1 if np.mean(mat) < 1 else 0)
                labels.append(label)
                record_data.append((target, query, score, mat, label,
                    end_time_query - start_time_query))

                try:
                    #print(scores)
                    auroc = roc_auc_score(labels, scores)
                    ap = average_precision_score(labels, scores)
                    acc = np.mean(np.equal(preds, labels))
                    print(args.dataset, acc, auroc, ap)
                except:
                    pass
    end_time = time.time()
    print("RUNTIME:", end_time - start_time)
    with open("data/alignment-{}-{}-{}.p".format(args.dataset, args.method_type if
        args.baseline == "" else args.baseline, args.agg_func), "wb") as f:
        pickle.dump(record_data, f)
    print("AGG FUNC:", args.agg_func)

    #np.save("results/alignment.npy", mat)
    #print("Saved alignment matrix in results/alignment.npy")

    #plt.imshow(mat, interpolation="nearest")
    #plt.savefig("plots/alignment.png")
    #print("Saved alignment matrix plot in plots/alignment.png")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()

