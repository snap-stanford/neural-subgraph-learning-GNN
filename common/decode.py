"""Run the decoder model on a pairwise MCS task."""
import argparse
from collections import defaultdict
from itertools import permutations
import pickle
from queue import PriorityQueue
import random
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn

import data
import models
import utils

def arg_parse():
    parser = argparse.ArgumentParser(
        description='decode order embeddings into graphs')
    utils.parse_optimizer(parser)

    parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--margin', type=float,
                        help='margin for loss')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--dataset_type', type=str,
                        help='"real" or "syn"')
    parser.add_argument('--method_type', type=str,
                        help='method type ("end2end" or "order")')
    parser.add_argument('--test_set', type=str,
                        help='test set filename')
    parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    parser.add_argument('--intersect_model_path', type=str,
                        help='path to save/load intersect model')
    parser.add_argument('--plots_path', type=str,
                        help='path to save plots',
                        default="plots/decode")
    parser.add_argument('--out_path', type=str,
                        help='path to save outputs',
                        default="out/decode.txt")
    parser.add_argument('--n_iters', type=int,
                        help='number of decoder iterations',
                        default=3000)
    parser.add_argument('--discrete_reg', type=float,
                        help='discretization regularization factor',
                        default=1e-2)
    parser.add_argument('--match_reg', type=float,
                        help='graph matching regularization factor',
                        default=0)
    parser.add_argument('--normalize', action="store_true",
                        help='whether to normalize embeddings')
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--n_trials', type=int)
    parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    parser.set_defaults(conv_type='GIN',
                        method_type='order',
                        dataset='bigger',
                        dataset_type='syn',
                        n_layers=4,
                        batch_size=32,
                        hidden_dim=64,
                        dropout=0.0,
                        n_batches=1000,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=1e-4,
                        margin=0.1,
                        regularization=0,
                        test_set='',
                        eval_interval=100,
                        n_workers=4,
                        n_trials=20,
                        model_path="log/model.pt",
                        intersect_model_path="log/intersect-model.pt",
                        tag='',
                        use_harder_examples=True,
                        n_inner_layers=1,
                        node_anchored=False)

    return parser.parse_args()

# start_graphs should be in nx format
def decode_emb(model, emb, start_graphs, n_iters=3000, thresh=0.5,
    discrete_reg=1e-2, match_reg=1e-2, normalize=False):
    """Decode an embedding starting from a known supergraph.

    model: the order embedding model
    emb: the target embedding
    start_graph: the networkx graph to start masking from

    Returns:
    - out_graph: the decoded graph
    - emb_pred: the embedding of the decoded graph
    - x: the mask used on `start_graph` to get to the decoded graph
    """
    emb = emb.clone().detach()
    emb = torch.zeros_like(emb)
    emb[0] = 1.0
    batches = [utils.batch_nx_graphs([start_graph]) for start_graph in
        start_graphs]
    best_loss, best = float("inf"), None
    for trial_n in range(1):
        xs = [nn.Parameter(torch.randn((len(start_graph),
            1)).to(utils.get_device())) for start_graph
            in start_graphs]
        #xs = [nn.Parameter(torch.zeros((len(start_graph), 1))) for start_graph
        #    in start_graphs]
        opt = optim.Adam(xs, lr=0.01)
        #sched = optim.lr_scheduler.CosineAnnealingLR(opt, 400)
        for i in range(n_iters):
            opt.zero_grad()
            loss = 0
            emb_preds = []
            for batch, x in zip(batches, xs):
                batch.x = torch.sigmoid(x)#F.softmax(x, dim=1)
                if i % 100 == 0:
                    print(batch.x)
                emb_pred_c = model.emb_model(batch)
                emb_pred = emb_pred_c if not normalize else \
                    emb_pred_c / torch.norm(emb_pred_c)
                emb_preds.append(emb_pred)
                regularization = -torch.sum(batch.x*torch.log(batch.x) +
                    (1-batch.x)*torch.log(1-batch.x))
                #print(regularization.item())
                #loss += F.mse_loss(emb_pred.flatten(), emb.flatten()) 
                #TODO: uncomment line above for MSE
                loss += -torch.dot(emb_pred.flatten(), emb.flatten())
                loss += discrete_reg*regularization
                #oob_err = torch.max(emb_pred - emb, torch.zeros_like(emb))
                #print(oob_err)
                #oob_err = torch.sum(oob_err)
                #print(oob_err)
                #loss += 100*oob_err
                # TODO: oob error
            emb_preds = torch.stack(emb_preds)
            graphs_reg = torch.mean((emb_preds - torch.mean(emb_preds,
                dim=0))**2)
            loss += match_reg*graphs_reg
            if i % 100 == 0:
                print(graphs_reg)
                print(trial_n, i, loss.item())#, emb - emb_pred)
                print("DOT:", torch.dot(emb_pred.flatten(),
                    emb.flatten()).item())
            #print(emb_preds)
            loss.backward(retain_graph=True)
            opt.step()
            #sched.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                #best_idx = torch.argmin(torch.sum((emb_preds - emb)**2, dim=-1))
                #print(best_idx)
                #best = xs[best_idx]
                best_idx = 0
                best = xs[best_idx]

    #best_loss = float("inf")
    #for i, (batch, x) in enumerate(zip(batches, best)):
    #    batch.x = (x > thresh).type(torch.float) #F.softmax(x, dim=1)
    #    print(batch.x)
    #    emb_pred = model.emb_model(batch)
    #    loss = F.mse_loss(emb_pred, emb)
    #    if loss < best_loss:
    #        best_loss = loss
    #        best_idx = i
    #best = best[best_idx]

    x = torch.sigmoid(best)
    out_graph = start_graphs[best_idx].copy()
    out_graph = nx.convert_node_labels_to_integers(out_graph)
    for i, node in enumerate(x):
        score = node.item()
        print(score)
        if score < thresh:
            out_graph.remove_node(i)
    if len(out_graph) > 1:
        out_graph = out_graph.subgraph(list(sorted(nx.connected_components(
            out_graph), key=len))[-1])
    return out_graph, emb_pred, x

def test_decode_intersection(in_queue, out_queue, logger, args):
    """Test pairwise MCS task; outputs images to plots/

    in_queue: input queue to an intersection computation worker
    out_queue: output queue to an intersection computation worker
    logger: logger for recording progress
    args: commandline args
    """
    if args.method_type == "end2end":
        model = models.End2EndOrder(1, args.hidden_dim, args)
    elif args.method_type == "order" or args.method_type == "order-min":
        model = models.OrderEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    model.eval()
    model.load_state_dict(torch.load(args.model_path,
        map_location=utils.get_device()))

    if args.method_type == "order":
        intersect_model = models.MLPIntersecter(args.hidden_dim, 256)
        intersect_model.to(utils.get_device())
        intersect_model.eval()
        intersect_model.load_state_dict(torch.load(args.intersect_model_path,
            map_location=utils.get_device()))

    if args.dataset_type == "syn":
        with open("data/motifs-{}.p".format(args.dataset), "rb") as f:
            motifs = pickle.load(f)
            #for i, motif in enumerate(motifs):
            #    perm = dict(enumerate(np.random.permutation(range(len(motif)))))
            #    motifs[i] = nx.relabel_nodes(motif, perm)
        #motifs = gen_motifs(30)
        n_motifs_total = len(motifs)
        n_motifs_train = int(len(motifs)*0.8)
        #with open("data/pairs-{}.p".format(args.dataset), "rb") as f:
        #    pos_all = set(pickle.load(f))

        batch = utils.batch_nx_graphs(motifs)
        embs = model.emb_model(batch).detach().cpu().numpy()
        norms_by_n_edges = defaultdict(list)
        for emb, m in zip(embs, motifs):
            norms_by_n_edges[len(m.edges)].append(np.linalg.norm(emb))
        norms_by_n_edges = {k: np.mean(v) for k, v in norms_by_n_edges.items()}
    elif args.dataset_type == "real":
        dataset = data.load_dataset(args.dataset)
        _, test_set, task = dataset

    mses_pred, mses_pred_gte = [], []
    mses_pred_norm, mses_pred_gte_norm = [], []
    all_gccs, all_out_graphs, all_gt_out_graphs = [], [], []
    if args.seed is not None:
        random.seed(args.seed)
    for trial_n in range(args.n_trials):
        print("Trial", trial_n)
        if args.dataset_type == "syn":
            a = random.randint(n_motifs_train, len(motifs)-1)
            b = random.randint(n_motifs_train, len(motifs)-1)
        elif args.dataset_type == "real":
            a, b = 0, 1
            graph, nodes = data.sample_neigh(test_set, random.randint(20, 20))
            graph_a = graph.subgraph(nodes)
            graph, nodes = data.sample_neigh(test_set, random.randint(20, 20))
            graph_b = graph.subgraph(nodes)
            motifs = [graph_a, graph_b]
        if len(motifs[a]) > len(motifs[b]):
            a, b = b, a

        in_queue.put(("intersect", (motifs[a], motifs[b], (a, b))))
        gccs, (a, b) = out_queue.get()
        all_gccs.append(gccs)

        batch = utils.batch_nx_graphs([motifs[a], motifs[b]])
        emb_a, emb_b = model.emb_model(batch)
        if args.method_type == "end2end":
            _, intersect_pred = model((emb_a.unsqueeze(0),
                emb_b.unsqueeze(0)))#, torch.min(emb_a, emb_b).unsqueeze(0))
        elif args.method_type == "order":
            intersect_pred = intersect_model((emb_a.unsqueeze(0),
                emb_b.unsqueeze(0)))
        elif args.method_type == "order-min":
            intersect_pred = torch.min(emb_a.unsqueeze(0), emb_b.unsqueeze(0))
        intersect_pred = intersect_pred.squeeze(0)
        if args.normalize:
            emb_a /= torch.norm(emb_a)
            emb_b /= torch.norm(emb_b)
            intersect_pred /= torch.norm(intersect_pred)
        emb_gccs = model.emb_model(utils.batch_nx_graphs(gccs))
        if args.normalize:
            emb_gccs = (emb_gccs.T / torch.norm(emb_gccs, dim=-1)).T
        gcc_idx = torch.argmin(torch.sum((intersect_pred - emb_gccs)**2,
            axis=-1))
        emb_gcc, gcc = emb_gccs[gcc_idx], gccs[gcc_idx]
        print(len(motifs[a]), len(motifs[b]))
        print(motifs[a].edges)
        print(motifs[b].edges)
        pos1 = nx.spring_layout(motifs[a], iterations=100)
        pos2 = nx.spring_layout(motifs[b], iterations=100)
        pos3 = nx.spring_layout(gcc, iterations=100)
        ax = plt.subplot(231)
        ax.set_title("Graph A")
        nx.draw(motifs[a], pos1, node_size=50)
        #nx.draw_networkx_labels(motifs[a], pos1)
        ax = plt.subplot(232)
        ax.set_title("Graph B")
        nx.draw(motifs[b], pos2, node_size=50)
        #nx.draw_networkx_labels(motifs[b], pos2)
        ax = plt.subplot(234)
        ax.set_title("$A \cap B$")
        nx.draw(gcc, pos3, node_size=50)
        #nx.draw_networkx_labels(gcc, pos3)
        #plt.subplot(223)
        #nx.draw(gcc, pos3)

        #start_graphs = [motifs[a].copy(), motifs[b].copy()]
        start_graphs = [motifs[a].copy()]
        print(torch.norm(intersect_pred - emb_gcc))
        print(torch.norm(torch.min(emb_a, emb_b) - emb_gcc))
        #x = decode_emb(model, torch.min(emb_a, emb_b), motifs[a])
        # uncomment for min target
        out_graph, emb, _ = decode_emb(model, intersect_pred, start_graphs,
            n_iters=args.n_iters, discrete_reg=args.discrete_reg,
            match_reg=args.match_reg, normalize=args.normalize)
        all_out_graphs.append(out_graph)
        #out_graph, emb, _ = decode_emb(model, torch.min(emb_a, emb_b), start_graph)
        pos = nx.spring_layout(out_graph, iterations=100)
        ax = plt.subplot(235)
        ax.set_title("$A \cap B$ (predict+decode)")
        nx.draw(out_graph, pos, node_size=50)
        #nx.draw_networkx_labels(out_graph, pos)
        mse = F.mse_loss(emb.flatten(), emb_gcc.flatten()).item()
        mses_pred.append(mse)
        mses_pred_norm.append(mse / len(gcc.edges))

        out_graph, emb, _ = decode_emb(model, emb_gcc, start_graphs,
            n_iters=args.n_iters, discrete_reg=args.discrete_reg,
            match_reg=args.match_reg, normalize=args.normalize)
        all_gt_out_graphs.append(out_graph)
        pos = nx.spring_layout(out_graph, iterations=100)
        ax = plt.subplot(236)
        ax.set_title("$A \cap B$ (decode)")
        nx.draw(out_graph, pos, node_size=50)
        #nx.draw_networkx_labels(out_graph, pos)
        mse = F.mse_loss(emb.flatten(), emb_gcc.flatten()).item()
        mses_pred_gte.append(mse)
        mses_pred_gte_norm.append(mse / len(gcc.edges))

        plt.savefig(os.path.join(args.plots_path,
            "decode-{}.pdf".format(trial_n)), bbox_inches="tight")
        plt.savefig(os.path.join(args.plots_path,
            "decode-{}.png".format(trial_n)), bbox_inches="tight")
        plt.close()

    print("Pred RMSE: {:.4f}. Normalized: {:.4f}".format(
        np.sqrt(np.mean(mses_pred)), np.sqrt(np.mean(mses_pred_norm))))
    print("Pred RMSE with ground truth intersect emb: {:.4f}. "
        "Normalized: {:.4f}".format(
        np.sqrt(np.mean(mses_pred_gte)), np.sqrt(np.mean(mses_pred_gte_norm))))

    dists = eval_preds(all_gccs, all_out_graphs)
    dists_gt = eval_preds(all_gccs, all_gt_out_graphs)
    with open(args.out_path, "w") as f:
        for dist in dists:
            f.write("{}\n".format(dist))
        f.write("\n")
        for dist in dists_gt:
            f.write("{}\n".format(dist))

def eval_preds(all_gccs, all_out_graphs):
    dists = []
    for gccs, out_graph in zip(all_gccs, all_out_graphs):
        # remove the added self loop
        out_graph = out_graph.copy()
        out_graph.remove_edges_from(nx.selfloop_edges(out_graph))
        #print(out_graph.edges)
        for i in range(len(gccs)):
            gccs[i] = gccs[i].copy()
            #print(gccs[i].edges)
            gccs[i].remove_edges_from(nx.selfloop_edges(gccs[i]))
        # find min edit dist
        dist = min(nx.graph_edit_distance(gcc, out_graph) for gcc in gccs)
        dists.append(dist)
        print(dist)
    print("Mean edit dist from pred to nearest ground truth: {:.4f}".format(
        np.mean(dists)))
    return dists

def main():
    args = arg_parse()

    if not os.path.exists(args.plots_path):
        os.makedirs(args.plots_path)

    print("Starting {} workers".format(args.n_workers))
    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = []
    for i in range(args.n_workers):
        worker = mp.Process(target=utils.intersect_worker_func,
            args=(in_queue, out_queue))
        worker.start()
        workers.append(worker)

    print("Using dataset {}".format(args.dataset))
    logger = Logger("log-decode/")
    logger.add_args(args)

    test_decode_intersection(in_queue, out_queue, logger, args)
    #test_decode_subgraph(in_queue, out_queue, logger, args)

    for i in range(len(workers)):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()

if __name__ == '__main__':
    main()
