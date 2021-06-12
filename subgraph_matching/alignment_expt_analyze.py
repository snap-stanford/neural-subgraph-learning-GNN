import networkx as nx
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import defaultdict

# note: remove '-hungarian' suffixes to read the older data files

#datasets = ["cox2", "dd", "msrc", "mmdb", "enzymes", "syn"]
datasets = ["imdb-binary", "WN"]
#methods = ["order", "mlp", "pfp", "isorank", "pseudo-hungarian"]
methods = ["order-hungarian", "mlp-hungarian", "pfp-hungarian",
    "isorank-hungarian", "pseudo-hungarian"]
#methods = ["pseudo-hungarian"]

ANALYZE_HUNGARIAN_FOR_ORDER = False
print("NOTE: ANALYZE HUNGARIAN FOR ORDER IS", ANALYZE_HUNGARIAN_FOR_ORDER)

if ANALYZE_HUNGARIAN_FOR_ORDER:
    methods = ["order"]

for ds in datasets:
    fn = "data/alignment-{}-pfp-hungarian.p".format(ds)
    if not os.path.exists(fn):
        print("SKIPPING", fn)
        continue

    with open(fn, "rb") as f:
        record_data = pickle.load(f)

    target_n_nodes, target_n_edges = [], []
    query_n_nodes, query_n_edges = [], []
    target_avg_deg, target_clust = [], []
    query_avg_deg, query_clust = [], []
    for target, query, score, mat, label, runtime in record_data:
        target_n_nodes.append(len(target.nodes))
        target_n_edges.append(len(target.edges))
        query_n_nodes.append(len(query.nodes))
        query_n_edges.append(len(query.edges))
        target_avg_deg.append(len(target.edges) / len(target.nodes))
        query_avg_deg.append(len(query.edges) / len(query.nodes))
        target_clust.append(nx.average_clustering(target))
        query_clust.append(nx.average_clustering(query))

    print("{}. T nodes: {:.4f}. Q nodes: {:.4f}. "
        "T edges: {:.4f}. Q edges: {:.4f}".format(ds,
            np.mean(target_n_nodes), np.mean(query_n_nodes),
            np.mean(target_n_edges), np.mean(query_n_edges)))
    print("{}. T avg deg: {:.4f}. Q avg deg: {:.4f}. "
        "T avg clust: {:.4f}. Q avg clust: {:.4f}".format(ds,
            np.mean(target_avg_deg), np.mean(query_avg_deg),
            np.mean(target_clust), np.mean(query_clust)))

runtime_groups_by_method = defaultdict(lambda: [[] for i in range(5)])
for method in methods:
    for ds in datasets:
        if not ANALYZE_HUNGARIAN_FOR_ORDER:
            fn = "data/alignment-{}-{}.p".format(ds, method)
        else:
            fn = "data/alignment-{}-{}-mean.p".format(ds, method)
        #print(fn)
        if not os.path.exists(fn):
            print("SKIPPING", fn)
            continue

        with open(fn, "rb") as f:
            record_data = pickle.load(f)

        scores, labels = [], []
        runtimes = []
        for target, query, score, mat, label, runtime in record_data:
            scores.append(score)
            labels.append(label)
            runtimes.append(runtime)

        scores = np.array(scores)
        labels = np.array(labels)
        runtimes = np.array(runtimes)
        #print(len(scores))
        group_size = len(scores) // 5

        aurocs_groups, runtimes_groups = [], []
        for i, s_idx in enumerate(range(0, len(scores), group_size)):
            e_idx = s_idx + group_size
            scores_group = scores[s_idx:e_idx]
            labels_group = labels[s_idx:e_idx]
            runtimes_group = runtimes[s_idx:e_idx]
            aurocs_groups.append(roc_auc_score(labels_group, scores_group))
            runtimes_groups.append(np.mean(runtimes_group))
            runtime_groups_by_method[method][i] += list(runtimes_group)

        print("{} {}. AUROC: {:.4f} ({:.4f}). Runtime: {:.4f} ({:.4f})".format(
            ds, method, np.mean(aurocs_groups), np.std(aurocs_groups, ddof=1),
            np.mean(runtimes_groups), np.std(runtimes_groups, ddof=1)))

for method, runtime_groups in runtime_groups_by_method.items():
    runtime_groups = [np.mean(l) for l in runtime_groups]
    print("{}. Runtime: {:.4f} ({:.4f})".format(method,
        np.mean(runtime_groups),
        np.std(runtime_groups, ddof=1)))
