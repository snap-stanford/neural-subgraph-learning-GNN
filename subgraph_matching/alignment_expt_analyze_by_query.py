import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

#datasets = ["cox2", "dd", "msrc", "mmdb", "enzymes", "syn"]
#methods = ["order", "mlp", "pfp", "isorank"]

datasets = ["cox2"]
#methods = ["order", "lrp-hungarian", "pseudo-hungarian", "pfp", "isorank"]
methods = ["order", "lrp-hungarian", "pfp", "isorank"]

ANALYZE_HUNGARIAN_FOR_ORDER = False
GET_DATASET_STATS = False

if ANALYZE_HUNGARIAN_FOR_ORDER:
    methods = ["order"]

if GET_DATASET_STATS:
    for ds in datasets:
        fn = "data/alignment-{}-pfp.p".format(ds)
        if not os.path.exists(fn):
            print("SKIPPING", fn)
            continue

        with open(fn, "rb") as f:
            record_data = pickle.load(f)

        target_n_nodes, target_n_edges = [], []
        query_n_nodes, query_n_edges = [], []
        for target, query, score, mat, label, runtime in record_data:
            target_n_nodes.append(len(target.nodes))
            target_n_edges.append(len(target.edges))
            query_n_nodes.append(len(query.nodes))
            query_n_edges.append(len(query.edges))
        print("{}. T nodes: {:.4f}. Q nodes: {:.4f}. "
            "T edges: {:.4f}. Q edges: {:.4f}".format(ds,
                np.mean(target_n_nodes), np.mean(query_n_nodes),
                np.mean(target_n_edges), np.mean(query_n_edges)))

runtime_groups_by_method = defaultdict(lambda: [[] for i in range(5)])
all_plots = defaultdict(list)
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
        query_sizes = []
        for target, query, score, mat, label, runtime in record_data:
            scores.append(score)
            labels.append(label)
            runtimes.append(runtime)
            query_sizes.append(len(query))

        scores = np.array(scores)
        if "lrp" in method:
            scores = -scores  # lrp doesn't generalize well to test distr...
        labels = np.array(labels)
        runtimes = np.array(runtimes)
        query_sizes = np.array(query_sizes)

        for i in range(1, 8):
            idxs = (query_sizes // 5) == i
            auroc = roc_auc_score(labels[idxs], scores[idxs])
            all_plots[method, ds].append(auroc)
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

for k, scores in all_plots.items():
    method, ds = k
    method = {"order": "NeuroMatch", "mlp": "NM-MLP", "isorank": "IsoRankN",
        "pfp": "FastPFP", "lrp-hungarian": "LRP",
        "pseudo-hungarian": "GQL-Approx"}[method]
    if ds == datasets[0]:
        plt.plot([5, 10, 15, 20, 25, 30, 35], scores, label=method, marker="o",
            linewidth=3)
plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
plt.xlabel("Query size")
plt.ylabel("Performance (AUROC)")
plt.xticks([5, 10, 15, 20, 25, 30, 35])
plt.savefig("results/score-by-query.png", bbox_inches="tight")
