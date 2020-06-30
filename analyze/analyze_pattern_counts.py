import networkx as nx
import argparse
import json
import numpy as np
import pickle
from scipy.stats import ttest_rel, ttest_ind
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    #parser.add_argument('--graphlets_path', type=str)
    parser.add_argument('--counts_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.set_defaults(counts_path="results/counts.json")
    #parser.set_defaults(graphlets_path="out/out-graphlets.p")
    parser.set_defaults(out_path="results/analysis.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()

    all_counts = {}
    for fn in os.listdir(args.counts_path):
        if not fn.endswith(".json"): continue

        with open(os.path.join(args.counts_path, fn), "r") as f:
            graphlet_lens, n_matches, n_matches_bl = json.load(f)
            name = fn[:-5]
            all_counts[name] = graphlet_lens, n_matches

    all_labels, all_xs, all_ys, all_ub_ys, all_lb_ys = [], [], [], [], []
    for name, (sizes, counts) in all_counts.items():
        all_labels.append(name)

        matches_by_size = defaultdict(list)
        for i in range(len(sizes)):
            matches_by_size[sizes[i]].append(counts[i])

        #print("By size:")
        ys = []
        ub_ys, lb_ys = [], []
        for size in sorted(matches_by_size.keys()):
            #a, b = (stats.t.interval(0.95, len(matches_by_size[size]) - 1,
            #    loc=np.mean(np.log10(matches_by_size[size])),
            #    scale=stats.sem(np.log10(matches_by_size[size]))))
            #s = np.std(np.log10(matches_by_size[size]), ddof=1)
            #m = np.mean(np.log10(matches_by_size[size]))
            #a, b = m - s, m + s
            a, b = np.percentile(np.log10(matches_by_size[size]), [25, 75])

            ub_ys.append(b)
            lb_ys.append(a)
            #ys.append(np.mean(np.log10(matches_by_size[size])))
            ys.append(np.median(np.log10(matches_by_size[size])))

        all_xs.append(list(sorted(matches_by_size.keys())))
        all_ys.append(ys)
        all_ub_ys.append(ub_ys)
        all_lb_ys.append(lb_ys)

        #print("By size (log):")
        #for size in sorted(matches_by_size.keys()):
        #    print("- {}. N: {}. Mean log count: {:.4f}. Baseline: {:.4f}. "
        #        "Different with p={:.4f}".format(size, len(matches_by_size[size]),
        #            np.mean(np.log10(matches_by_size[size])),
        #            np.mean(np.log10(matches_by_size_bl[size])),
        #            ttest_ind(np.log10(matches_by_size[size]),
        #                np.log10(matches_by_size_bl[size])).pvalue))

    for i in range(len(all_xs)):
        sns.set()
        plt.plot(all_xs[i], np.power(10, all_ys[i]), label=all_labels[i],
            marker="o")
        plt.fill_between(all_xs[i], np.power(10, all_lb_ys[i]),
            np.power(10, all_ub_ys[i]), alpha=0.3)
    plt.xlabel("Graph size")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.legend()
    plt.savefig("plots/pattern-counts.png")
