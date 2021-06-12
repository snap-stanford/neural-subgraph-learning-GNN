import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
plt.rcParams.update({'font.size': 18})

# NOTE: change x axis limit to 40 for imdb plot
nice_names = {
    "order": "NeuroMatch",
    "CECI-CECI-CECI": "CECI",
    "CFL-CFL-EXPLORE": "CFL",
    "GQL-GQL-GQL": "GQL",
    "VF2-VF2-VF2": "VF2",
    "lrp": "LRP"
}

for dataset in ["WN", "ppi", "dd", "imdb-binary"]:
    for method in [
        "order",
        "CECI-CECI-CECI",
        "CFL-CFL-EXPLORE",
        #"DPiso-DPiso-DPiso",
        "GQL-GQL-GQL",
        "VF2-VF2-VF2"] + (
            ["lrp"] if dataset == "dd" else []):
        fn = "data/runtime-expt-{}-{}.p".format(dataset, method)
        if not os.path.exists(fn):
            print("SKIPPING", fn)
            continue
        with open(fn, "rb") as f:
            record_data = pickle.load(f)

        xs_scatter, ys_scatter = [], []
        qry_size_to_ts = defaultdict(list)
        for t, qry, tgt in record_data:
            if qry >= 5 and qry < 35:
                qry_size_to_ts[qry // 5].append(t)
                xs_scatter.append(qry)
                ys_scatter.append(t)
        print(min(qry_size_to_ts.keys()), max(qry_size_to_ts.keys()), method)
        #del qry_size_to_ts[max(qry_size_to_ts.keys())]

        xs, ys = zip(*sorted(qry_size_to_ts.items()))
        xs = np.array(xs) * 5
        ys = [np.mean(y) for y in ys]
        plt.plot(xs, ys, label=nice_names[method], marker="o",
            linewidth=3)
        #plt.scatter(xs_scatter, ys_scatter, label=nice_names[method], alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.xlabel("Query size")
    plt.ylabel("Runtime (avg sec/query)")
    plt.xticks([5, 10, 15, 20, 25, 30])#, 35])
    plt.yscale("log")
    plt.savefig("results/runtime-by-query-{}.png".format(dataset),
        bbox_inches="tight")
    plt.close()
