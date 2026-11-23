"""Locate disagreements between two predicted partitions, with arbitrary cluster names aligned."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

root = Path(__file__).resolve().parents[1]
series = root / "code/q2_clustering/outputs/sensitivity_v1"


def compare(first, second, samples):
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(counts, (a, b), 1)
    rows, cols = linear_sum_assignment(-counts)
    if len(cols) != counts.shape[1]:
        raise ValueError("Expected equal numbers of occupied clusters")
    mapping = np.zeros(counts.shape[1], dtype=int)
    mapping[cols] = rows
    different = a != mapping[b]
    return dict(total_disagreements=int(different.sum()),
                zero_based_rows=np.flatnonzero(different).tolist(),
                evaluation_sample={case: dict(size=len(ix), disagreements=int(different[ix].sum()))
                                   for case, ix in samples.items()})


if __name__ == "__main__":
    config = json.loads((series / "raw/config.json").read_text())
    samples = {case: np.load(series / case / "diagnostic_indices.npy") for case in ["raw", "sample", "full_eval"]}
    with np.load(series / "raw/candidate_clusters.npz") as runs:
        by_seed = {s: compare(runs[f"kmeans_k10_seed{s}"], runs[f"gmm_k10_seed{s}"], samples) for s in config["seeds"]}
    result = dict(k=10, comparison="KMeans and GMM predictions; labels aligned using only predictions",
                  rows_zero_based=True, by_fit_seed=by_seed)
    (root / "logs/q2_partition_difference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(by_seed[17], indent=2))
