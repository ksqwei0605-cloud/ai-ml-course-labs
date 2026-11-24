"""Ablate priors and diagonal variances at fixed fitted GMM means, without refitting."""
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
Q2 = ROOT / "code/q2_clustering"
RAW = Q2 / "outputs/sensitivity_v1/raw"


def main():
    with np.load(Q2 / "data/features.npz") as data:
        x = data["X"]
    case = RAW / "candidates/gmm_k10_seed17"
    with np.load(case / "model.npz") as fitted:
        means = fitted["means"].astype(np.float64)
        priors = fitted["weights"].astype(np.float64)
        variances = fitted["variances"].astype(np.float64)
        offset, scale = fitted["preprocessing_mean"], fitted["preprocessing_scale"]
    km = np.load(RAW / "clusters.npy")
    gmm = np.load(case / "clusters.npy")
    counts = np.zeros((len(means), len(means)), dtype=int)
    np.add.at(counts, (gmm, km), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = np.empty(len(rows), dtype=int)
    mapping[rows] = cols
    common_var = np.average(variances.mean(axis=1), weights=priors)
    log_prior = np.log(priors)
    log_det = np.log(variances).sum(axis=1)
    predictions = {k: [] for k in ["spherical_equal", "spherical_fitted", "diagonal_equal", "diagonal_fitted"]}
    # Hold priors uniform and separate the quadratic term from the log determinant.
    predictions.update(diagonal_distance_only=[], spherical_with_diagonal_logdet=[])
    for start in range(0, len(x), 256):
        z = (x[start:start+256].astype(np.float64) - offset) / scale
        square = np.square(z[:, None, :] - means)
        spherical = -0.5 * square.sum(axis=2) / common_var
        weighted = -0.5 * (square / variances).sum(axis=2)
        diagonal = weighted - 0.5 * log_det
        predictions["diagonal_distance_only"].append(weighted.argmax(axis=1))
        predictions["spherical_with_diagonal_logdet"].append((spherical - 0.5 * log_det).argmax(axis=1))
        for key, score in [("spherical", spherical), ("diagonal", diagonal)]:
            predictions[key+"_equal"].append(score.argmax(axis=1))
            predictions[key+"_fitted"].append((score + log_prior).argmax(axis=1))
    labels = {k: np.concatenate(chunks) for k, chunks in predictions.items()}
    if not np.array_equal(labels["diagonal_fitted"], gmm):
        raise ValueError("Independent GMM scoring disagrees with archived model")
    summary = {}
    for key, pred in labels.items():
        diff = np.flatnonzero(mapping[pred] != km)
        summary[key] = dict(different_from_kmeans=len(diff), different_from_fitted_gmm=int((pred != gmm).sum()),
                            zero_based_rows_vs_kmeans=diff.tolist())
    disagreed = summary["diagonal_fitted"]["different_from_kmeans"]
    output = dict(n=len(x), k=len(means), fit_seed=17, dtype="float64", fixed_means="archived GMM means",
                  pooled_scalar_variance=float(common_var), variants=summary,
                  probability_miss_all_disagreements={size: comb(len(x)-size, disagreed)/comb(len(x),disagreed)
                                                      for size in [1000, 3000, 10000, 28000, 56000]})
    (ROOT / "logs/q2_assignment_ablation.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
