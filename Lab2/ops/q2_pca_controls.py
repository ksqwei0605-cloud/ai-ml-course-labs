"""PCA dimension and GMM variance-floor controls on all rows, with raw-space scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
Q2 = ROOT / "code/q2_clustering"
RAW = Q2 / "outputs/sensitivity_v1/raw"
sys.path.insert(0, str(Q2))
from utils import runtime
import submission
from train import fit_gmm, fit_kmeans


def disagreement(predictions, reference):
    _, a = np.unique(predictions, return_inverse=True)
    _, b = np.unique(reference, return_inverse=True)
    counts = np.zeros((a.max() + 1, b.max() + 1), dtype=int)
    np.add.at(counts, (a, b), 1)
    row, col = linear_sum_assignment(-counts)
    return int(len(b) - counts[row, col].sum())


def rotated_parameter_control(projected, mean, basis, reference):
    """Project the raw GMM once, retaining the diagonal of each rotated covariance."""
    case = RAW / "candidates/gmm_k10_seed17"
    with np.load(case / "model.npz") as fitted:
        assert np.allclose(fitted["preprocessing_mean"], 0) and np.allclose(fitted["preprocessing_scale"], 1)
        means = (fitted["means"].astype(np.float64) - mean) @ basis
        variances = fitted["variances"].astype(np.float64) @ np.square(basis)
        constant = np.log(fitted["weights"].astype(np.float64)) - 0.5 * np.log(variances).sum(axis=1)
    labels = []
    for start in range(0, len(projected), 256):
        square = np.square(projected[start:start+256, None, :] - means)
        scores = constant - 0.5 * (square / variances).sum(axis=2)
        labels.extend(scores.argmax(axis=1))
    return dict(disagreements_kmeans=disagreement(labels, reference),
                disagreements_raw_gmm=int(np.count_nonzero(labels != np.load(case / "clusters.npy"))))


def main():
    runtime.configure("cpu", "float32", 4)
    with np.load(Q2 / "data/features.npz") as data:
        x = torch.from_numpy(data["X"])
    reference = np.load(RAW / "clusters.npy")
    ix = np.load(RAW / "diagnostic_indices.npy")
    assert x.shape == (56000, 512) and len(ix) == 3000
    centered = x.double()
    mean = centered.mean(0)
    centered -= mean
    eigenvalues, vectors = torch.linalg.eigh(centered.T @ centered / (len(x) - 1))
    eigenvalues = eigenvalues.flip([0])
    vectors = vectors.flip([1])
    projected = (centered @ vectors).float()
    retained = (eigenvalues.cumsum(0) / eigenvalues.sum()).tolist()
    records = ROOT / "records/q2_pca_controls"
    records.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(records / "pca.npz", mean=mean.numpy(), basis=vectors.numpy(), eigenvalues=eigenvalues.numpy())
    del centered
    rows, partitions = [], {}
    for dim in [0, 2, 5, 9, 32, 512]:
        fit_x = projected[:, :dim].contiguous() if dim else x
        for method in ["kmeans", "gmm"]:
            floors = [1e-6, 1e-4, 1e-2] if dim == 9 and method == "gmm" else [1e-4]
            for floor in floors:
                for seed in [17, 42]:
                    if method == "gmm":
                        fit = fit_gmm(fit_x, 10, seed, submission, reg_covar=floor)
                    else:
                        fit = fit_kmeans(fit_x, 10, seed, submission)
                    labels = np.asarray(fit["labels"])
                    assert labels.shape == reference.shape and len(np.unique(labels)) == 10
                    labs = torch.as_tensor(labels, dtype=torch.long)
                    key = f"{dim}_{method}_{floor}_{seed}"
                    partitions[key] = labels
                    rows.append(dict(space="pca" if dim else "raw", dimension=dim or 512, method=method,
                                     seed=seed, reg_covar=floor if method == "gmm" else None,
                                     explained=retained[dim-1] if dim else 1.,
                                     silhouette=float(submission.silhouette_score(x[ix], labs[ix])),
                                     db=float(submission.davies_bouldin_score(x, labs)),
                                     disagreements=disagreement(labels, reference),
                                     ari=float(adjusted_rand_score(labels, reference)),
                                     n_iter=fit["n_iter"], converged=fit["converged"],
                                     restart_outcomes=[r["stop_reason"] for r in fit["restarts"]]))
                    print(key, rows[-1]["disagreements"], flush=True)
    np.savez_compressed(records / "predictions.npz", **partitions)
    out = dict(input="course features, original row order", input_range=[x.min().item(), x.max().item()],
               n_rows=len(x), k=10, seeds=[17,42], n_init=3, torch=str(torch.__version__), device="cpu",
               pca="all-row centered covariance; float64 eigendecomposition; descending; no whitening",
               evaluation="raw 512D; DB full data; silhouette same 3000 diagnostic indices as formal raw run",
               reference="formal five-seed raw KMeans k10 seed17", retained_variance=retained, rows=rows,
               projected_parameters=rotated_parameter_control(projected.numpy(), mean.numpy(),
                                                              vectors.numpy(), reference))
    (ROOT / "logs/q2_pca_controls.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4), torch.no_grad():
        main()
