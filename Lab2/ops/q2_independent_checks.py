"""Recompute selected labels and diagnostics using only the published features."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from threadpoolctl import threadpool_limits

base = Path(__file__).resolve().parents[1]
q2 = base / "code/q2_clustering"
series = q2 / "outputs/sensitivity_v1"

def loadj(path):
    return json.loads(path.read_text())

def main():
    folder = series / "raw"
    r = loadj(folder/"results.json")
    sel = r["selected"]
    assert sel["method"] == "kmeans"
    with np.load(q2/"data/features.npz") as d:
        X = d["X"].astype(np.float64)
    with np.load(folder/"model.npz") as d:
        means = d["centers"].astype(np.float64)
        origin, scale = d["preprocessing_mean"], d["preprocessing_scale"]
    labels = np.load(folder/"clusters.npy")
    sample = np.load(folder/"diagnostic_indices.npy")
    predictions, margins = [], []
    objective = 0.
    for start in range(0,len(X),512):
        z = (X[start:start+512]-origin)/scale
        # Direct differences in double precision, independent of the submitted dot product identity.
        dist = ((z[:,None,:]-means[None,:,:])**2).sum(axis=2)
        predictions.extend(dist.argmin(axis=1))
        ordered = np.sort(dist,axis=1)
        margins.extend((ordered[:,1]-ordered[:,0]).tolist())
        objective += ordered[:,0].sum()
    assert np.array_equal(predictions,labels)
    expected = {"silhouette":silhouette_score(X[sample],labels[sample]),
                "calinski_harabasz":calinski_harabasz_score(X,labels),
                "davies_bouldin":davies_bouldin_score(X,labels)}
    for key,val in expected.items():
        assert np.isclose(sel[key],val,rtol=2e-4,atol=1e-6), (key,sel[key],val)
    checks = {}
    for key in ("scaled","ch","db","sample","full_eval"):
        labs=np.load(series/key/"clusters.npy")
        checks[key]=dict(ari_vs_raw=adjusted_rand_score(labels,labs),
                         same_ids_fraction=float((labs==labels).mean()),
                         silhouette=loadj(series/key/"results.json")["selected"]["silhouette"])
    groups = loadj(folder/"groups.json")
    full_groups = loadj(series/"full_eval/groups.json")
    gaps=[abs(a["silhouette_mean"]-b["silhouette_mean"]) for a,b in zip(groups,full_groups,strict=True)]
    compare = {}
    with np.load(folder/"candidate_clusters.npz") as archive:
        for k in (9,10,11,12):
            for method in ("gmm","kmeans"):
                g=next(x for x in groups if x["k"]==k and x["method"]==method)
                parts=[archive[name] for name in g["run_keys"]]
                scores=[adjusted_rand_score(a,b) for a,b in combinations(parts,2)]
                compare[f"{method}_{k}"]=dict(seed_ari_mean=float(np.mean(scores)), seed_ari_min=float(min(scores)),
                    n_samples_in_smallest_cluster=min(int(np.unique(p,return_counts=True)[1].min()) for p in parts))
        compare["matched_method_k10_ari"]={str(seed):adjusted_rand_score(archive[f"kmeans_k10_seed{seed}"],archive[f"gmm_k10_seed{seed}"])
                                          for seed in r["config"]["seeds"]}
    out=dict(n_rows=len(labels),prediction_agreement=1.0,independent_metrics=expected,
             torch_metrics={k:sel[k] for k in expected},inertia_float64=float(objective),inertia_saved=sel["objective"],
             squared_distance_margin_quantiles=np.quantile(margins,[0,.01,.5,1]).tolist(),
             sensitivity=checks,max_group_silhouette_change_full_vs_sample=max(gaps),
             structural_comparison=compare)
    (base/"logs/q2_independent_checks.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
