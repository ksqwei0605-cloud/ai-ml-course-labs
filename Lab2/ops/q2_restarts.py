"""One-factor restart-budget comparison near the selected K on the formal data."""
import json
import subprocess
import sys
from pathlib import Path

from sklearn.metrics import adjusted_rand_score
import numpy as np

root = Path(__file__).resolve().parents[1]
q2 = root / "code/q2_clustering"
base = q2 / "outputs/sensitivity_v1"
read = lambda p: json.loads(p.read_text(encoding="utf-8"))


def run():
    ks = [9, 10, 11, 12]
    ref = read(base / "raw/config.json")
    config = dict(ref, candidate_k=ks, n_init=10)
    case = base / "restarts10"
    config_path = base / "restarts10.json"
    config_path.write_text(json.dumps(config, indent=2))
    if not (case / "export_manifest.json").exists():
        subprocess.run([sys.executable, "main.py", "--device", "cuda", "--config", str(config_path),
                        "--output", str(case)], cwd=q2, check=True)
    check = subprocess.check_output([sys.executable, "extras.py", "check-output", "--output", str(case)], cwd=q2)
    (base / "restarts10_check.json").write_bytes(check)
    sources = [base / "raw", case]
    runs = [{r["key"]: r for r in read(p / "runs.json") if r["k"] in ks} for p in sources]
    if runs[0].keys() != runs[1].keys() or not all(
        r["status"] == "succeeded" for mapping in runs for r in mapping.values()
    ):
        raise RuntimeError("Incomplete paired seed/method/K comparison")
    groups = []
    with np.load(sources[0] / "candidate_clusters.npz") as labels3, np.load(case / "candidate_clusters.npz") as labels10:
        for method in ["kmeans", "gmm"]:
            for k in ks:
                keys = [f"{method}_k{k}_seed{s}" for s in config["seeds"]]
                ari = [adjusted_rand_score(labels3[key], labels10[key]) for key in keys]
                groups.append(dict(method=method, k=k, seed_ari_3_vs_10=ari,
                                   silhouette_means=[np.mean([m[key]["metrics"]["silhouette"]["value"] for key in keys]) for m in runs],
                                   objective_means=[np.mean([m[key]["objective"] for key in keys]) for m in runs],
                                   elapsed_medians=[np.median([m[key]["elapsed_seconds"] for key in keys]) for m in runs]))
    result = dict(k_values=ks, seeds=config["seeds"], budgets=[3,10], groups=groups,
                  candidate=read(case / "export_manifest.json"),
                  final_partition_ari=adjusted_rand_score(np.load(sources[0]/"clusters.npy"), np.load(case/"clusters.npy")))
    (root / "logs/q2_restart_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["candidate"], flush=True)


if __name__ == "__main__":
    run()
