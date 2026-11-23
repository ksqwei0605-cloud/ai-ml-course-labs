"""Re-fit the formal raw grid with a higher GMM step cap and pair results by seed."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
q2 = root / "code/q2_clustering"
series = q2 / "outputs/sensitivity_v1"


def read(path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    case = series / "gmm_iter400"
    config_path = case.with_suffix(".json")
    config = dict(read(series / "raw/config.json"), gmm_max_iter=400)
    config_path.write_text(json.dumps(config, indent=2))
    if not (case / "export_manifest.json").exists():
        subprocess.run([sys.executable, "main.py", "--device", "cuda", "--config", str(config_path),
                        "--output", str(case)], cwd=q2, check=True)
    output = subprocess.check_output([sys.executable, "extras.py", "check-output", "--output", str(case)], cwd=q2)
    (series / "gmm_iter400_check.json").write_bytes(output)
    rows = []
    for name in ("raw", "gmm_iter400"):
        folder = series / name
        runs = read(folder / "runs.json")
        gmm = [r for r in runs if r["method"] == "gmm"]
        groups = [g for g in read(folder / "groups.json") if g["eligible"]]
        rows.append(dict(case=name, succeeded=sum(r["status"] == "succeeded" for r in runs),
                         gmm_capped=sum(r["stop_reason"] == "max_iter" for r in gmm),
                         gmm_converged=sum(r["converged"] for r in gmm),
                         per_k={g["k"]: dict(capped=sum(r["k"] == g["k"] and r["stop_reason"] == "max_iter" for r in gmm),
                                             silhouette=g["silhouette_mean"]) for g in groups if g["method"] == "gmm"},
                         winner=read(folder / "export_manifest.json")))
    (root / "logs/q2_iteration_cap.json").write_text(json.dumps(rows, indent=2) + "\n")
    print([(r["case"], r["gmm_capped"], r["winner"]["run_key"]) for r in rows])
