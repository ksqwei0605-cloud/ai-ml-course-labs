"""Formal Q2 sensitivity runs. Invoke with the course Python after the baseline."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    q2 = Path(__file__).resolve().parents[1] / "code/q2_clustering"
    root = q2 / "outputs" / "sensitivity_v1"
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads((q2 / "config.json").read_text())
    expanded = dict(base, candidate_k=list(range(5, 31)), seeds=[17, 42, 73, 101, 137])
    configs = {
        "raw": expanded,
        "scaled": dict(expanded, standardize=True),
        "sample": dict(expanded, silhouette_seed=20261012),
        "ch": dict(expanded, selection_metric="calinski_harabasz"),
        "db": dict(expanded, selection_metric="davies_bouldin"),
    }
    for name, config in configs.items():
        folder = root / name
        path = root / (name + ".json")
        path.write_text(json.dumps(config, indent=2))
        if not (folder / "export_manifest.json").exists():
            cmd = (["main.py"] if name in ("raw", "scaled") else
                   ["extras.py", "evaluate", "--experiment", str(root / "raw")])
            subprocess.run([sys.executable, *cmd, "--config", str(path), "--device", "cuda",
                            "--output", str(folder)], cwd=q2, check=True)
        with (root / (name + "_check.json")).open("w") as log:
            subprocess.run([sys.executable, "extras.py", "check-output", "--output", str(folder)],
                           cwd=q2, stdout=log, check=True)
        print(name, (folder / "selection.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
