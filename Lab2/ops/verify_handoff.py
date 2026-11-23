"""Audit local copies against formal course manifests; optionally stage the four experimental files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q2 = ROOT / "code/q2_clustering"
STAMP = "20261123T155816Z"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check(value, reason):
    if not value:
        raise ValueError(reason)


def verify_q1():
    result = {}
    for task in ("expression", "gradient"):
        folder = ROOT / "code/q1_neural_network/outputs" / STAMP / task
        run = read(folder / "results.json")
        check(run["torch"] == "2.11.0+cu128" and run["device"] == "cpu", task)
        check(len(run["runs"]) == 4 and {r["group"] for r in run["runs"]} == set("ABCD"), task)
        points = {}
        with np.load(folder / "predictions.npz") as arrays:
            ref = arrays["reference"].astype("float64")
            for r in run["runs"]:
                key = r["group"]
                pred = arrays[key]
                check(pred.shape == ref.shape == (4225, 1), f"{task}/{key}: grid")
                mse = np.mean(np.square(pred - ref))
                check(np.isclose(mse, r["grid_mse"], rtol=1e-6, atol=1e-12), f"{task}/{key}: MSE")
                steps = [h["step"] for h in r["history"]]
                check(steps == list(range(0, r["steps"] + 1, 100)), f"{task}/{key}: steps")
                if task == "gradient":
                    for h in (r["history"][0], r["history"][-1]):
                        layers = h["gradient_profile"]["layers"]
                        check(len(layers) == len(r["widths"]), f"{key}: layers")
                        check(all(np.isfinite(l["relative_to_output"]) for l in layers), f"{key}: finite")
                points[key] = float(mse)
        for name in ["summary.csv", "models.pt", "training.png",
                     "fitting.png" if task == "expression" else "gradients.png"]:
            check((folder / name).stat().st_size > 0, str(folder / name))
        result[task] = points
    return result


def verify_q2(folder, count, input_sha, code_sha):
    plan = read(folder / "plan.json")
    manifest = read(folder / "manifest.json")
    export = read(folder / "export_manifest.json")
    check(plan["input_sha256"] == input_sha and plan["implementation_sha256"] == code_sha, str(folder))
    check(plan["development"] is False and plan["teacher_labels_used"] is False, str(folder))
    check(manifest["status"] == "completed" and manifest["successful_fits"] == count
          and manifest["planned_runs"] == manifest["completed_runs"] == count, str(folder))
    expected = read(folder / "config.json")
    keys = {f"{m}_k{k}_seed{s}" for m in ["kmeans", "gmm"] for k in expected["candidate_k"] for s in expected["seeds"]}
    runs = read(folder / "runs.json")
    check(len(runs) == count and {r["key"] for r in runs} == keys, f"{folder}: scheduled grid")
    converged = 0
    for r in runs:
        sub = folder / "candidates" / r["key"]
        check(r["status"] == "succeeded" and sum(r["cluster_sizes"]) == 56000, str(sub))
        check(sha(sub / "model.npz") == r["model_sha256"] and sha(sub / "clusters.npy") == r["clusters_sha256"], str(sub))
        converged += int(r["converged"])
    check(sha(folder / "clusters.npy") == export["clusters_sha256"], f"{folder}: export")
    check(sha(folder / "model.npz") == export["model_sha256"], f"{folder}: model")
    labels = np.load(folder / "clusters.npy", allow_pickle=False)
    ids, sizes = np.unique(labels, return_counts=True)
    check(labels.shape == (56000,) and labels.dtype.kind in "iu" and ids.min() >= 0 and 5 <= len(ids) <= 30, str(folder))
    return dict(candidate_count=count, selected_runs_converged=converged, winner=export["run_key"],
                clusters_sha256=export["clusters_sha256"], actual_k=len(ids),
                min_size=int(sizes.min()), max_size=int(sizes.max()))


def main(stage):
    q1 = verify_q1()
    input_sha = sha(Q2 / "data/features.npz")
    check(input_sha == (Q2 / "data/SHA256SUMS.txt").read_text().split()[0], "released features")
    src_sha = sha(Q2 / "submission.py")
    ext = Q2 / "outputs/sensitivity_v1"
    folders = [(Q2 / "outputs" / f"baseline_{STAMP}", 32)]
    folders.extend((ext / name, 40 if name == "restarts10" else 260)
                   for name in ["raw", "scaled", "ch", "db", "sample", "full_eval", "restarts10", "gmm_iter400"])
    q2 = {f.name: verify_q2(f, n, input_sha, src_sha) for f, n in folders}
    candidate = ext / "raw/clusters.npy"
    sources = {ROOT / "code/q1_neural_network/models.py": "code/q1_neural_network/models.py",
               ROOT / "code/q1_neural_network/train.py": "code/q1_neural_network/train.py",
               Q2 / "submission.py": "code/q2_clustering/submission.py", candidate: "code/data/clusters.npy"}
    cloud_sha = dict(reversed(line.split(maxsplit=1)) for line in
                     (ROOT / "logs/submit_sources.sha256").read_text().splitlines())
    target = ROOT / "submit"
    for src, dest in sources.items():
        check(sha(src) == cloud_sha[src.relative_to(ROOT).as_posix()], str(src))
        path = target / dest
        if stage:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                check(sha(path) == sha(src), f"Existing staged file differs: {path}")
            else:
                shutil.copyfile(src, path)
        check(sha(path) == sha(src), str(path))
    files = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    check(files.difference({"report.pdf"}) == set(sources.values()), "unexpected/missing file in submit/")
    out = dict(official_run=STAMP, features_sha256=input_sha, q1_grid_mse_recomputed=q1, q2=q2,
               submission={dest: sha(target / dest) for dest in sources.values()}, report_pdf_present="report.pdf" in files)
    print(json.dumps(out, indent=2))
    (ROOT / "logs/local_handoff_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="copy verified experimental files into submit/")
    main(parser.parse_args().stage)
