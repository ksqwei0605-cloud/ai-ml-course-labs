"""Build report tables from the formal outputs. Run inside the cloud LAB2 directory."""
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "report/evidence"
DEST.mkdir(parents=True, exist_ok=True)
Q2 = ROOT / "code/q2_clustering/outputs/sensitivity_v1"

def read(path):
    return json.loads(path.read_text())

def csv_file(name, rows):
    with (DEST / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    q1 = ROOT / "code/q1_neural_network/outputs" / sys.argv[1]
    summary = {}
    for task in ("expression","gradient"):
        folder = q1 / task
        r = read(folder / "results.json")
        assert r["torch"] == "2.11.0+cu128"
        for image in folder.glob("*.png"):
            shutil.copyfile(image, DEST / (task+"_"+image.name))
        summary[task] = []
        with np.load(folder / "predictions.npz") as d:
            ix = [np.argmin(((d["grid"]-[x,0])**2).sum(axis=1)) for x in (-0.65,0,0.65)]
            for row in r["runs"]:
                group = row["group"]
                out = {key:row[key] for key in ("group","parameters","train_mse","grid_mse","steps")}
                if task == "expression":
                    out["section_values"] = d[group][ix,0].tolist()
                else:
                    out["first_relative_gradient"] = [row["history"][i]["gradient_profile"]["layers"][0]["relative_to_output"] for i in (0,-1)]
                    out["last_relative_gradient"] = [row["history"][i]["gradient_profile"]["layers"][-1]["relative_to_output"] for i in (0,-1)]
                summary[task].append(out)
            if task == "expression":
                summary["section_reference"] = d["reference"][ix,0].tolist()
    summary["experiments"] = []
    all_groups, all_runs = [], []
    for name in ("raw","scaled","sample","ch","db"):
        folder = Q2 / name
        check = read(Q2 / (name+"_check.json"))
        assert check["passed"] and check["assigned"]==56000
        r = read(folder/"results.json")
        assert r["environment"]["torch"]=="2.11.0+cu128"
        g = r["selection"]
        group = next(x for x in r["groups"] if (x["method"],x["k"]) == (g["method"],g["k"]))
        run = r["selected"]
        with np.load(folder/"candidate_clusters.npz") as archive:
            labels = [archive[k] for k in group["run_keys"]]
            aris = [adjusted_rand_score(a,b) for a,b in itertools.combinations(labels,2)]
        out = dict(name=name, method=g["method"], k=g["k"], seed=run["seed"],
                   silhouette=group["silhouette_mean"], ch=group["calinski_harabasz_mean"],
                   db=group["davies_bouldin_mean"], stability_mean=np.mean(aris), stability_min=min(aris),
                   min_size=min(run["cluster_sizes"]),max_size=max(run["cluster_sizes"]),
                   covered=run["sample_covered_clusters"], failures=sum(x["status"]!="succeeded" for x in r["runs"]),
                   seconds=r["elapsed_seconds"],hash=r["clusters_sha256"])
        summary["experiments"].append(out)
        if name in ("raw","scaled"):
            for x in r["groups"]:
                all_groups.append(dict(experiment=name,method=x["method"],k=x["k"],eligible=x["eligible"],
                         **{k:v for k,v in x.items() if k.endswith(("_mean","_std"))}))
            for x in r["runs"]:
                all_runs.append(dict(experiment=name,method=x["method"],k=x["k"],seed=x["seed"],status=x["status"],
                       actual_k=x.get("actual_k"),min_size=min(x.get("cluster_sizes") or [0]),
                       max_size=max(x.get("cluster_sizes") or [0]),objective=x.get("objective"),
                       silhouette=x.get("silhouette"),ch=x.get("calinski_harabasz"),db=x.get("davies_bouldin")))
        if name=="raw":
            summary["env"]=r["environment"]
            summary["protocol"]=check
    csv_file("groups.csv",all_groups)
    csv_file("runs.csv",all_runs)
    (DEST/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary["experiments"],indent=2))

if __name__ == "__main__":
    main()
