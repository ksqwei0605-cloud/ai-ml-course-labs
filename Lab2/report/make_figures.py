"""Render LAB2 figures and the complete K comparison from the archived course runs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
Q1 = ROOT / "code/q1_neural_network/outputs/20261123T155816Z"
OUT = REPORT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9, "axes.unicode_minus": False,
                     "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False})
colors = ["#14658a", "#be6825", "#237958", "#8d488a"]
styles = ["-", "--", "-.", ":"]
marks = ["o", "s", "^", "D"]


def finish(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def training(ax, runs):
    for row, color, style in zip(runs, colors, styles):
        h = row["history"]
        ax.semilogy([p["step"] for p in h], [p["train_mse"] for p in h], style, lw=1.7, color=color, label=row["group"])
    ax.set(xlabel="参数更新次数", ylabel="训练 MSE（对数轴）")
    ax.grid(axis="y", alpha=.18)
    ax.legend(ncol=2, fontsize=8)


def neural_figures():
    exp = json.loads((Q1 / "expression/results.json").read_text())["runs"]
    grad = json.loads((Q1 / "gradient/results.json").read_text())["runs"]
    with np.load(Q1 / "expression/predictions.npz") as data:
        xy = data["grid"]
        mid = np.flatnonzero(xy[:, 1] == 0)
        mid = mid[np.argsort(xy[mid, 0])]
        fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.5), layout="constrained")
        training(axes[0], exp)
        ax = axes[1]
        ax.plot(xy[mid, 0], data["reference"][mid, 0], color="#222222", lw=2.9, alpha=.6, label="目标")
        for group, color, style in zip("ABCD", colors, styles):
            ax.plot(xy[mid, 0], data[group][mid, 0], style, color=color, lw=1.7, label=group)
        ax.set(xlabel="横坐标 $x_1$（$x_2=0$）", ylabel="函数值")
        ax.set_ylim(-0.08, 1.34)
        ax.legend(ncol=3, fontsize=7.5, loc="upper center")
        finish(fig, "expression")
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), layout="constrained")
        order = np.lexsort((xy[:, 0], xy[:, 1]))
        for ax, group in zip(axes, "AD"):
            values = abs(data[group][:, 0] - data["reference"][:, 0])[order].reshape(65, 65)
            im = ax.imshow(values.clip(1e-6, 1), origin="lower", extent=(-1.5234375,1.5234375,-1.5234375,1.5234375),
                           norm=LogNorm(1e-6,1), interpolation="nearest", cmap="magma")
            ax.set(title=f"{group} 组", xlabel="$x_1$", ylabel="$x_2$", xticks=[-1,0,1], yticks=[-1,0,1])
        fig.colorbar(im, ax=axes, label="绝对误差", fraction=.035)
        finish(fig, "error_maps")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.45), layout="constrained",
                             gridspec_kw={"width_ratios":[1.2,1,1]})
    training(axes[0], grad)
    axes[0].set_title("训练轨迹")
    for ax, index, title in zip(axes[1:], [0,-1], ["更新前", "第 1,000 步"]):
        for row, color, style, marker in zip(grad, colors, styles, marks):
            layers = row["history"][index]["gradient_profile"]["layers"]
            ax.semilogy([v["layer"] for v in layers], [v["relative_to_output"] for v in layers],
                        style, marker=marker, fillstyle="none", ms=4, lw=1.3, color=color, label=row["group"])
        ax.set(title=title, xlabel="隐藏层（输入端 → 输出端）", xticks=[1,2,4,6,8], ylim=(1e-6,1))
        ax.grid(axis="y", alpha=.2)
    axes[1].set_ylabel("相对梯度 RMS")
    finish(fig, "gradient")


def clusters():
    with (REPORT / "evidence/groups.csv").open() as f:
        groups = {(r["method"],int(r["k"])): r for r in csv.DictReader(f) if r["experiment"] == "raw"}
    with (REPORT / "evidence/runs.csv").open() as f:
        runs = [r for r in csv.DictReader(f) if r["experiment"] == "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), layout="constrained")
    ks = np.arange(5,31)
    lines = []
    for k in ks:
        cells = [str(k)]
        for method in ["kmeans","gmm"]:
            g = groups[method,k]
            cohort = [r for r in runs if r["method"] == method and int(r["k"]) == k]
            assert len(cohort) == 5 and all(r["status"] == "succeeded" for r in cohort)
            cells.extend([f'{float(g["silhouette_mean"]):.4f}', f'{float(g["silhouette_std"]):.4f}',
                          str(min(int(r["min_size"]) for r in cohort))])
        lines.append(" & ".join(cells)+r" \\")
    table = [r"\begin{tabular}{rrrrrrr}\toprule",
             r"&\multicolumn{3}{c}{KMeans}&\multicolumn{3}{c}{对角 GMM}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"$K$ & 轮廓均值 & SD & 最小簇 & 轮廓均值 & SD & 最小簇\\\midrule",
             *lines, r"\bottomrule\end{tabular}"]
    (OUT / "grid_table.tex").write_text("\n".join(table)+"\n", encoding="utf-8")
    for method, name, color, style in [("kmeans","KMeans",colors[0],"-o"),("gmm","GMM",colors[1],"--s")]:
        axes[0].plot(ks,[float(groups[method,k]["silhouette_mean"]) for k in ks],style,color=color,ms=3,label=name)
        sizes = [min(int(r["min_size"]) for r in runs if r["method"] == method and int(r["k"]) == k) for k in ks]
        axes[1].semilogy(ks,sizes,style,color=color,ms=3,label=name)
    for ax in axes:
        ax.set_xlabel("簇数 K")
        ax.set_xticks([5,10,15,20,25,30])
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("抽样轮廓系数（种子均值）")
    axes[1].set_ylabel("五次运行中的最小簇人数")
    finish(fig, "cluster_grid")


if __name__ == "__main__":
    neural_figures()
    clusters()
