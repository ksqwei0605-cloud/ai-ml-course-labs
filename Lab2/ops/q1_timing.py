"""Repeat the course expression runs on one CPU thread and verify the grid MSE."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
q1 = root / "code/q1_neural_network"
sys.path.insert(0, str(q1))
from main import PLANS
from models import MLP
from train import train
from utils.data import load_data, load_initial_states


def collect(stamp):
    torch.set_num_threads(1)
    data = load_data("expression", "cpu")
    states = load_initial_states("expression")
    official = json.loads((q1 / "outputs" / stamp / "expression/results.json").read_text())
    reference = {row["group"]: row["grid_mse"] for row in official["runs"]}
    groups = list(PLANS["expression"])
    rows = []
    # Warm up the optimizer and CPU kernels before any timed block.
    warmup = MLP(2, [16], "relu")
    train(warmup, data["train_x"], data["train_y"], steps=25, learning_rate=0.01, optimizer="adam")
    for repeat in range(len(groups)):
        for group in groups[repeat:] + groups[:repeat]:
            widths, activation, steps, lr = PLANS["expression"][group]
            model = MLP(2, widths, activation)
            model.load_state_dict(states[group], strict=True)
            started = time.perf_counter()
            train(model, data["train_x"], data["train_y"], steps=steps, learning_rate=lr, optimizer="adam")
            elapsed = time.perf_counter() - started
            with torch.no_grad():
                mse = (model(data["grid_x"]) - data["grid_y"]).square().mean().item()
            if mse != reference[group]:
                raise ValueError(f"{group}: grid MSE {mse} differs from official CPU result {reference[group]}")
            rows.append(dict(group=group, repeat=repeat + 1, seconds=elapsed, grid_mse=mse))
    medians = {g: statistics.median(r["seconds"] for r in rows if r["group"] == g) for g in groups}
    return dict(torch=str(torch.__version__), machine=platform.machine(), cpu_threads=torch.get_num_threads(),
                official_run=stamp, measured_region="train() only; CPU wall time, warm optimizer kernels",
                rows=rows, median_seconds=medians,
                relative_to_A={g: t/medians["A"] for g,t in medians.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stamp")
    result = collect(parser.parse_args().stamp)
    (root / "logs/q1_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["median_seconds"], flush=True)
