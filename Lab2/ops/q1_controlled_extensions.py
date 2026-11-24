"""Sensitivity of Q1 to initialization, activation and SGD learning rate."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/q1_neural_network"))
from models import MLP
from train import train
from utils.data import load_data, load_initial_states


def run(model, data, steps, lr, optimizer, diagnostics=False):
    history = train(model, data["train_x"], data["train_y"], steps=steps,
                    learning_rate=lr, optimizer=optimizer, diagnostics=diagnostics)
    with torch.no_grad():
        diff = model(data["grid_x"]) - data["grid_y"]
    row = dict(grid_mse=diff.square().mean().item(), first_train_mse=history[0]["train_mse"],
               final_train_mse=history[-1]["train_mse"])
    if diagnostics:
        row["first_layer_relative_gradient"] = [
            h["gradient_profile"]["layers"][0]["relative_to_output"] for h in (history[0], history[-1])]
    return row


def collect():
    torch.set_num_threads(1)
    expression = load_data("expression")
    states = load_initial_states("expression")
    architecture = []
    for seed in [None, 17, 42, 73, 101, 137]:
        for group, widths in [("A", [16]), ("D", [8, 4])]:
            if seed is not None:
                torch.manual_seed(seed)
            model = MLP(2, widths, "relu")
            if seed is None:
                model.load_state_dict(states[group])
            row = run(model, expression, 3000, 0.01, "adam")
            architecture.append(dict(group=group, seed=seed, **row))
    gradient = load_data("gradient")
    start = load_initial_states("gradient")["B"]
    factorial = []
    for activation in ["sigmoid", "relu"]:
        for lr in [0.025, 0.05, 0.1]:
            model = MLP(2, [32] * 8, activation)
            model.load_state_dict(start)
            row = run(model, gradient, 1000, lr, "sgd", diagnostics=True)
            factorial.append(dict(activation=activation, learning_rate=lr, **row))
    output = dict(torch=str(torch.__version__), device="cpu", threads=1,
                  initialization="null = course weights, integer = PyTorch Linear default reset with given seed",
                  architecture=architecture, activation_learning_rate=factorial)
    (ROOT / "logs/q1_controlled_extensions.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    collect()
