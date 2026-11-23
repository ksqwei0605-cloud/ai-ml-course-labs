"""Independent numerical evidence for the six Q1 analysis questions."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
q1 = root / "code/q1_neural_network"
sys.path.insert(0, str(q1))
from models import MLP
from utils.data import load_data, load_initial_states

def check_construction():
    model = MLP(2, [8, 4], "relu").double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        first, second = model.hidden
        for j, c in enumerate((-0.65, 0.65)):
            first.weight[2*j, 0], first.bias[2*j] = 1, -c
            first.weight[2*j+1, 0], first.bias[2*j+1] = -1, c
            second.weight[j, [2*j,2*j+1,4,5]] = -1/0.6
            second.bias[j] = 1
        first.weight[4,1], first.weight[5,1] = 1, -1
        model.output.weight[0,:2] = 1
        data = load_data("expression")
        result = {}
        for split in ("train","grid"):
            x = data[split+"_x"].double()
            y = data[split+"_y"].double()
            reference = sum(torch.relu(1-((x[:,0:1]-c).abs()+x[:,1:2].abs())/0.6) for c in (-0.65,0.65))
            prediction = model(x)
            result[split] = dict(max_error_vs_exact=float((prediction-reference).abs().max()),
                                 max_error_vs_supplied_float32=float((prediction-y).abs().max()))
            assert result[split]["max_error_vs_exact"] < 1e-13
    return result

def learning_rate():
    data, states = load_data("gradient"), load_initial_states("gradient")
    gradients, deltas, losses = {}, {}, {}
    for group, lr in (("B",0.05),("D",0.1)):
        model = MLP(2,[32]*8,"sigmoid")
        model.load_state_dict(states[group], strict=True)
        old = torch.cat([p.detach().flatten() for p in model.parameters()])
        loss = (model(data["train_x"])-data["train_y"]).square().mean()
        loss.backward()
        gradients[group] = torch.cat([p.grad.flatten() for p in model.parameters()])
        torch.optim.SGD(model.parameters(),lr=lr).step()
        new = torch.cat([p.detach().flatten() for p in model.parameters()])
        deltas[group] = new-old
        losses[group] = float(loss.detach())
    gap = float((gradients["B"]-gradients["D"]).abs().max())
    ratio = float(deltas["D"].norm()/deltas["B"].norm())
    assert gap==0 and abs(ratio-2)<1e-4
    return dict(max_initial_gradient_difference=gap, initial_losses=losses, first_update_norm_ratio=ratio)

def region_errors(folder):
    with np.load(folder/"predictions.npz") as p:
        y=p["reference"].reshape(-1)
        masks={"zero_background": y==0, "positive_support": y>0, "upper_peak":y>=0.75}
        result={}
        for region,mask in masks.items():
            result[region]=dict(points=int(mask.sum()), **{g:float(np.mean((p[g].reshape(-1)[mask]-y[mask])**2)) for g in "ABCD"})
        return result

if __name__=="__main__":
    torch.set_num_threads(1)
    stamp=sys.argv[1]
    out=dict(analytic_two_layer=check_construction(), same_initial_state=learning_rate(),
             grid_region_mse=region_errors(q1/"outputs"/stamp/"expression"))
    dest=root/"logs/q1_independent_checks.json"
    dest.write_text(json.dumps(out,indent=2))
    print(dest.read_text())
