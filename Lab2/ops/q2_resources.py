"""Measure the default course workflow in a fresh process with PyTorch's CUDA allocator."""
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
q2 = root / "code/q2_clustering"
sys.path.insert(0, str(q2))
from main import run

if __name__ == "__main__":
    baseline = q2 / "outputs/baseline_20261123T155816Z/export_manifest.json"
    expected = json.loads(baseline.read_text())["clusters_sha256"]
    with tempfile.TemporaryDirectory(dir=q2 / "outputs", prefix="profile_") as temp:
        out = Path(temp) / "out"
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        status = run(output=out, device="cuda")
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        exported = json.loads((out / "export_manifest.json").read_text())
        assert status == 0 and exported["clusters_sha256"] == expected
        stats = dict(config="course default, 32 fits, three starts; independent repeat",
                     elapsed_seconds=duration, device=torch.cuda.get_device_name(0),
                     max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                     max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                     linux_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     selected=exported["run_key"], same_labels_sha256=expected)
    (root / "logs/q2_resources.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
