#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
py="$(cd .. && pwd)/env/bin/python"
mkdir -p logs
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee -a "logs/run_${stamp}.log") 2>&1
trap 'echo "$(date -Is) FAILED line=$LINENO command=$BASH_COMMAND"' ERR
echo "$(date -Is) baseline preparation"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
while pgrep -f '^env/bin/python -m pip install torch==' >/dev/null; do sleep 10; done
"$py" -m pip install -r code/requirements-gpu.txt
"$py" -c 'import torch; assert torch.__version__ == "2.11.0+cu128" and torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"$py" -m pip freeze > "logs/environment_${stamp}.txt"
(cd code/q2_clustering/data && sha256sum -c SHA256SUMS.txt)
cd code/q1_neural_network
"$py" -m pytest -q
for task in expression gradient; do
    echo "$(date -Is) Q1 $task"
    "$py" main.py --task "$task" --device cpu --output "outputs/$stamp/$task"
done
cd "$root/code/q2_clustering"
"$py" -m pytest -q
for device in cpu cuda; do
    "$py" extras.py self-check --device "$device"
done
echo "$(date -Is) Q2 formal fit and evaluation"
output="outputs/baseline_$stamp"
"$py" main.py --device cuda --output "$output"
"$py" extras.py check-output --output "$output"
echo "$(date -Is) baseline complete output=$output"
