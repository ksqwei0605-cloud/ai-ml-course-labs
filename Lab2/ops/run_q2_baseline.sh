#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
py="$(cd "$root/.." && pwd)/env/bin/python"
stamp="${1:?supply the experiment timestamp}"
cd "$root/code/q2_clustering"
# This one framework test specifically requires the unimplemented starter.
case_unfinished=tests/test_workflow.py::WorkflowTests::test_unfinished_submission_saves_status_without_final_output
starter="$(mktemp -d)"
tar -xzf "$root/../inputs.tgz" -C "$starter" --exclude='*/data/features.npz' ./code/q2_clustering
(cd "$starter/code/q2_clustering" && "$py" -m pytest -q "$case_unfinished")
"$py" -m pytest -q --deselect="$case_unfinished"
for device in cpu cuda; do
    "$py" extras.py self-check --device "$device"
done
output="outputs/baseline_$stamp"
"$py" main.py --device cuda --output "$output"
"$py" extras.py check-output --output "$output"
echo "$(date -Is) baseline complete output=$output"

