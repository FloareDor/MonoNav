#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
image_name="mononav-demo:1.0"

docker build \
  --file "${script_dir}/Dockerfile.demo" \
  --tag "${image_name}" \
  "${repo_dir}"

docker run --rm --gpus all --ipc=host \
  --volume "${repo_dir}:/workspace/MonoNav" \
  --volume mononav-torch-cache:/root/.cache/torch \
  --workdir /workspace/MonoNav \
  --env MONONAV_AUTO_CAPTURE=true \
  "${image_name}" \
  bash docker/run_demo_in_container.sh

echo "MonoNav offline demo complete."
echo "Artifacts:"
echo "  ${repo_dir}/data/demo_hallway/fusion_view.png"
echo "  ${repo_dir}/data/demo_hallway/planner_view.png"
echo "  ${repo_dir}/data/demo_hallway/mononav_rgb_depth.mp4"
echo "  ${repo_dir}/data/demo_hallway/pointcloud.ply"
echo "  ${repo_dir}/data/demo_hallway/vbg.npz"
