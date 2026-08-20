#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
image_name="mononav-demo:1.0"

if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
  echo "Docker image ${image_name} does not exist. Build it once with:"
  echo "  ${repo_dir}/docker/run_offline_demo.sh"
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set. Run this script from the graphical desktop session."
  exit 1
fi

xauthority="${XAUTHORITY:-${HOME}/.Xauthority}"
if [[ ! -f "${xauthority}" ]]; then
  echo "Xauthority file not found: ${xauthority}"
  echo "Set XAUTHORITY to the desktop session's Xauthority file and retry."
  exit 1
fi

docker run --rm --gpus all --ipc=host \
  --env DISPLAY="${DISPLAY}" \
  --env XAUTHORITY=/tmp/mononav.xauth \
  --env MONONAV_AUTO_CAPTURE=false \
  --env MONONAV_LIVE_VISUALIZATION=true \
  --env MONONAV_LIVE_HOLD="${MONONAV_LIVE_HOLD:-true}" \
  --env MONONAV_LIVE_DEPTH_DELAY_MS="${MONONAV_LIVE_DEPTH_DELAY_MS:-1}" \
  --env MONONAV_LIVE_FUSION_DELAY_MS="${MONONAV_LIVE_FUSION_DELAY_MS:-75}" \
  --env MONONAV_LIVE_PLANNER_DELAY_MS="${MONONAV_LIVE_PLANNER_DELAY_MS:-350}" \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "${xauthority}:/tmp/mononav.xauth:ro" \
  --volume "${repo_dir}:/workspace/MonoNav" \
  --volume mononav-torch-cache:/root/.cache/torch \
  --workdir /workspace/MonoNav \
  "${image_name}" \
  bash docker/run_live_demo_in_container.sh
