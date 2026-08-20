# MonoNav: MAV Navigation via Monocular<br>Depth Estimation and Reconstruction

Original repo: [natesimon/MonoNav](https://github.com/natesimon/MonoNav)

## AirStack online demo

This fork runs the MonoNav perception and motion-primitive planner online with an
AirStack Iris in NVIDIA Isaac Sim. AirStack publishes the simulated left RGB image,
camera intrinsics, and ground-truth optical-camera pose through a small ROS 2/HTTP
bridge. This repository runs ZoeDepth, Open3D TSDF fusion, and MonoNav planning in a
separate GPU container, then returns the selected primitive to AirStack as a trajectory.

The separate containers are intentional. AirStack uses ROS 2 Jazzy, while the original
MonoNav stack requires an older Python/CUDA environment. The supplied MonoNav image is
based on PyTorch 1.13.1, CUDA 11.6, Open3D 0.18.0, OpenCV 4.6.0.66, NumPy 1.24.1,
SciPy 1.10.0, and timm 0.6.12. ZoeDepth is pinned as a Git submodule at `edb6daf`.

The integration preserves the MonoNav core: monocular ZoeDepth inference, Open3D TSDF
fusion, and goal-directed collision checking over motion primitives. Integration-only
additions are the AirStack bridge, simulator-depth A/B mode, start/pause gate, a local
sliding TSDF window, conservative synthetic-depth speckle filtering, trajectory command
holds, and yaw-scan recovery. It is therefore best described as **MonoNav integrated
with AirStack**, not a bit-for-bit reproduction of the original flight stack.

### Requirements

- Ubuntu with an NVIDIA GPU and a driver compatible with CUDA 11.6 and Isaac Sim
- Docker Engine, Docker Compose, and NVIDIA Container Toolkit
- A graphical X11 desktop (`DISPLAY` and `~/.Xauthority` available)
- Access to `github.com/castacks` and the CMU AirLab Docker registry
- At least 25 GB of free disk space for the AirStack images

### Installation

Clone AirStack and this fork as sibling directories. Recursive clone is required for
both PegasusSimulator and ZoeDepth.

```bash
mkdir dsta_ws
cd dsta_ws
git clone --recursive -b eungchang/adv-ws2 git@github.com:castacks/AirStack.git
git clone https://github.com/engcang/MonoNav.git
```

Set up AirStack and obtain its images. Skip `install` when Docker and NVIDIA Container
Toolkit are already configured.

```bash
cd AirStack
./airstack.sh install       # optional on an already configured host
./airstack.sh setup
docker login airlab-docker.andrew.cmu.edu
./airstack.sh image-pull
```

Build the MonoNav GPU image from the checked-in Dockerfile. No pre-existing local image
is assumed.

```bash
cd ../MonoNav
./docker/build_image.sh
```

### Run the online demo

Open three terminals in the same graphical desktop session.

Terminal 1 — start only Isaac Sim and the desktop robot stack. This avoids starting the
GCS/Foxglove service. The deterministic demo scene is an empty Isaac room with three
colored slalom obstacles, and the RTX lidar is disabled because MonoNav does not use it.

```bash
cd dsta_ws/AirStack
PLAY_SIM_ON_START=true \
ISAAC_SIM_ENVIRONMENT='Default Environment' \
MONONAV_DEMO_OBSTACLES=true \
DRONE_INIT_X=0.0 \
DRONE_INIT_Y=0.0 \
DRONE_INIT_Z=0.30 \
ENABLE_LIDAR=false \
./airstack.sh up isaac-sim robot-desktop
```

Wait for the ROS workspace build and AirStack bringup to finish. In the AirStack GUI,
issue `Takeoff` and wait until the Iris is stably hovering.

Terminal 2 — launch the bridge with trajectory execution enabled:

```bash
docker exec -it airstack-robot-desktop-1 bash -lc \
  'sws && ros2 launch mononav_bridge mononav_bridge.launch.xml \
  mononav_bridge_execute_commands:=true'
```

Terminal 3 — start the MonoNav worker:

```bash
cd dsta_ws/MonoNav
MONONAV_EXECUTE=true ./docker/run_airstack_live.sh \
  --depth-source zoe \
  --zoe-depth-scale 1.68 \
  --wait-for-start \
  --rate 1.0 \
  --warmup-frames 6 \
  --min-tsdf-points 1000 \
  --velocity 0.4 \
  --goal-distance 8.0 \
  --min-dist2obs 0.6 \
  --planning-height-band 0.30 \
  --clearance-recovery-distance 0.25 \
  --max-altitude-deviation 0.6 \
  --stop-confirm-frames 3 \
  --command-hold-seconds 1.25 \
  --recovery-hold-seconds 2.5 \
  --tsdf-block-warning 80000
```

Arrange the Isaac Sim, RViz, RGB, depth, and Open3D windows before flight. Wait until the
RGB overlay shows `map_ready=True` (at least 1000 TSDF points), focus either OpenCV RGB
or depth window, and press `S` or `Space` to set an 8 m goal in the camera's current
forward direction and enable commands. Press `P` to pause and hover; press `Q` or `Esc`
to stop the worker. The Open3D window supports left-drag rotation, Ctrl+left-drag pan,
mouse-wheel zoom, and `R` to reset to its Z-up view.

To compare the same integration against metric simulator depth, replace the first two
worker options with:

```bash
--depth-source ground-truth --min-tsdf-points 0
```

The Zoe scale `1.68` is an empirical calibration for this simulated ZED image path. It
is not caused solely by image downsampling and is not universal: FOV/intrinsics,
content, and synthetic-to-real domain shift all affect monocular metric scale. Refit it
after changing the camera pipeline, scene domain, or depth model.

### Shutdown

Stop the MonoNav worker and bridge with `Ctrl+C`, then remove the AirStack containers:

```bash
cd dsta_ws/AirStack
./airstack.sh down
```

### Main demo parameters

- `--rate`: maximum planning/fusion loop rate in Hz; ZoeDepth inference may be slower.
- `--warmup-frames`: frames fused before the planner is allowed to select commands.
- `--velocity`: commanded speed along a one-second primitive in m/s.
- `--goal-distance`: forward distance from the pose at which the start key is pressed.
- `--min-dist2obs`: required primitive clearance from reconstructed obstacles in metres.
- `--command-hold-seconds`: minimum time a selected primitive remains active before
  replanning can replace it.
- `--tsdf-local-radius`: radius of the spatial sliding map; old blocks are pruned
  gradually rather than clearing the complete TSDF.

The AirStack profile contains 13 gentler primitives at 0.4 m/s in
`utils/trajlib_airstack`. The original seven-primitive library in `utils/trajlib` is not
modified.
