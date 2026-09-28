"""Run MonoNav online from AirStack RGB and ground-truth optical-camera poses."""

import argparse
import copy
import os
from pathlib import Path
import json
import struct
import time
import urllib.error
import urllib.request
import zlib

import cv2
import numpy as np
import open3d as o3d
import torch

import sys

sys.path.insert(0, "ZoeDepth")
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from utils.utils import (
    VoxelBlockGrid,
    choose_primitive,
    compute_depth,
    get_traj_linesets,
    get_trajlist,
    load_config,
    transform_image,
    transform_depth_image,
)


def fetch_frame(server_url, include_depth=False):
    with urllib.request.urlopen(server_url.rstrip("/") + "/frame", timeout=3.0) as response:
        body = response.read()
    metadata_length = struct.unpack("!I", body[:4])[0]
    metadata = json.loads(body[4 : 4 + metadata_length])
    payload = body[4 + metadata_length :]
    jpeg_length = int(metadata.get("jpeg_bytes", len(payload)))
    jpeg = np.frombuffer(payload[:jpeg_length], dtype=np.uint8)
    bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("AirStack bridge returned an invalid JPEG")
    if not include_depth:
        return metadata, bgr
    depth_length = int(metadata.get("depth_bytes", 0))
    if depth_length <= 0:
        return metadata, bgr, None
    depth_payload = payload[jpeg_length : jpeg_length + depth_length]
    depth = np.frombuffer(zlib.decompress(depth_payload), dtype="<f4").reshape(
        int(metadata["depth_height"]), int(metadata["depth_width"])
    )
    return metadata, bgr, depth


def post_trajectory(server_url, primitive_index, points, velocity, execute):
    payload = json.dumps(
        {
            "primitive_index": int(primitive_index),
            "velocity": float(velocity),
            "execute": bool(execute),
            "replace": True,
            "waypoints": points.tolist(),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/trajectory",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())


def make_yaw_scan(camera_pose, direction, yaw_degrees, radius, point_count=17):
    """Generate a tiny-radius arc that changes heading with negligible translation."""
    position = camera_pose[:3, 3]
    forward = camera_pose[:3, 2]
    heading = float(np.arctan2(forward[1], forward[0]))
    yaw_delta = float(direction) * np.deg2rad(yaw_degrees)
    theta = np.linspace(0.0, yaw_delta, point_count)
    magnitude = np.abs(theta)
    forward_offset = radius * np.sin(magnitude)
    left_offset = np.sign(yaw_delta) * radius * (1.0 - np.cos(magnitude))
    forward_xy = np.asarray([np.cos(heading), np.sin(heading)])
    left_xy = np.asarray([-np.sin(heading), np.cos(heading)])
    xy = (
        position[:2]
        + forward_offset[:, None] * forward_xy
        + left_offset[:, None] * left_xy
    )
    xyz = np.column_stack((xy, np.full(point_count, position[2])))
    yaw = heading + theta
    return np.column_stack((xyz, yaw))


def post_pause(server_url):
    request = urllib.request.Request(
        server_url.rstrip("/") + "/pause",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"success": False, "reason": str(exc)}


def camera_matrix(metadata):
    return np.asarray(metadata["k"], dtype=np.float64).reshape(3, 3)


def metric_depth_colormap(depth_m, depth_max):
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0 * (1.0 - depth_m[valid] / float(depth_max)), 0.0, 255.0
    ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def filter_isolated_near_depth(
    depth_mm,
    kernel_size=3,
    absolute_threshold_mm=250.0,
    relative_threshold=0.12,
):
    """Replace isolated, spuriously-near Zoe pixels with the local median.

    MonoNav's offline hallway data contains spatially smooth depth estimates.
    Synthetic RGB can instead produce small near-depth islands in otherwise
    farther regions; TSDF fusion turns those islands into persistent floating
    surfaces.  A near-only median gate removes those islands while preserving
    ordinary depth variation and the interiors of real foreground objects.

    This is deliberately adapter-local: the original offline MonoNav pipeline
    and its saved depth maps are not changed.
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    if kernel_size <= 1:
        return depth_mm, 0
    if kernel_size % 2 == 0 or kernel_size > 5:
        raise ValueError("Zoe speckle kernel must be odd and no larger than 5")
    local_median = cv2.medianBlur(depth, int(kernel_size))
    threshold = np.maximum(
        float(absolute_threshold_mm),
        float(relative_threshold) * local_median,
    )
    isolated_near = (
        (depth > 0.0)
        & (local_median > 0.0)
        & ((local_median - depth) > threshold)
    )
    # A true foreground boundary can be in the minority of a 3x3 window too.
    # Preserve it when at least three pixels locally support the same near
    # surface; a salt-and-pepper island has only itself (or one neighbour).
    radius = int(kernel_size) // 2
    padded = cv2.copyMakeBorder(
        depth, radius, radius, radius, radius, cv2.BORDER_REPLICATE
    )
    support = np.zeros(depth.shape, dtype=np.uint8)
    agreement = np.maximum(75.0, 0.05 * depth)
    for row_offset in range(int(kernel_size)):
        for col_offset in range(int(kernel_size)):
            neighbour = padded[
                row_offset : row_offset + depth.shape[0],
                col_offset : col_offset + depth.shape[1],
            ]
            support += (np.abs(neighbour - depth) <= agreement).astype(np.uint8)
    isolated_near &= support <= 2
    filtered = depth.copy()
    filtered[isolated_near] = local_median[isolated_near]
    return np.clip(filtered, 0.0, 65535.0).astype(np.uint16), int(
        np.count_nonzero(isolated_near)
    )


def reproject_depth_to_camera(depth_mm, source_pose, target_pose, intrinsic):
    """Z-buffer a prior metric depth image into the current optical camera."""
    depth = np.asarray(depth_mm, dtype=np.float32) / 1000.0
    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.float32)

    z = depth[valid]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    points = np.vstack(
        (
            (cols[valid] - cx) * z / fx,
            (rows[valid] - cy) * z / fy,
            z,
        )
    )
    source_to_target = np.linalg.inv(target_pose) @ source_pose
    target_points = (
        source_to_target[:3, :3] @ points + source_to_target[:3, 3:4]
    )
    target_z = target_points[2]
    in_front = target_z > 1.0e-4
    projected_col = np.rint(
        fx * target_points[0, in_front] / target_z[in_front] + cx
    ).astype(np.int32)
    projected_row = np.rint(
        fy * target_points[1, in_front] / target_z[in_front] + cy
    ).astype(np.int32)
    in_image = (
        (projected_col >= 0)
        & (projected_col < width)
        & (projected_row >= 0)
        & (projected_row < height)
    )
    flat_index = (
        projected_row[in_image] * width + projected_col[in_image]
    )
    reference = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(reference, flat_index, target_z[in_front][in_image])
    reference[~np.isfinite(reference)] = 0.0
    return reference.reshape(height, width) * 1000.0


def filter_temporal_near_depth(
    depth_mm,
    reference_mm,
    absolute_threshold_mm=300.0,
    relative_threshold=0.12,
    max_component_pixels=200,
):
    """Reject small surfaces that appear implausibly nearer than the prior view."""
    depth = np.asarray(depth_mm, dtype=np.float32)
    reference = np.asarray(reference_mm, dtype=np.float32)
    threshold = np.maximum(
        float(absolute_threshold_mm),
        float(relative_threshold) * reference,
    )
    sudden_near = (
        (depth > 0.0)
        & (reference > 0.0)
        & ((reference - depth) > threshold)
    )
    if not np.any(sudden_near) or max_component_pixels <= 0:
        return depth_mm, 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        sudden_near.astype(np.uint8), connectivity=8
    )
    areas = stats[:, cv2.CC_STAT_AREA]
    reject_label = np.zeros(count, dtype=bool)
    reject_label[1:] = areas[1:] <= int(max_component_pixels)
    rejected = reject_label[labels]
    filtered = depth.copy()
    filtered[rejected] = reference[rejected]
    return np.clip(filtered, 0.0, 65535.0).astype(np.uint16), int(
        np.count_nonzero(rejected)
    )


def transform_primitive(lineset, camera_pose):
    transformed = copy.deepcopy(lineset)
    transformed.transform(camera_pose)
    points = np.asarray(transformed.points).copy()
    if len(points) < 2:
        return points
    delta = np.diff(points[:, :2], axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    yaw = np.concatenate((yaw, yaw[-1:]))
    return np.column_stack((points, yaw))


def transform_execution_primitive(trajectory, camera_pose, point_count=17):
    """Transform only the one-period motion, excluding its look-ahead extension.

    MonoNav appends a straight one-metre extension to ``x_sample/y_sample`` so
    collision checking sees beyond the one-second control primitive.  That
    extension is not a command and must never be sent to AirStack.
    """
    x_forward = np.asarray(trajectory["xvals"], dtype=np.float64)
    y_left = np.asarray(trajectory["yvals"], dtype=np.float64)
    sample_indices = np.linspace(
        0, len(x_forward) - 1, min(point_count - 1, len(x_forward)), dtype=int
    )
    # Optical camera coordinates are +Z forward, +X right, +Y down.
    local_points = np.column_stack(
        (-y_left[sample_indices], np.zeros(len(sample_indices)), x_forward[sample_indices])
    )
    local_points = np.vstack((np.zeros(3), local_points))
    homogeneous = np.column_stack((local_points, np.ones(len(local_points))))
    world_points = (camera_pose @ homogeneous.T).T[:, :3]
    delta = np.diff(world_points[:, :2], axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    yaw = np.concatenate((yaw, yaw[-1:]))
    return np.column_stack((world_points, yaw))


def set_default_view(visualizer, lookat):
    """Use a stable Z-up isometric view instead of Open3D's arbitrary auto-fit view."""
    control = visualizer.get_view_control()
    control.set_lookat(np.asarray(lookat, dtype=np.float64))
    control.set_front(np.asarray([0.634, -0.634, -0.444]))
    control.set_up(np.asarray([0.314, -0.314, 0.896]))
    control.set_zoom(0.45)


def create_visualizer():
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    ok = visualizer.create_window(
        window_name="MonoNav - live TSDF and motion primitives",
        width=880,
        height=650,
        left=20,
        top=40,
    )
    if not ok:
        raise RuntimeError("Open3D could not create a GUI window")
    options = visualizer.get_render_option()
    options.background_color = np.asarray([0.06, 0.07, 0.09])
    options.point_size = 3.0
    state = {
        "cloud": o3d.geometry.PointCloud(),
        "lines": [],
        "cloud_added": False,
        "lines_added": False,
        "view_initialized": False,
        "lookat": np.zeros(3),
    }

    def reset_view_callback(vis):
        set_default_view(vis, state["lookat"])
        return False

    visualizer.register_key_callback(ord("R"), reset_view_callback)
    return visualizer, state


def update_open3d(visualizer, state, vbg, weight_threshold, trajectory_lines):
    point_cloud = vbg.extract_point_cloud(weight_threshold).cpu().to_legacy()
    point_count = len(point_cloud.points)
    if os.environ.get('WS2_INFERENCE_DIR') and state is not None:
        points=np.asarray(point_cloud.points)
        state['preview_points']=points[::max(1,len(points)//6000)].copy()
    if visualizer is None:
        return point_count

    # Keep the same geometry objects registered with the renderer. Clearing and
    # re-adding them every fusion frame makes mouse navigation stutter and can
    # disturb the user's chosen camera view.
    state["cloud"].points = point_cloud.points
    state["cloud"].colors = point_cloud.colors
    state["cloud"].normals = point_cloud.normals
    if not state["lines_added"]:
        state["lines"] = [copy.deepcopy(line) for line in trajectory_lines]
        for line in state["lines"]:
            visualizer.add_geometry(line, reset_bounding_box=False)
        state["lines_added"] = True
    else:
        for displayed, updated in zip(state["lines"], trajectory_lines):
            displayed.points = updated.points
            displayed.lines = updated.lines
            displayed.colors = updated.colors
            visualizer.update_geometry(displayed)

    # The first few frames can be empty. Initialize the view only once a useful
    # reconstruction exists. Open3D refuses to register an empty PointCloud, so
    # adding it on frame zero and merely calling update_geometry later results in
    # a permanently black window even though the TSDF itself is populated.
    if point_count >= 200 and not state["view_initialized"]:
        visualizer.add_geometry(state["cloud"], reset_bounding_box=True)
        state["cloud_added"] = True
        state["lookat"] = point_cloud.get_axis_aligned_bounding_box().get_center()
        set_default_view(visualizer, state["lookat"])
        state["view_initialized"] = True
    elif state["cloud_added"]:
        visualizer.update_geometry(state["cloud"])
    visualizer.poll_events()
    visualizer.update_renderer()
    return point_count


def planning_voxel_count(vbg, weight_threshold):
    """Count the weighted negative-TSDF voxels used by choose_primitive."""
    weights = vbg.attribute("weight").reshape((-1))
    tsdf = vbg.attribute("tsdf").reshape((-1))
    _, voxel_indices = vbg.voxel_coordinates_and_flattened_indices()
    mask = (weights[voxel_indices] > weight_threshold) & (tsdf[voxel_indices] < 0.0)
    return int(np.count_nonzero(mask.cpu().numpy()))


def pump_open3d_events(visualizer, duration):
    """Keep mouse navigation responsive while waiting for the next fusion tick."""
    if visualizer is None:
        time.sleep(max(0.0,duration))
        return
    deadline = time.monotonic() + max(0.0, duration)
    while time.monotonic() < deadline:
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless',action='store_true',help='Run without OpenCV/Open3D windows')
    parser.add_argument("--server", default="http://airstack-robot-desktop-1:8765")
    parser.add_argument(
        "--depth-source",
        choices=("zoe", "ground-truth"),
        default="zoe",
        help="use ZoeDepth inference or simulator metric depth for an A/B baseline",
    )
    parser.add_argument(
        "--zoe-depth-scale",
        type=float,
        default=1.0,
        help="fixed metric scale calibration applied to ZoeDepth output",
    )
    parser.add_argument(
        "--zoe-speckle-kernel",
        type=int,
        default=3,
        help="near-depth outlier median kernel; 1 disables the adapter-local filter",
    )
    parser.add_argument(
        "--zoe-speckle-absolute-mm",
        type=float,
        default=250.0,
        help="minimum near-depth disagreement before a Zoe pixel is replaced",
    )
    parser.add_argument(
        "--zoe-speckle-relative",
        type=float,
        default=0.12,
        help="relative local-depth disagreement before a Zoe pixel is replaced",
    )
    parser.add_argument(
        "--zoe-temporal-near-mm",
        type=float,
        default=300.0,
        help="absolute current-vs-warped-prior near-depth disagreement gate",
    )
    parser.add_argument(
        "--zoe-temporal-near-relative",
        type=float,
        default=0.12,
        help="relative current-vs-warped-prior near-depth disagreement gate",
    )
    parser.add_argument(
        "--zoe-temporal-max-component",
        type=int,
        default=200,
        help="maximum sudden-near component area to reject; 0 disables the gate",
    )
    parser.add_argument("--planner-debug", action="store_true")
    parser.add_argument(
        "--trajlib-dir",
        default="utils/trajlib_airstack",
        help="AirStack-specific motion primitive library",
    )
    parser.add_argument("--rate", type=float, default=2.0, help="maximum fusion rate in Hz")
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="request actual AirStack execution")
    parser.add_argument(
        "--wait-for-start",
        action="store_true",
        help="fuse and visualize immediately, but wait for S/Space in an OpenCV window before flight",
    )
    parser.add_argument("--goal-distance", type=float, default=8.0)
    parser.add_argument('--goal-radius',type=float,default=None,help='Override configured goal completion radius (metres)')
    parser.add_argument("--velocity", type=float, default=0.4)
    parser.add_argument(
        "--command-hold-seconds",
        type=float,
        default=1.25,
        help=(
            "minimum time to let an accepted motion primitive advance before "
            "replacing it; an unsafe plan still pauses immediately"
        ),
    )
    parser.add_argument(
        "--min-dist2obs",
        type=float,
        default=0.8,
        help="minimum 3D clearance between a candidate primitive and TSDF obstacles (m)",
    )
    parser.add_argument(
        "--max-altitude-deviation",
        type=float,
        default=0.35,
        help="pause if camera altitude differs from its initial value by more than this (m)",
    )
    parser.add_argument(
        "--stop-confirm-frames",
        type=int,
        default=3,
        help="end the mission only after this many consecutive frames have no safe primitive",
    )
    parser.add_argument(
        "--planning-height-band",
        type=float,
        default=0.4,
        help="collision-check TSDF voxels within this distance above/below the camera (m)",
    )
    parser.add_argument(
        "--clearance-recovery-distance",
        type=float,
        default=0.25,
        help="initial path length allowed to exit an already-inflated obstacle margin",
    )
    parser.add_argument("--recovery-yaw-deg", type=float, default=25.0)
    parser.add_argument("--recovery-radius", type=float, default=0.03)
    parser.add_argument("--recovery-velocity", type=float, default=0.03)
    parser.add_argument(
        "--recovery-hold-seconds",
        type=float,
        default=2.0,
        help="minimum time to finish one yaw scan before another recovery command",
    )
    parser.add_argument("--max-recovery-steps", type=int, default=12)
    parser.add_argument(
        "--min-tsdf-points",
        type=int,
        default=1000,
        help="do not execute motion until this many weighted TSDF surface points exist",
    )
    parser.add_argument(
        "--tsdf-local-radius",
        type=float,
        default=3.0,
        help="retain only TSDF voxel blocks within this 3D radius of the camera (m)",
    )
    parser.add_argument(
        "--tsdf-device",
        choices=("CPU:0", "CUDA:0"),
        help="override the Open3D voxel-grid device without changing ZoeDepth's device",
    )
    parser.add_argument(
        "--tsdf-weight-threshold",
        type=float,
        help="override the minimum integrated TSDF weight used for planning",
    )
    parser.add_argument(
        "--tsdf-block-warning",
        "--max-tsdf-blocks",
        dest="tsdf_block_warning",
        type=int,
        default=80000,
        help=(
            "warn when the local TSDF exceeds this many active blocks; the map "
            "is never globally reset (0 disables the warning; "
            "--max-tsdf-blocks is a backward-compatible alias)"
        ),
    )
    args = parser.parse_args()
    if args.headless and args.wait_for_start:
        parser.error('--wait-for-start requires a GUI')

    config = load_config("config.yml")
    if args.goal_radius is not None:
        if not np.isfinite(args.goal_radius) or args.goal_radius<=0:parser.error('--goal-radius must be positive')
        config['min_dist2goal']=args.goal_radius
    if args.tsdf_weight_threshold is not None and (
        not np.isfinite(args.tsdf_weight_threshold) or args.tsdf_weight_threshold < 0
    ):
        parser.error('--tsdf-weight-threshold must be non-negative')
    zoe = None
    if args.depth_source == "zoe":
        print("Loading ZoeDepth...")
        zoe_config = get_config("zoedepth", config["zoedepth_mode"])
        zoe = build_model(zoe_config)
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        zoe = zoe.to(device_name)
        zoe.eval()
        print(f"ZoeDepth ready on {device_name}")
    else:
        print("Using Isaac Sim ground-truth metric depth (ZoeDepth bypassed)")

    settings = config["VoxelBlockGrid"]
    tsdf_device = args.tsdf_device or settings["device"]
    tsdf_weight_threshold = (
        args.tsdf_weight_threshold
        if args.tsdf_weight_threshold is not None
        else config["weight_threshold"]
    )
    def make_vbg_wrapper():
        return VoxelBlockGrid(
            settings["depth_scale"],
            settings["depth_max"],
            settings["trunc_voxel_multiplier"],
            o3d.core.Device(tsdf_device),
        )

    vbg_wrapper = make_vbg_wrapper()
    trajectory_list = get_trajlist(args.trajlib_dir)
    trajectory_lines, _, forward_speed, _ = get_traj_linesets(trajectory_list)
    visualizer, visualizer_state = (None,{}) if args.headless else create_visualizer()
    rgb_window = "MonoNav - live AirStack RGB"
    depth_window = "MonoNav - live ZoeDepth"
    for window_name, left, top in (
        (rgb_window, 930, 40),
        (depth_window, 930, 450),
    ):
        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 360)
            cv2.moveWindow(window_name, left, top)
    kinect = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )

    goal_position = None
    initial_camera_altitude = None
    last_sequence = -1
    fused_frames = 0
    selected_index = len(trajectory_lines) // 2
    min_period = 1.0 / max(args.rate, 0.1)
    mission_complete = False
    pause_sent = False
    unsafe_frame_count = 0
    recovery_steps = 0
    recovery_direction = 1
    recovery_mode = "none"
    point_count = 0
    tsdf_block_count = 0
    stop_reason = ""
    flight_started = not args.wait_for_start
    last_motion_command_time = 0.0
    last_recovery_command_time = 0.0
    active_primitive_index = None
    last_capacity_warning_time = 0.0
    previous_zoe_depth_mm = None
    previous_zoe_pose = None

    print(f"Connecting to AirStack bridge at {args.server}")
    print(
        "Open3D controls: left-drag rotate, Ctrl+left-drag pan, wheel zoom, "
        "R reset to the Z-up view; press Q in RGB/depth to stop"
    )
    print(f"Trajectory execution requested: {args.execute}")
    if args.execute and args.wait_for_start:
        print("START GATE CLOSED: arrange the GUI/recording, then focus RGB or Depth and press S or Space")

    try:
        while True:
            loop_start = time.monotonic()
            try:
                if args.depth_source == "ground-truth":
                    metadata, bgr, source_depth_m = fetch_frame(
                        args.server, include_depth=True
                    )
                    if source_depth_m is None:
                        raise RuntimeError("AirStack bridge has no ground-truth depth frame")
                else:
                    metadata, bgr = fetch_frame(args.server)
                    source_depth_m = None
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                print(f"Waiting for AirStack frame: {exc}")
                time.sleep(1.0)
                continue
            if metadata["sequence"] == last_sequence:
                if visualizer is not None:
                    visualizer.poll_events()
                    visualizer.update_renderer()
                time.sleep(0.02)
                continue
            last_sequence = metadata["sequence"]

            pose = np.asarray(metadata["t_world_camera"], dtype=np.float64).reshape(4, 4)
            if goal_position is None:
                camera_goal = np.array([0.0, 0.0, args.goal_distance, 1.0])
                goal_position = (pose @ camera_goal)[:3].reshape(1, 3)
                initial_camera_altitude = float(pose[2, 3])
                print(f"World-frame goal initialized at {goal_position[0].round(3).tolist()}")
            distance_to_goal = float(np.linalg.norm(pose[:3, 3] - goal_position[0]))
            if flight_started and distance_to_goal <= float(config["min_dist2goal"]):
                mission_complete = True
                stop_reason = "goal threshold reached"
            altitude_deviation = abs(float(pose[2, 3]) - initial_camera_altitude)
            if flight_started and altitude_deviation > args.max_altitude_deviation:
                print(
                    f"Safety stopping at altitude deviation {altitude_deviation:.2f} m "
                    f"(limit {args.max_altitude_deviation:.2f} m)"
                )
                mission_complete = True
                stop_reason = "altitude deviation"

            # Keep a spatial sliding window.  Pruning before integration frees
            # capacity gradually and, unlike rebuilding the VBG, never makes
            # the entire reconstruction disappear at once.
            tsdf_block_count, pruned_before = vbg_wrapper.prune_outside_radius(
                pose[:3, 3], args.tsdf_local_radius
            )
            if pruned_before > 0:
                print(
                    f"TSDF sliding window pre-prune: "
                    f"blocks={tsdf_block_count}, pruned={pruned_before}"
                )

            source_intrinsic = camera_matrix(metadata)
            zero_distortion = np.zeros(5, dtype=np.float64)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kinect_rgb = transform_image(rgb, source_intrinsic, zero_distortion, kinect)

            inference_start = time.monotonic()
            if args.depth_source == "zoe":
                with torch.inference_mode():
                    depth_mm, depth_colormap = compute_depth(kinect_rgb, zoe)
                depth_mm = np.clip(
                    depth_mm.astype(np.float32) * args.zoe_depth_scale,
                    0.0,
                    65535.0,
                ).astype(np.uint16)
                depth_mm, spatial_speckle_pixels = filter_isolated_near_depth(
                    depth_mm,
                    args.zoe_speckle_kernel,
                    args.zoe_speckle_absolute_mm,
                    args.zoe_speckle_relative,
                )
                temporal_speckle_pixels = 0
                if (
                    previous_zoe_depth_mm is not None
                    and args.zoe_temporal_max_component > 0
                ):
                    reference_mm = reproject_depth_to_camera(
                        previous_zoe_depth_mm,
                        previous_zoe_pose,
                        pose,
                        kinect.intrinsic_matrix,
                    )
                    depth_mm, temporal_speckle_pixels = filter_temporal_near_depth(
                        depth_mm,
                        reference_mm,
                        args.zoe_temporal_near_mm,
                        args.zoe_temporal_near_relative,
                        args.zoe_temporal_max_component,
                    )
                previous_zoe_depth_mm = depth_mm.copy()
                previous_zoe_pose = pose.copy()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET
                )
            else:
                spatial_speckle_pixels = 0
                temporal_speckle_pixels = 0
                source_depth_m = transform_depth_image(
                    source_depth_m, source_intrinsic, zero_distortion, kinect
                )
                source_depth_m = np.nan_to_num(
                    source_depth_m, nan=0.0, posinf=0.0, neginf=0.0
                )
                depth_mm = np.clip(source_depth_m * 1000.0, 0.0, 65535.0).astype(
                    np.uint16
                )
                depth_colormap = metric_depth_colormap(
                    source_depth_m, settings["depth_max"]
                )
            inference_seconds = time.monotonic() - inference_start
            vbg_wrapper.integration_step(
                cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR), depth_mm, pose
            )
            point_count = planning_voxel_count(
                vbg_wrapper.vbg, tsdf_weight_threshold
            )
            tsdf_block_count, pruned_blocks = vbg_wrapper.prune_outside_radius(
                pose[:3, 3], args.tsdf_local_radius
            )
            if pruned_blocks > 0:
                print(
                    f"TSDF sliding window post-prune: "
                    f"blocks={tsdf_block_count}, pruned={pruned_blocks}"
                )
            now = time.monotonic()
            if (
                args.tsdf_block_warning > 0
                and tsdf_block_count >= args.tsdf_block_warning
                and now - last_capacity_warning_time >= 5.0
            ):
                print(
                    f"TSDF CAPACITY WARNING: {tsdf_block_count} active blocks; "
                    "map retained, consider reducing --tsdf-local-radius"
                )
                last_capacity_warning_time = now
            fused_frames += 1

            transformed_lines = []
            for index, candidate in enumerate(trajectory_lines):
                line = copy.deepcopy(candidate)
                line.transform(pose)
                line.paint_uniform_color([0.65, 0.65, 0.65])
                transformed_lines.append(line)

            should_stop = False
            chosen = None
            if fused_frames >= args.warmup_frames:
                try:
                    should_stop, chosen = choose_primitive(
                        vbg_wrapper.vbg,
                        pose,
                        trajectory_lines,
                        goal_position,
                        args.min_dist2obs,
                        False,  # TSDF is in AirStack map coordinates; Y is not camera-down.
                        config["filterWeights"],
                        config["filterTSDF"],
                        tsdf_weight_threshold,
                        2,  # AirStack map frame is Z-up.
                        float(pose[2, 3]),
                        args.planning_height_band,
                        args.planner_debug,
                        True,  # AirStack-only recovery from an inflated-margin start.
                        args.clearance_recovery_distance,
                    )
                    if chosen is not None:
                        selected_index = chosen
                except (ValueError, RuntimeError) as exc:
                    print(f"Planner waiting for a usable reconstruction: {exc}")

            transformed_lines[selected_index].paint_uniform_color([0.0, 1.0, 0.0])
            world_primitive = transform_execution_primitive(
                trajectory_list[selected_index], pose
            )
            map_ready = point_count >= args.min_tsdf_points
            extreme_selected = chosen in (0, len(trajectory_lines) - 1)
            if chosen == 0:
                recovery_direction = 1
            elif chosen == len(trajectory_lines) - 1:
                recovery_direction = -1
            recovery_mode = "none"
            if flight_started and map_ready and should_stop:
                unsafe_frame_count += 1
                if args.execute and unsafe_frame_count == 1:
                    print(f"UNSAFE HOLD: {post_pause(args.server)}")
                if unsafe_frame_count >= args.stop_confirm_frames:
                    recovery_mode = "blocked yaw scan"
            elif flight_started and map_ready and extreme_selected:
                unsafe_frame_count = 0
                recovery_mode = "extreme-primitive yaw scan"
            else:
                unsafe_frame_count = 0
                if flight_started:
                    recovery_steps = 0

            if recovery_mode != "none" and recovery_steps >= args.max_recovery_steps:
                mission_complete = True
                stop_reason = (
                    f"no central safe primitive after {recovery_steps} yaw scans"
                )
            if mission_complete and args.execute and not pause_sent:
                print(
                    f"Mission stopping ({stop_reason}) at goal distance "
                    f"{distance_to_goal:.2f} m: {post_pause(args.server)}"
                )
                pause_sent = True
                flight_started = False
            execute_this_primitive = (
                args.execute
                and flight_started
                and fused_frames >= args.warmup_frames
                and map_ready
                and not should_stop
                and not extreme_selected
                and not mission_complete
            )
            if (
                args.execute
                and flight_started
                and map_ready
                and recovery_mode != "none"
                and not mission_complete
            ):
                now = time.monotonic()
                recovery_due = (
                    last_recovery_command_time == 0.0
                    or now - last_recovery_command_time >= args.recovery_hold_seconds
                )
                if recovery_due:
                    yaw_scan = make_yaw_scan(
                        pose,
                        recovery_direction,
                        args.recovery_yaw_deg,
                        args.recovery_radius,
                    )
                    response = post_trajectory(
                        args.server, -1, yaw_scan, args.recovery_velocity, True
                    )
                    last_recovery_command_time = now
                    last_motion_command_time = 0.0
                    active_primitive_index = None
                    recovery_steps += 1
                    print(
                        f"RECOVERY {recovery_steps}/{args.max_recovery_steps}: "
                        f"{recovery_mode}, "
                        f"yaw_step={recovery_direction * args.recovery_yaw_deg:.1f} deg"
                    )
                else:
                    held_for = now - last_recovery_command_time
                    response = {
                        "accepted": False,
                        "reason": (
                            f"holding recovery {recovery_steps} "
                            f"({held_for:.1f}/{args.recovery_hold_seconds:.1f}s)"
                        ),
                    }
            else:
                now = time.monotonic()
                command_due = (
                    last_motion_command_time == 0.0
                    or now - last_motion_command_time >= args.command_hold_seconds
                )
                if execute_this_primitive and command_due:
                    response = post_trajectory(
                        args.server,
                        selected_index,
                        world_primitive,
                        args.velocity,
                        True,
                    )
                    last_motion_command_time = now
                    last_recovery_command_time = 0.0
                    active_primitive_index = selected_index
                elif execute_this_primitive:
                    held_for = now - last_motion_command_time
                    response = {
                        "accepted": False,
                        "reason": (
                            f"holding primitive {active_primitive_index} "
                            f"({held_for:.1f}/{args.command_hold_seconds:.1f}s)"
                        ),
                    }
                else:
                    response = post_trajectory(
                        args.server,
                        selected_index,
                        world_primitive,
                        float(forward_speed),
                        False,
                    )

            update_open3d(
                visualizer,
                visualizer_state,
                vbg_wrapper.vbg,
                tsdf_weight_threshold,
                transformed_lines,
            )

            display_rgb = cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR)
            status = (
                f"frame={fused_frames}  Zoe={inference_seconds * 1000:.0f} ms  "
                f"primitive={selected_index}/{len(trajectory_lines)-1}  "
                f"goal={distance_to_goal:.2f} m  clearance={args.min_dist2obs:.2f} m  "
                f"TSDF={point_count}/{args.min_tsdf_points} pts  map_ready={map_ready}  "
                f"blocks={tsdf_block_count}  "
                f"speckles={spatial_speckle_pixels}/{temporal_speckle_pixels}  "
                f"unsafe={unsafe_frame_count}/{args.stop_confirm_frames}  "
                f"blocked={should_stop}  recovery={recovery_mode}  "
                f"stop={stop_reason if mission_complete else 'no'}"
            )
            cv2.putText(display_rgb, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if args.execute and not flight_started:
                cv2.putText(
                    display_rgb,
                    "READY - click here and press S/SPACE to fly or restart",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 215, 255),
                    2,
                )
            depth_label = (
                "ZoeDepth metric depth"
                if args.depth_source == "zoe"
                else "Isaac Sim ground-truth depth"
            )
            cv2.putText(depth_colormap, depth_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if not args.headless:
                cv2.imshow(rgb_window, display_rgb)
                cv2.imshow(depth_window, depth_colormap)
            if os.environ.get('WS2_INFERENCE_DIR'):
                # The same inference images as the desktop windows, plus actual
                # TSDF points and candidate primitives projected onto world XY.
                upper=np.hstack([cv2.resize(cv2.cvtColor(kinect_rgb,cv2.COLOR_RGB2BGR),(640,360)),cv2.resize(depth_colormap,(640,360))])
                cv2.rectangle(upper,(0,0),(1279,34),(20,20,20),-1)
                cv2.putText(upper,'RGB input',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(235,235,235),1)
                cv2.putText(upper,depth_label,(652,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(235,235,235),1)
                lower=np.full((260,1280,3),24,dtype=np.uint8)
                def project_xy(points):
                    xy=(np.asarray(points)[:,:2]-pose[:2,3])*45
                    return np.column_stack((xy[:,0]+320,145-xy[:,1])).astype(np.int32)
                points=visualizer_state.get('preview_points',np.empty((0,3)))
                if len(points):
                    pixels=project_xy(points);valid=(pixels[:,0]>=0)&(pixels[:,0]<640)&(pixels[:,1]>=35)&(pixels[:,1]<260)
                    lower[pixels[valid,1],pixels[valid,0]]=(130,130,130)
                for idx,line in enumerate(transformed_lines):
                    pts=project_xy(np.asarray(line.points))
                    cv2.polylines(lower,[pts],False,(0,230,0) if idx==selected_index else (90,90,90),2 if idx==selected_index else 1)
                cv2.circle(lower,(320,145),5,(255,255,255),-1)
                cv2.putText(lower,'TSDF + candidate paths (XY top view)',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),1)
                lines=[f'MonoNav | {args.depth_source} depth -> TSDF -> motion primitive',
                       f'Chosen primitive: {selected_index}   Goal distance: {distance_to_goal:.2f} m',
                       f'TSDF points: {point_count}   Map ready: {map_ready}',
                       f'Inference: {inference_seconds*1000:.0f} ms   Frame: {fused_frames}',
                       f'Blocked: {should_stop}   Recovery: {recovery_mode}']
                for idx,label in enumerate(lines):cv2.putText(lower,label,(650,45+idx*40),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1)
                directory=Path(os.environ['WS2_INFERENCE_DIR']);directory.mkdir(parents=True,exist_ok=True)
                ok,encoded=cv2.imencode('.jpg',np.vstack([upper,lower]),[cv2.IMWRITE_JPEG_QUALITY,85])
                if ok:
                    temp=directory/'mononav.jpg.tmp';temp.write_bytes(encoded.tobytes());temp.replace(directory/'mononav.jpg')
                    info={'planner':'mononav','run_id':os.environ.get('WS2_RUN_ID'),'wall_time':time.time(),
                          'sequence':int(metadata['sequence']),'sim_stamp':metadata['stamp'],
                          'inference_ms':inference_seconds*1000,'primitive':int(selected_index),
                          'goal_distance':float(distance_to_goal),'tsdf_points':int(point_count),
                          'blocked':bool(should_stop),'depth_source':args.depth_source}
                    temp=directory/'mononav.json.tmp';temp.write_text(json.dumps(info));temp.replace(directory/'mononav.json')
            key = -1 if args.headless else cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.execute and not flight_started and key in (ord("s"), ord(" ")):
                mission_complete = False
                stop_reason = "running"
                pause_sent = False
                unsafe_frame_count = 0
                recovery_steps = 0
                last_motion_command_time = 0.0
                last_recovery_command_time = 0.0
                active_primitive_index = None
                flight_started = True
                goal_position = (pose @ np.array([0.0, 0.0, args.goal_distance, 1.0]))[:3].reshape(1, 3)
                initial_camera_altitude = float(pose[2, 3])
                print(f"START GATE OPEN: flight enabled; goal={goal_position[0].round(3).tolist()}")
            elif args.execute and flight_started and key == ord("p"):
                print(f"MANUAL PAUSE: {post_pause(args.server)}")
                flight_started = False
                last_motion_command_time = 0.0
                last_recovery_command_time = 0.0
                active_primitive_index = None
            print(
                f"frame={fused_frames} seq={last_sequence} primitive={selected_index} "
                f"tsdf_points={point_count} tsdf_blocks={tsdf_block_count} blocked={should_stop} "
                f"speckles={spatial_speckle_pixels}/{temporal_speckle_pixels} "
                f"zoe={inference_seconds:.3f}s bridge={response}",
                flush=True,
            )

            remaining = min_period - (time.monotonic() - loop_start)
            if remaining > 0:
                pump_open3d_events(visualizer, remaining)
    finally:
        if args.execute:
            print(f"Requesting hover: {post_pause(args.server)}")
        cv2.destroyAllWindows()
        if visualizer is not None:
            visualizer.destroy_window()


if __name__ == "__main__":
    main()
