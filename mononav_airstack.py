"""Run MonoNav online from AirStack RGB and ground-truth optical-camera poses."""

import argparse
import copy
import json
import struct
import time
import urllib.error
import urllib.request

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
)


def fetch_frame(server_url):
    with urllib.request.urlopen(server_url.rstrip("/") + "/frame", timeout=3.0) as response:
        body = response.read()
    metadata_length = struct.unpack("!I", body[:4])[0]
    metadata = json.loads(body[4 : 4 + metadata_length])
    jpeg = np.frombuffer(body[4 + metadata_length :], dtype=np.uint8)
    bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("AirStack bridge returned an invalid JPEG")
    return metadata, bgr


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


def pump_open3d_events(visualizer, duration):
    """Keep mouse navigation responsive while waiting for the next fusion tick."""
    deadline = time.monotonic() + max(0.0, duration)
    while time.monotonic() < deadline:
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://airstack-robot-desktop-1:8765")
    parser.add_argument("--rate", type=float, default=2.0, help="maximum fusion rate in Hz")
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="request actual AirStack execution")
    parser.add_argument(
        "--wait-for-start",
        action="store_true",
        help="fuse and visualize immediately, but wait for S/Space in an OpenCV window before flight",
    )
    parser.add_argument("--goal-distance", type=float, default=8.0)
    parser.add_argument("--velocity", type=float, default=0.35)
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
    parser.add_argument("--recovery-yaw-deg", type=float, default=25.0)
    parser.add_argument("--recovery-radius", type=float, default=0.03)
    parser.add_argument("--recovery-velocity", type=float, default=0.03)
    parser.add_argument("--max-recovery-steps", type=int, default=12)
    parser.add_argument(
        "--min-tsdf-points",
        type=int,
        default=1000,
        help="do not execute motion until this many weighted TSDF surface points exist",
    )
    args = parser.parse_args()

    config = load_config("config.yml")
    print("Loading ZoeDepth...")
    zoe_config = get_config("zoedepth", config["zoedepth_mode"])
    zoe = build_model(zoe_config)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = zoe.to(device_name)
    zoe.eval()
    print(f"ZoeDepth ready on {device_name}")

    settings = config["VoxelBlockGrid"]
    vbg_wrapper = VoxelBlockGrid(
        settings["depth_scale"],
        settings["depth_max"],
        settings["trunc_voxel_multiplier"],
        o3d.core.Device(settings["device"]),
    )
    trajectory_list = get_trajlist(config["trajlib_dir"])
    trajectory_lines, _, forward_speed, _ = get_traj_linesets(trajectory_list)
    visualizer, visualizer_state = create_visualizer()
    rgb_window = "MonoNav - live AirStack RGB"
    depth_window = "MonoNav - live ZoeDepth"
    for window_name, left, top in (
        (rgb_window, 930, 40),
        (depth_window, 930, 450),
    ):
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
    stop_reason = ""
    flight_started = not args.wait_for_start

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
                metadata, bgr = fetch_frame(args.server)
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                print(f"Waiting for AirStack frame: {exc}")
                time.sleep(1.0)
                continue
            if metadata["sequence"] == last_sequence:
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

            source_intrinsic = camera_matrix(metadata)
            zero_distortion = np.zeros(5, dtype=np.float64)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kinect_rgb = transform_image(rgb, source_intrinsic, zero_distortion, kinect)

            inference_start = time.monotonic()
            with torch.inference_mode():
                depth_mm, depth_colormap = compute_depth(kinect_rgb, zoe)
            inference_seconds = time.monotonic() - inference_start
            vbg_wrapper.integration_step(
                cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR), depth_mm, pose
            )
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
                        config["weight_threshold"],
                        2,  # AirStack map frame is Z-up.
                        float(pose[2, 3]),
                        args.planning_height_band,
                    )
                    if chosen is not None:
                        selected_index = chosen
                except (ValueError, RuntimeError) as exc:
                    print(f"Planner waiting for a usable reconstruction: {exc}")

            transformed_lines[selected_index].paint_uniform_color([0.0, 1.0, 0.0])
            world_primitive = transform_primitive(trajectory_lines[selected_index], pose)
            map_ready = point_count >= args.min_tsdf_points
            extreme_selected = chosen in (0, len(trajectory_lines) - 1)
            if chosen == 0:
                recovery_direction = 1
            elif chosen == len(trajectory_lines) - 1:
                recovery_direction = -1
            recovery_mode = "none"
            if flight_started and map_ready and should_stop:
                unsafe_frame_count += 1
                if unsafe_frame_count == 1:
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
                yaw_scan = make_yaw_scan(
                    pose,
                    recovery_direction,
                    args.recovery_yaw_deg,
                    args.recovery_radius,
                )
                response = post_trajectory(
                    args.server, -1, yaw_scan, args.recovery_velocity, True
                )
                recovery_steps += 1
                print(
                    f"RECOVERY {recovery_steps}/{args.max_recovery_steps}: "
                    f"{recovery_mode}, yaw_step={recovery_direction * args.recovery_yaw_deg:.1f} deg"
                )
            else:
                response = post_trajectory(
                    args.server,
                    selected_index,
                    world_primitive,
                    args.velocity if execute_this_primitive else float(forward_speed),
                    execute_this_primitive,
                )

            point_count = update_open3d(
                visualizer,
                visualizer_state,
                vbg_wrapper.vbg,
                config["weight_threshold"],
                transformed_lines,
            )

            display_rgb = cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR)
            status = (
                f"frame={fused_frames}  Zoe={inference_seconds * 1000:.0f} ms  "
                f"primitive={selected_index}/{len(trajectory_lines)-1}  "
                f"goal={distance_to_goal:.2f} m  clearance={args.min_dist2obs:.2f} m  "
                f"TSDF={point_count}/{args.min_tsdf_points} pts  map_ready={map_ready}  "
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
            cv2.putText(depth_colormap, "ZoeDepth metric depth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(rgb_window, display_rgb)
            cv2.imshow(depth_window, depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.execute and not flight_started and key in (ord("s"), ord(" ")):
                mission_complete = False
                stop_reason = "running"
                pause_sent = False
                unsafe_frame_count = 0
                recovery_steps = 0
                flight_started = True
                goal_position = (pose @ np.array([0.0, 0.0, args.goal_distance, 1.0]))[:3].reshape(1, 3)
                initial_camera_altitude = float(pose[2, 3])
                print(f"START GATE OPEN: flight enabled; goal={goal_position[0].round(3).tolist()}")
            elif args.execute and flight_started and key == ord("p"):
                print(f"MANUAL PAUSE: {post_pause(args.server)}")
                flight_started = False
            print(
                f"frame={fused_frames} seq={last_sequence} primitive={selected_index} "
                f"tsdf_points={point_count} blocked={should_stop} "
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
        visualizer.destroy_window()


if __name__ == "__main__":
    main()
