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


def create_visualizer():
    visualizer = o3d.visualization.Visualizer()
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
    return visualizer


def update_open3d(visualizer, vbg, weight_threshold, trajectory_lines, first_view):
    point_cloud = vbg.extract_point_cloud(weight_threshold).cpu().to_legacy()
    point_count = len(point_cloud.points)
    visualizer.clear_geometries()
    # The first few fused frames may not yet meet the TSDF weight threshold.
    # Keep requesting an automatic view fit until a non-empty cloud exists;
    # otherwise Open3D permanently points at an empty bounding box.
    reset_view = first_view and point_count > 0
    visualizer.add_geometry(point_cloud, reset_bounding_box=reset_view)
    for line in trajectory_lines:
        visualizer.add_geometry(line, reset_bounding_box=False)
    visualizer.poll_events()
    visualizer.update_renderer()
    return not reset_view and first_view, point_count


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
    visualizer = create_visualizer()
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
    first_view = True
    selected_index = len(trajectory_lines) // 2
    min_period = 1.0 / max(args.rate, 0.1)
    mission_complete = False
    pause_sent = False
    stop_reason = ""
    flight_started = not args.wait_for_start

    print(f"Connecting to AirStack bridge at {args.server}")
    print("Controls: drag/scroll in the Open3D window to change viewpoint; press Q in RGB/depth to stop")
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
                    )
                    if chosen is not None:
                        selected_index = chosen
                except (ValueError, RuntimeError) as exc:
                    print(f"Planner waiting for a usable reconstruction: {exc}")

            transformed_lines[selected_index].paint_uniform_color([0.0, 1.0, 0.0])
            world_primitive = transform_primitive(trajectory_lines[selected_index], pose)
            if flight_started and should_stop:
                mission_complete = True
                stop_reason = f"no primitive satisfies {args.min_dist2obs:.2f} m clearance"
            if mission_complete and args.execute and not pause_sent:
                print(
                    f"Mission stopping ({stop_reason}) at goal distance "
                    f"{distance_to_goal:.2f} m: {post_pause(args.server)}"
                )
                pause_sent = True
            execute_this_primitive = (
                args.execute
                and flight_started
                and fused_frames >= args.warmup_frames
                and not should_stop
                and not mission_complete
            )
            response = post_trajectory(
                args.server,
                selected_index,
                world_primitive,
                args.velocity if execute_this_primitive else float(forward_speed),
                execute_this_primitive,
            )

            first_view, point_count = update_open3d(
                visualizer,
                vbg_wrapper.vbg,
                config["weight_threshold"],
                transformed_lines,
                first_view,
            )

            display_rgb = cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR)
            status = (
                f"frame={fused_frames}  Zoe={inference_seconds * 1000:.0f} ms  "
                f"primitive={selected_index}/{len(trajectory_lines)-1}  "
                f"goal={distance_to_goal:.2f} m  clearance={args.min_dist2obs:.2f} m  "
                f"TSDF={point_count} pts  stop={mission_complete}"
            )
            cv2.putText(display_rgb, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if args.execute and not flight_started and not mission_complete:
                cv2.putText(
                    display_rgb,
                    "READY - click this window and press S or SPACE to fly",
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
            if args.execute and not flight_started and not mission_complete and key in (ord("s"), ord(" ")):
                flight_started = True
                goal_position = (pose @ np.array([0.0, 0.0, args.goal_distance, 1.0]))[:3].reshape(1, 3)
                initial_camera_altitude = float(pose[2, 3])
                print(f"START GATE OPEN: flight enabled; goal={goal_position[0].round(3).tolist()}")
            elif args.execute and flight_started and key == ord("p"):
                print(f"MANUAL PAUSE: {post_pause(args.server)}")
                flight_started = False
            print(
                f"frame={fused_frames} seq={last_sequence} primitive={selected_index} "
                f"tsdf_points={point_count} zoe={inference_seconds:.3f}s bridge={response}",
                flush=True,
            )

            remaining = min_period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if args.execute:
            print(f"Requesting hover: {post_pause(args.server)}")
        cv2.destroyAllWindows()
        visualizer.destroy_window()


if __name__ == "__main__":
    main()
