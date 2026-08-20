"""
  __  __                   _   _             
 |  \/  | ___  _ __   ___ | \ | | __ ___   __
 | |\/| |/ _ \| '_ \ / _ \|  \| |/ _` \ \ / /
 | |  | | (_) | | | | (_) | |\  | (_| |\ V / 
 |_|  |_|\___/|_| |_|\___/|_| \_|\__,_| \_/  
Copyright (c) 2023 Nate Simon
License: MIT
Authors: Nate Simon and Anirudha Majumdar, Princeton University
Project Page: https://natesimon.github.io/mononav

The purpose of this script is to fuse depth images and poses into a 3D reconstruction.
Here, we use Open3D's tensor reconstruction system: the VoxelBlockGrid.

After fusion, the reconstruction is visualized (in addition to the camera poses), and saved to file.

"""

import numpy as np
import time
import os

import open3d as o3d
from PIL import Image
import numpy as np
import yaml
from utils.utils import *
#####################################################################

addPose = True
auto_capture = os.environ.get("MONONAV_AUTO_CAPTURE", "false").lower() in ("1", "true", "yes")
live_visualization = os.environ.get("MONONAV_LIVE_VISUALIZATION", "false").lower() in ("1", "true", "yes")
live_hold = os.environ.get("MONONAV_LIVE_HOLD", "true").lower() in ("1", "true", "yes")
live_delay_s = int(os.environ.get("MONONAV_LIVE_FUSION_DELAY_MS", "75")) / 1000.0

CONFIG_PATH = "config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

data_dir = config["data_dir"] # parent directory to look for RGB images, and save depth images

source = "kinect" # meaning: crazyflie images have been undistorted to match kinect
rgb_dir = data_dir + "/" + source + "-rgb-images/"
depth_dir = data_dir + "/" + source + "-depth-images"
pose_dir = data_dir + "/crazyflie-poses/"
#####################################################################

# Initialize TSDF VoxelBlockGrid
depth_scale = config["VoxelBlockGrid"]["depth_scale"]
depth_max = config["VoxelBlockGrid"]["depth_max"]
trunc_voxel_multiplier = config["VoxelBlockGrid"]["trunc_voxel_multiplier"]
weight_threshold = config["weight_threshold"] # for planning and visualization (!! important !!)
device = config["VoxelBlockGrid"]["device"]

vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device))
        
#####################################################################

poses = [] # for visualization
t_start = time.time()

live_visualizer = None
live_pcd = None
live_pose_lineset = None
live_window_open = False
if live_visualization:
    live_visualizer = o3d.visualization.Visualizer()
    live_window_open = live_visualizer.create_window(
        window_name="MonoNav 2/3 - Live TSDF fusion (close to continue)",
        width=1280,
        height=720,
    )

depth_files = [name for name in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, name)) and name.endswith(".jpg")]
depth_files = sorted(depth_files)

# Get last frame
first_frame = split_filename(depth_files[0])
end_frame = split_filename(depth_files[-1])

for filename in depth_files:
    # Get the frame number from the depth filename
    frame_number = split_filename(filename)
    print("Integrating frame %d/%d"%(frame_number,end_frame))
    # Get rbg_file
    rgb_file = rgb_dir + source + "_frame-%06d.rgb.jpg"%(frame_number)

    # Read in camera pose
    pose_file = data_dir + "/crazyflie-poses/crazyflie_frame-%06d.pose.txt"%(frame_number)
    cam_pose = np.loadtxt(pose_file)
    poses.append(cam_pose)

    # Get color image with Pillow and convert to RGB
    color = Image.open(rgb_file).convert("RGB")  # load

    # Integrate
    depth_file = depth_dir + "/" + source + "_frame-%06d.depth.npy"%(frame_number)
    depth_numpy = np.load(depth_file) # mm
    vbg.integration_step(color, depth_numpy, cam_pose)

    if live_window_open:
        updated_pcd = vbg.vbg.extract_point_cloud(weight_threshold).to_legacy()
        updated_pose_lineset = get_poses_lineset(poses)
        if (live_pcd is None and len(updated_pcd.points) > 0
                and len(updated_pose_lineset.lines) > 0):
            live_pcd = updated_pcd
            live_pose_lineset = updated_pose_lineset
            live_visualizer.add_geometry(live_pcd)
            live_visualizer.add_geometry(live_pose_lineset, reset_bounding_box=False)
            live_visualizer.reset_view_point(True)
        elif live_pcd is not None:
            live_pcd.points = updated_pcd.points
            live_pcd.colors = updated_pcd.colors
            live_pose_lineset.points = updated_pose_lineset.points
            live_pose_lineset.lines = updated_pose_lineset.lines
            live_pose_lineset.colors = updated_pose_lineset.colors
            live_visualizer.update_geometry(live_pcd)
            live_visualizer.update_geometry(live_pose_lineset)
        live_window_open = live_visualizer.poll_events()
        live_visualizer.update_renderer()
        time.sleep(live_delay_s)

#####################################################################
# Print out timing information
t_end = time.time()
print("Time taken (s): ", t_end - t_start)
print("FPS: ", end_frame/(t_end - t_start))

pcd = vbg.vbg.extract_point_cloud(weight_threshold)

if live_visualization and live_window_open:
    if live_hold:
        print("Live TSDF fusion complete. Adjust the view, then close the window to continue to planning.")
        live_visualizer.run()
    live_visualizer.destroy_window()
elif live_visualization and live_visualizer is not None:
    live_visualizer.destroy_window()
elif addPose:
    pose_lineset = get_poses_lineset(poses)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="MonoNav TSDF reconstruction")
    visualizer.add_geometry(pcd.to_legacy())
    visualizer.add_geometry(pose_lineset)
    for pose in poses:
        # Add coordinate frame ( The x, y, z axis will be rendered as red, green, and blue arrows respectively.)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.5, center=(0, 0, 0))
        visualizer.add_geometry(coordinate_frame.transform(pose))
    if auto_capture:
        visualizer.reset_view_point(True)
        for _ in range(30):
            visualizer.poll_events()
            visualizer.update_renderer()
        capture_path = os.path.join(data_dir, "fusion_view.png")
        visualizer.capture_screen_image(capture_path, do_render=True)
        print("Saved reconstruction capture to:", capture_path)
    else:
        visualizer.run()
    visualizer.destroy_window()
else:
    o3d.visualization.draw([pcd])

#####################################################################

npz_filename = os.path.join(data_dir, "vbg.npz")
ply_filename = os.path.join(data_dir, "pointcloud.ply")
print('Saving npz to {}...'.format(npz_filename))
print('Saving ply to {}...'.format(ply_filename))

vbg.vbg.save(npz_filename)
o3d.io.write_point_cloud(ply_filename, pcd.to_legacy())

print('Saving finished')
