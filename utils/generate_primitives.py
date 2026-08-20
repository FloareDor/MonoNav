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

The purpose of this script is to generate a library of motion primitives (trajectories)
as described in the paper. Each motion primitive is defined by a sinusoidal yawrate profile,
which keeps the initial and final yawrates zero (for smooth chaining of primitives).
The trajectory library is visualized in trajlib/visualization.png.
By changing the trajectory constants and extension segment, you can generate a diverse variety of primitives.
If you change the trajlib_dir, be sure to update `trajlib_dir: 'utils/trajlib/'` in config.yml.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Original MonoNav trajectory constants.  The AirStack adapter uses a separate
# profile and output directory so the offline baseline remains unchanged.
T = 1.0 # s, period of the primitive
num_commands = 65 # number of points in the trajectory
num_points = 8 # how many points should be in each primitive and each extension segment? (for primitive evaluation)

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=("original", "airstack"), default="original")
parser.add_argument("--output-dir")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if args.profile == "original":
    V = 0.5
    max_yawrate = 0.7
    num_trajectories = 7
    default_dir = "trajlib"
else:
    V = 0.4
    max_yawrate = 0.35
    num_trajectories = 13
    default_dir = "trajlib_airstack"

script_dir = os.path.dirname(os.path.abspath(__file__))
trajlib_dir = os.path.abspath(
    args.output_dir or os.path.join(script_dir, default_dir)
)

# Extension segment - straight line at the end of the trajectory (to encourage foresight)
x_ext = np.linspace(0., 1.0, num_points) # extension segment in the body frame (x = forward)
y_ext = np.zeros_like(x_ext)
ext = np.vstack((x_ext, y_ext))

# Assertion statements
assert num_trajectories % 2 == 1, "num_trajectories should be odd"
assert max_yawrate > 0, "max_yawrate should be positive"

# Derived quantities
omega = np.pi/T # 1/s, angular frequency
t = np.linspace(0,T,num_commands) # s, time vector
Avals = np.linspace(max_yawrate, -max_yawrate, num_trajectories) # rad/s, yawrate amplitude vector


# Create a subplot
fig, (ax2, ax1) = plt.subplots(2, 1)#, sharex=True)
fig.tight_layout()

traj_num = 0
os.makedirs(trajlib_dir, exist_ok=True)
for filename in os.listdir(trajlib_dir):
    if (filename.startswith("traj-") and filename.endswith(".npz")) or filename == "visualization.png":
        os.remove(os.path.join(trajlib_dir, filename))

trajlist = []

# For each yawrate profile, determine the associate xdot, ydot commands (which are used for the Crazyflie)
# numerically integrate to obtain x(t) and y(t) (for collision avoidance)
for A in Avals:
    yawrate = A*np.sin(omega*t)
    psi = A*(1 - np.cos(omega*t))
    # Calculate xdot and ydot
    xdot = V * np.cos(psi)
    ydot = V * np.sin(psi)
    # Numerical integration to obtain x(t) and y(t)
    x = np.cumsum(xdot) * (t[1] - t[0])
    y = np.cumsum(ydot) * (t[1] - t[0])

    # rotate and add the extension segment
    yaw = psi[-1]
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
    ext_rot = np.matmul(R,ext) + np.array([[x[-1]],[y[-1]]])
    # sample every num_points for efficient primitive evaluation
    sample = int(num_commands/num_points)
    x_sample = np.hstack((x[::sample], ext_rot[0,:]))
    y_sample = np.hstack((y[::sample], ext_rot[1,:]))

    trajectory = {
        'xvals': x,
        'yvals': y,
        'yawvals': psi,
        'x_sample': x_sample,
        'y_sample': y_sample,
        'yawrates': yawrate,
        'times': t,
        'period': T,
        'forward_speed': V,
        'amplitude': A
    }

    # Save to file
    np.savez(os.path.join(trajlib_dir, 'traj-%02d.npz'%(traj_num)), **trajectory)
    traj_num += 1

    # Top plot: yawrate and psi
    ax2.plot(t, yawrate, label=r'$\dot{\psi}_{max}=$ %.2f' % A)

    # Bottom plot
    ax1.plot(x,y)#, label=r'$\dot{\psi}_{max}=$ %.2f' % A)
    # Plot samples and distance
    ax1.plot(x_sample, y_sample, 'o', color = 'red') # slice every other value

ax1.set_title('Trajectory @ V = %.2f m/s' % V)
ax1.set_xlabel('X position (m)')
ax1.set_ylabel('Y position (m)')
ax1.axis('equal')
#ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Yawrates rad/s')
ax2.set_title('Yawrates')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Suggested plotting adjustment
plt.subplots_adjust(0.125, 0.1, 0.75, 0.9, 0.2, 0.5)

plt.savefig(os.path.join(trajlib_dir, 'visualization.png'), dpi=300)

if args.show:
    plt.show()
else:
    plt.close(fig)
