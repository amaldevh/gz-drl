# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import pandas as pd 
import yaml
import numpy as np
waypoints = yaml.safe_load(open("/tmp/waypoints.yaml", "r"))
n_waypoints = waypoints["num_waypoints"]
waypoints = waypoints["waypoints"]
wpts = np.array([waypoints[i]["center"] for i in range(n_waypoints)])
corners = np.array([waypoints[i]["corners"] for i in range(n_waypoints)])
rots = []
for i in range(n_waypoints):
    corner = corners[i]
    center = wpts[i]
    up = corner[0] + corner[1]
    left = corner[0] + corner[3]
    forward = np.cross(left, up)
    rot = np.array((forward, left, up)).T
    rots.append(rot)
    
wx, wy, wz = wpts.T.tolist()

data = pd.read_csv("/tmp/inference_log.csv")
xyz = (data[[f"state_{i}" for i in range(3)]]).to_numpy()
vxyz = (data[[f"state_{i}" for i in range(3, 6)]]).to_numpy()
reward, total_reward = data["reward"].to_numpy(), data["total_reward"].to_numpy()
vel_norm = np.linalg.norm(vxyz, axis=1)
print("Average velocity norm: ", vel_norm.mean())
# COMPUTE AVERGAE ANALYTICAL VEL FOR A SINGLE LAP AROUND THE WAYPOINTS
# std::vector<double> base_curve_amps_ {1.5, 1.5 , 0.32}; // changed
# std::vector<double> base_curve_freqs_ {1.0, 2.0, 2.0}; // changed
# std::vector<double> base_curve_phases_ {0.0, 0.0, M_PI/2.0}; // changed
# calciulate period by with a, b, c freqs
# calculate_period(double freq_a, double freq_b, double freq_c) {
#         // 1. Find the GCD of all three frequencies
#         // GCD(a, b, c) = GCD(a, GCD(b, c))
#         double common_freq = gcd_double(freq_a, gcd_double(freq_b, freq_c));

#         // Avoid division by zero if frequencies are 0 (static position)
#         if (common_freq < EPSILON) {
#             return 0.0; 
#         }

#         // 2. Calculate Period
#         return (2.0 * PI) / common_freq;
#     }

def calculate_gcd(a, b):
    while b:
        a, b = b, a % b
    return a
# gcd_double(double a, double b) {
#         a = std::abs(a);
#         b = std::abs(b);
        
#         while (b > EPSILON) {
#             double remainder = std::fmod(a, b);
            
#             // Handle precision edge case where fmod returns almost b
#             if (std::abs(b - remainder) < EPSILON) {
#                 remainder = 0.0;
#             }
            
#             a = b;
#             b = remainder;
#         }
#         return a;
#     }
def calculate_gcd_double(a, b, eps=1e-6):
    a = abs(a)
    b = abs(b)
    
    while b > eps:
        remainder = np.fmod(a, b)
        
        # Handle precision edge case where fmod returns almost b
        if abs(b - remainder) < eps:
            remainder = 0.0
        
        a, b = b, remainder
    
    return a
def calculate_period(freq_a, freq_b, freq_c):
    common_freq = calculate_gcd_double(freq_a, calculate_gcd_double(freq_b, freq_c))
    if common_freq < 1e-6:
        return 0.0
    return (2.0 * np.pi) / common_freq


base_curve_amps_ = np.array([1.5, 1.5 , 0.32])
base_curve_freqs_ = np.array([1.0*np.pi/6, 2.0*np.pi/6, 2.0*np.pi/6])
base_curve_phases_ = np.array([0.0, 0.0, np.pi/2.0])
t = np.linspace(0, calculate_period(*base_curve_freqs_), 1000)
curve_vel = base_curve_freqs_ * base_curve_amps_ * np.cos(base_curve_freqs_ * t[:,None] + base_curve_phases_)
curve_vel_norm = np.linalg.norm(curve_vel, axis=1)

print("Average analytical velocity norm: ", curve_vel_norm.mean())
x, y, z = xyz.T.tolist()
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

norm = plt.Normalize(vel_norm.min(), vel_norm.max())
xyzp = xyz.reshape(-1,1,3)
segments = np.concatenate([xyzp[:-1], xyzp[1:]], axis=1)
lc = Line3DCollection(segments, cmap='viridis', norm=norm)
lc.set_array(vel_norm[:-1])
lc.set_linewidth(2)
# compute min dist to each wpt
min_dists = []
for i in range(n_waypoints):
    gate_pos = wpts[i]
    dists = np.linalg.norm(xyz - gate_pos, axis=1)
    min_dists.append(dists.min())
    print(f"Min distance to gate {i}: {dists.min():.3f}")
    
ax = plt.figure().add_subplot(projection='3d')
# plot the gatezs
rots = np.array(rots)
for (gate, center, rot) in zip(corners, wpts, rots):
    ax.plot(gate[:,0], gate[:,1], gate[:,2], c='b')
    ax.plot(gate[[-1,0],0], gate[[-1,0],1], gate[[-1,0],2], c='b')
    # ax.quiver(center[0], center[1], center[2], rot[0,0], rot[1,0], rot[2,0], length=0.25, normalize=True)
    
ax.add_collection3d(lc)
ax.plot(x, y, z, label='Drone Path')
ax.scatter(wx, wy, wz, c='r', marker='o', label='Waypoints')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(0, 5)
ax.legend()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(reward, label='Reward')
# ax.plot(total_reward, label='Total Reward')
ax.set_xlabel('Timestep')
ax.set_ylabel('Reward')
ax.legend()
plt.show()