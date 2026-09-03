#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

"""
Plot trajectory tracking results from the C++ NMPC controller.
Reads data from trajectory_results.csv and creates visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys
import os
import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot NMPC trajectory tracking results from C++.")
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to the CSV file containing trajectory results.')
    return parser.parse_args()

def load_results(csv_file='trajectory_results.csv'):
    """Load results from CSV file."""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Make sure to run ./nmpc_test first to generate the data.")
        sys.exit(1)
    
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    
    # Parse columns
    time = data[:, 0]
    states = data[:, 1:14]  # x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz
    desired_states = data[:, 14:20]  # des_x, des_y, des_z, des_vx, des_vy, des_vz
    
    return time, states, desired_states

def plot_performance(time, states, desired_states):
    """Create performance plots similar to the Python NMPC script."""
    
    # Extract state components
    pos = states[:, 0:3]
    vel = states[:, 3:6]
    quat = states[:, 6:10]
    omega = states[:, 10:13]
    
    # Extract desired components
    des_pos = desired_states[:, 0:3]
    des_vel = desired_states[:, 3:6]
    
    # Convert quaternions to Euler angles (Roll, Pitch, Yaw)
    rpys = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler('xyz')
    
    # Create figure with position and phase plots
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    fig.suptitle('NMPC Trajectory Tracking Performance (C++)', fontsize=16)
    
    labels = ['x', 'y', 'z']
    colors_actual = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_desired = ['#aec7e8', '#ffbb78', '#98df8a']
    
    for i in range(3):
        # Position tracking
        axs[i, 0].plot(time, pos[:, i], label=f'Actual {labels[i]}', 
                      color=colors_actual[i], linewidth=2)
        axs[i, 0].plot(time, des_pos[:, i], label=f'Desired {labels[i]}', 
                      color=colors_actual[i], linestyle='--', linewidth=2, alpha=0.7)
        axs[i, 0].set_xlabel('Time (s)', fontsize=10)
        axs[i, 0].set_ylabel(f'{labels[i]} Position (m)', fontsize=10)
        axs[i, 0].legend(loc='best', fontsize=9)
        axs[i, 0].grid(True, alpha=0.3)
        
        # Phase portrait
        axs[i, 1].plot(pos[:, i], vel[:, i], color=colors_actual[i], linewidth=1.5)
        axs[i, 1].set_xlabel(f'{labels[i]} Position (m)', fontsize=10)
        axs[i, 1].set_ylabel(f'{labels[i]} Velocity (m/s)', fontsize=10)
        axs[i, 1].set_title(f'{labels[i]}-axis Phase Portrait', fontsize=10)
        axs[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create figure for attitude
    fig2, ax2 = plt.subplots(3, 1, figsize=(8, 3))
    fig2.suptitle('Attitude (Roll, Pitch, Yaw)', fontsize=16)
    
    rpy_labels = ['Roll', 'Pitch', 'Yaw']
    rpy_colors = ['#d62728', '#9467bd', '#8c564b']
    
    for i in range(3):
        ax2[i].plot(time, np.rad2deg(rpys[:, i]), label=rpy_labels[i], 
                   color=rpy_colors[i], linewidth=2)
        ax2[i].set_xlabel('Time (s)', fontsize=10)
        ax2[i].set_ylabel(f'{rpy_labels[i]} (deg)', fontsize=10)
        ax2[i].legend(loc='best', fontsize=9)
        ax2[i].grid(True, alpha=0.3)
    
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    print("Loading trajectory results...")
    
    csv_file = parse_args().csv
    time, states, desired_states = load_results(csv_file)
    
    print(f"Loaded {len(time)} data points")
    print(f"Time range: {time[0]:.3f} - {time[-1]:.3f} seconds")
    
    plot_performance(time, states, desired_states)
