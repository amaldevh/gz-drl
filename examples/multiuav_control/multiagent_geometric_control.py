# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import gzdrl as grl
import time
import numpy as np
import matplotlib.pyplot as plt 
import tqdm 
import argparse
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-drones", type=int, default=2, help="Number of drones")
    return parser.parse_args()

def make_world(N_agents):
    templ_file = grl.get_sdf_path("world_formation_template.sdf")
    tree = ET.parse(templ_file)
    root = tree.getroot()
    world = root.find("world")

    if world is not None:
        for i in range(N_agents):
            include_tag = ET.Element("include")
            name_tag = ET.SubElement(include_tag, "name")
            name_tag.text = f"quadrotor{i+1}"
            uri_tag = ET.SubElement(include_tag, "uri")
            uri_tag.text = "model://quadrotor.sdf"  
            pose_tag = ET.SubElement(include_tag, "pose")
            pose_tag.text = f"{i*0.6} 0. 4.0 0 0 0"  
            world.append(include_tag)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ") 
    tree.write("/tmp/world_temp.sdf", encoding="utf-8", xml_declaration=True)
    return "/tmp/world_temp.sdf"

if __name__ == "__main__":
    N_drones = parse_args().n_drones 
    model_names = [ f"quadrotor{i+1}" for i in range(N_drones)]
    world = make_world(N_drones)
    serv = grl.DRLServer( "0", 
                    world,
                    model_names,
                    False)
    serv.run_N(10)
    gain_map = grl.GAIN_MAP()
    param_map = grl.PARAMETER_MAP()
    geom_gains = gain_map["qdrone2"]["geometric_controller"]
    geom_params = param_map["qdrone2"]["geometric_controller"]

    controllers = []
    mapping_funcs = []
    trajectory_funcs = []
    for i in range(N_drones):
        controller = grl.GeometricController(geom_gains["kp"], geom_gains["kd"],
                geom_gains["kp_att"], geom_gains["kd_att"],
                geom_params.max_accel,
                geom_params.gravity_vec,
                geom_params.mass,
                geom_params.inertia)
        serv.set_controller(model_names[i], "quadrotor/base_link", controller)
        controllers.append(controller)
        # get map
        drone_mapping_func = serv.get_thrust_moment_to_rotor_velocity_mapping_function(f"quadrotor{i+1}")
        mapping_funcs.append(drone_mapping_func)
        A = min(3., 1.0*(max(N_drones/4, 1)))
        B = min(3., 1.0*(max(N_drones/4, 1)))
        C = 0.5

        c = 1 #2*np.pi/15.0
        b = 1 #c*2
        a = 0.5
        phase = i * np.pi * 2 / N_drones
        trajectory_function = lambda t, phase=phase, A=A, B=B, C=C, a=a, b=b, c=c: np.array([
            A * np.sin(a * (t - phase)),
            B * np.sin(b * (t - phase)),
            7.5 + np.sin(c * (t - phase) + np.pi / 2),
            A * a * np.cos(a * (t - phase)),
            B * b * np.cos(b * (t - phase)),
            C * c * np.cos(c * (t - phase) + np.pi / 2),
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        trajectory_funcs.append(trajectory_function)
    rtf_cb = lambda : time.sleep(1e-3)
  
    
    for i in range(N_drones):
        serv.reset_pos(model_names[i], trajectory_funcs[i](0)[:3], (0,0,0))
    
    for t in range(60000):
        ts = time.time()
        serv.update_control_states()
        for i in range(N_drones):
            state_state_dot = serv.control_states[f"quadrotor{i+1}"]["quadrotor/base_link"] 
            des_states = trajectory_funcs[i](t*1e-3) 
            controls = controllers[i].calculate_thrust_moments(*state_state_dot, des_states) 
            rot_vels = mapping_funcs[i](controls)
            serv.set_rotor_velocity_cmd(model_names[i], "quadrotor/base_link", rot_vels)
        serv.run_N(1)
        time.sleep(max(0, 1e-3- (time.time() - ts)))
