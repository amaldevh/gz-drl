# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import argparse 
import numpy as np
import onnxruntime as ort
import gzdrl.envs as envs 
algo_singlerl = ["ppo", "sac", "td3", "ddpg", "a2c"]
algo_singlerl = algo_singlerl + [algo.upper() for algo in algo_singlerl]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-env_name", type=str, required=True, help="Name of the environment", choices=envs.list_all_envs())
    parser.add_argument("-onnxfile", type=str, required=True, help="Checkpoint to load")
    return parser.parse_args()


def control_loop(env, obs, session):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = np.expand_dims(obs, axis=0)
    action = session.run(["action"], {"observation": obs})[0]
    obs, rew, done, trunc, info = env.step(action)
    return obs, rew, done, trunc, info

def plot(states, trajectories):
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    fig, axs = plt.subplots(3,1)
    fig2, axs2 = plt.subplots(3,1)
    rpy_tracked = Rotation.from_quat(states[:,6:10], scalar_first=True).as_euler('xyz')
    rpy_des = Rotation.from_quat(trajectories[:,6:10], scalar_first=True).as_euler('xyz')

    for i in range(3):
        axs[i].plot(trajectories[:,i ], '--')
        axs[i].plot(states[:,i])
        axs2[i].plot(rpy_des[:,i ], '--')
        axs2[i].plot(rpy_tracked[:,i])
    plt.show()

if __name__ == "__main__":
    args = parse()
    onnxfile = args.onnxfile
    session  = ort.InferenceSession(onnxfile, providers=["GPUExecutionProvider",
    "CPUExecutionProvider"])
    env = envs.make_gymnasium(args.env_name)
    def eval():
        states = []
        trajectories = []
        obs, info = env.reset()
        states.append(info["state"])
        trajectories.append(info["trajectory"])
        done = False
        
        while not done:
            obs, rew, done, trunc, info = control_loop(
                env, obs, session
            )
            states.append(info["state"])
            trajectories.append(info["trajectory"])
            done = done or trunc
        return np.array(states).squeeze(), np.array(trajectories).squeeze()
    st, tr = eval()
    plot(st, tr)
