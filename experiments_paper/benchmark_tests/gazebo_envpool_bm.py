# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import os 
import gzdrl.envs as envs as envpool
import argparse
import time 
import numpy as np
import yaml 
import pandas as pd
from multiprocessing import Process, Pipe
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def parse_sweep():
    """ Checks if we are running a sweep and returns the sweep parameters """
    parser = argparse.ArgumentParser()
    parser.add_argument("-sweep", action="store_true", help="Run a sweep over n_envs and batch_size")
    parser.add_argument("-sweep_config", type=str, default="sweep_config.yml", help="Path to sweep configuration yaml file")
    return parser.parse_known_args()[0] 

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_envs", type=int, default=os.cpu_count(), help="Number of parallel envs")
    parser.add_argument("-env_name", type=str, default="GazeboPoolHoverEnv-v0", help="Name of the environment", choices=envpool.list_all_envs())
    parser.add_argument("-batch_size", type=int, required=True, help="Batch size for RL training")
    parser.add_argument("-n_steps", type=int, default=500, help="Number of steps to run for benchmark")
    parser.add_argument("-thread_affinity", action="store_true", help="Set thread affinity for envpool workers")
    parser.add_argument("-num_threads", type=int, default=0, help="Number of threads per envpool worker. 0 for auto determination")
    return parser.parse_args()

def generate_sweep_parameters(sweep_config):
    """ Generates a list of sweep parameters from the sweep configuration dictionary """
    n_envs_list = sweep_config.get("n_envs")
    if n_envs_list.get("log_scale", False):
        base = n_envs_list["step"]
        n_envs_list = list(np.logspace(np.log(n_envs_list["start"])/np.log(base),
                                      np.log(n_envs_list["end"])/np.log(base),
                                      num=int((np.log(n_envs_list["end"])-np.log(n_envs_list["start"]))/np.log(base))+1,
                                      base=base,
                                      dtype=int))
    else:
        n_envs_list = list(range(n_envs_list["start"],
                             n_envs_list["end"] + 1,
                             n_envs_list["step"]))
    batch_size_list = sweep_config.get("batch_size")
    if batch_size_list.get("log_scale", False):
        base = batch_size_list["step"]
        batch_size_list = list(np.logspace(np.log(batch_size_list["start"])/np.log(base),
                                          np.log(batch_size_list["end"])/np.log(base),
                                          num=int((np.log(batch_size_list["end"])-np.log(batch_size_list["start"]))/np.log(base))+1,
                                          base=base,
                                          dtype=int))
    else:
        batch_size_list = list(range(batch_size_list["start"], 
                                 batch_size_list["end"] + 1,
                                 batch_size_list["step"]))
    
    num_threads_list = sweep_config.get("num_threads")
    if num_threads_list.get("log_scale", False):
        base = num_threads_list["step"]
        num_threads_list = list(np.logspace(np.log(num_threads_list["start"])/np.log(base),
                                           np.log(num_threads_list["end"])/np.log(base),
                                           num=int((np.log(num_threads_list["end"])-np.log(num_threads_list["start"]))/np.log(base))+1,
                                           base=base,
                                           dtype=int))
    else:
        num_threads_list = list(range(num_threads_list["start"],
                                  num_threads_list["end"] + 1,
                                  num_threads_list["step"]))
    thread_affinity_list = sweep_config.get("thread_affinity")
    if thread_affinity_list.get("log_scale", False):
        base = thread_affinity_list["step"]
        thread_affinity_list = list(np.logspace(np.log(thread_affinity_list["start"])/np.log(base),
                                               np.log(thread_affinity_list["end"])/np.log(base),
                                               num=int((np.log(thread_affinity_list["end"])-np.log(thread_affinity_list["start"]))/np.log(base))+1,
                                               base=base,
                                               dtype=int))
    else:
        thread_affinity_list = list(range(thread_affinity_list["start"],
                                      thread_affinity_list["end"] + 1,
                                      thread_affinity_list["step"]))
    env_name = sweep_config.get("env_name")
    sweep_parameters = []
    # df = pd.DataFrame(columns=["num_envs", 
    #                            "batch_size", 
    #                            "num_threads", 
    #                            "thread_affinity_offset", 
    #                            "env_name"]
    #                   )
    for n_envs in n_envs_list:
        for batch_size in batch_size_list:
            for num_threads in num_threads_list:
                for thread_affinity in thread_affinity_list:
                    if batch_size<=n_envs:
                        params = {
                            "num_envs": n_envs,
                            "batch_size": batch_size,
                            "num_threads": num_threads,
                            "thread_affinity_offset": thread_affinity,
                            "env_name": env_name
                        }
                        sweep_parameters.append(params.copy())
                    # df = pd.concat([df, pd.DataFrame([params])], ignore_index=True)
    # df.to_csv("sweep_parameters.csv", index=False)
    return sweep_parameters

def run_process(kwargs, conn):
    """Runs envpool in separate process,
    this ensures that the envpool envs are destroyed properly after use
    Otherwise C++ side may keep accumulating envs and lead to memory leak
    """
    import gzdrl.envs as envs as envpool
    
    n_envs = kwargs["num_envs"]
    env_name = kwargs["env_name"]
    batch_size = kwargs["batch_size"]
    thread_affinity = kwargs["thread_affinity_offset"] 
    num_threads = kwargs["num_threads"] 
    n_steps = kwargs.get("n_steps", 500)
    
    env = envpool.make_gym( env_name, num_envs=n_envs,
                            batch_size=batch_size,
                            thread_affinity_offset=thread_affinity,
                            num_threads=num_threads)
    env.async_reset()
    action = 0.0*np.array([env.action_space.sample() for _ in range(batch_size)])
    t = time.perf_counter()
    for _ in range(n_steps):
        info = env.recv()[-1]
        env.send(action, info["env_id"])
    duration = time.perf_counter() - t
    fps = n_steps * batch_size / duration 
    print(f"Duration = {duration:.2f}s")
    print(f"GazeboEnvPool FPS = {fps:.2f}")
    del env
    conn.send((duration, fps))
    conn.close()
            
if __name__ == "__main__":
    sweep_args = parse_sweep()
    if sweep_args.sweep:
        """Sweep benchmark run"""
        with open(sweep_args.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)
        parameters = generate_sweep_parameters(sweep_config)
        for i in range(len(parameters)):
            print(f"Running benchmark {i+1}/{len(parameters)} with parameters: {parameters[i]}")
            parent_conn, child_conn = Pipe()
            p = Process(target=run_process, args=(parameters[i], child_conn))
            p.start()
            duration, fps = parent_conn.recv()
            p.join()
            parameters[i]["duration"] = duration
            parameters[i]["fps"] = fps
        import pickle 
        with open("sweep_benchmark_results.pkl", "wb") as f:
            pickle.dump(parameters, f)
        df = pd.DataFrame(parameters)
        df.to_csv("sweep_benchmark_results.csv", index=False)
    else:
        """Normal benchmark run"""
        args = parse()
        n_envs = args.n_envs 
        env_name = args.env_name
        batch_size = args.batch_size
        thread_affinity = 0 if args.thread_affinity else -1
        num_threads = args.num_threads 
        env = envpool.make_gym( env_name, num_envs=n_envs,
                                batch_size=batch_size,
                                thread_affinity_offset=thread_affinity,
                                num_threads=num_threads)
        env.async_reset()
        action = 0.0*np.array([env.action_space.sample() for _ in range(batch_size)])
        for _ in range(n_envs):
            info = env.recv()[-1]
            env.send(action, info["env_id"])
        t = time.perf_counter()
        for _ in range(args.n_steps):
            info = env.recv()[-1]
            env.send(action, info["env_id"])
        duration = time.perf_counter() - t
        fps = args.n_steps * args.batch_size / duration 
        print(f"Duration = {duration:.2f}s")
        print(f"GazeboEnvPool FPS = {fps:.2f}")
