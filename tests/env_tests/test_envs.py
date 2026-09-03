# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

import unittest
import numpy as np
import gzdrl.envs as envs

class TestEnvs(unittest.TestCase):
    
    def test_all_envs(self):
        all_envs = envs.list_all_envs()
        for env_name in all_envs:
            print(f"Testing environment: {env_name}")
            env = envs.make_gymnasium(env_name, num_envs=1)
            obs = env.reset()
            done = False
            step_count = 0
            while not done and step_count < 100:
                action = env.action_space.sample().reshape(1,*env.action_space.shape)
                obs, reward, done,term, info = env.step(action)
            
                step_count += 1
            self.assertEqual(obs.size, np.prod(env.observation_space.shape))
            del env
            print(f"Environment {env_name} passed the test.")   

if __name__ == "__main__":
    unittest.main()
