# Copyright 2021 Garena Online Private Limited
# Copyright (c) 2025-2026 Amal Dev Haridevan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gym
import gzdrl.envs as envs
from abc import ABC , abstractmethod
import numpy as np
import gymnasium as gym
import gymnasium
from typing import Optional
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvIndices,
  VecEnvObs,
  VecEnvStepReturn,
)
from packaging import version
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
from gzdrl.envs.python.protocol import EnvPool
import typing
from numpy.typing import NDArray

class VecAdapter(VecEnvWrapper):
  """
  Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

  """

  def __init__(self, venv: EnvPool)->None:
    """VecAdapter converts custom vectorized environments such as Gazebo-Envpool
    and AsyncDRLServer API based envs to be compatible with stbale_baselines3 VecEnv

    Parameters
    ----------
    venv : EnvPool
        vectorized env, see python_envs.vectorized_hover_env or Gazebo-Envpool envs
    """
    # Retrieve the number of environments from the config
    venv.num_envs = venv.spec.config.num_envs
    super().__init__(venv=venv)
  
  
  def step_async(self, actions: NDArray) -> None:
    """Performs async step (immediately returns)

    Parameters
    ----------
    actions : NDArray
        actions for all environments
    """
    self.actions = actions

  def reset(self) -> VecEnvObs:
    """Resets the environments

    Returns
    -------
    VecEnvObs
        Returns the vecenv obs
    """
    if is_legacy_gym:
      return self.venv.reset()
    else:
      return self.venv.reset()[0]

  def seed(self, seed: Optional[int] = None) -> None:
    """Seeding

    Parameters
    ----------
    seed : Optional[int], optional
        ignored, non functional, kept for API compatibility, by default None
    """
    # You can only seed EnvPool env by calling envpool.make()
    pass

  def step_wait(self) -> VecEnvStepReturn:
    """Steps and waits for all environments to return.
    If any env is done, then that env is automatically reset.

    Returns
    -------
    VecEnvStepReturn
        Result of step
    """
    if is_legacy_gym:
      obs, rewards, dones, info_dict = self.venv.step(self.actions)
    else:
      obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
      dones = np.logical_or(terms, truncs)
    infos = []
    # Convert dict to list of dict
    # and add terminal observation
    for i in range(self.num_envs):
      infos.append(
        {
          key: info_dict[key][i]
          for key in info_dict.keys()
          if isinstance(info_dict[key], np.ndarray)
        }
      )
      if not is_legacy_gym:
        # Stable-Baselines3 bootstraps the terminal observation at a time
        # limit only when this conventional flag is present.
        infos[i]["TimeLimit.truncated"] = bool(truncs[i] and not terms[i])
      if dones[i]:
        infos[i]["terminal_observation"] = obs[i]
        if is_legacy_gym:
          obs[i] = self.venv.reset(np.array([i]))
        else:
          obs[i] = self.venv.reset(np.array([i]))[0]
    return obs, rewards, dones, infos


class VecAdapterImageEnv(VecAdapter):
  """
  Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
  This is specialized for image obs, the obs will be resized to image dims

  :param venv: The envpool object.
  """

  def __init__(self, venv: EnvPool, image_shape: np.ndarray):
    # Retrieve the number of environments from the config
    venv.num_envs = venv.spec.config.num_envs
    self.image_shape = image_shape
    super().__init__(venv=venv)
    self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)


  def reset(self) -> VecEnvObs:
    """Resets the env, key difference from VecAdapter is that we reshape the obs to image shape

    Returns
    -------
    VecEnvObs
        obs with image
    """
    if is_legacy_gym:
      return self.venv.reset().reshape(self.venv.num_envs, *self.image_shape)
    else:
      return self.venv.reset()[0].reshape(self.venv.num_envs, *self.image_shape)

  def step_wait(self) -> VecEnvStepReturn:
    """Steps and waits for return. THe observaton is reshaped to image shape

    Returns
    -------
    VecEnvStepReturn
        Observation
    """
    if is_legacy_gym:
      obs, rewards, dones, info_dict = self.venv.step(self.actions)
    else:
      obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
      dones = np.logical_or(terms, truncs)
    obs = obs.reshape(self.venv.num_envs, *self.image_shape)
    infos = []
    # Convert dict to list of dict
    # and add terminal observation
    for i in range(self.num_envs):
      infos.append(
        {
          key: info_dict[key][i]
          for key in info_dict.keys()
          if isinstance(info_dict[key], np.ndarray)
        }
      )
      if not is_legacy_gym:
        infos[i]["TimeLimit.truncated"] = bool(truncs[i] and not terms[i])
      if dones[i]:
        infos[i]["terminal_observation"] = obs[i]
        if is_legacy_gym:
          obs[i] = self.venv.reset(np.array([i])).reshape(*self.image_shape)
        else:
          obs[i] = self.venv.reset(np.array([i]))[0].reshape(*self.image_shape)
    return obs, rewards, dones, infos

class AsyncDRLServerVecAdapter(VecEnv):
  """Expose a batched ``AsyncDRLServerPool`` environment as an SB3 VecEnv.

  Unlike :class:`VecAdapter`, this adapter wraps one Python object that owns a
  native batch rather than an EnvPool object.  The wrapped environment must
  expose ``batch``, batched ``reset``/``step`` methods, and Gymnasium spaces.
  """

  def __init__(self, env: gymnasium.Env) -> None:
    self.env = env
    self.actions: Optional[NDArray] = None
    super().__init__(
      num_envs=int(env.batch),
      observation_space=env.observation_space,
      action_space=env.action_space,
    )

  def step_async(self, actions: NDArray) -> None:
    self.actions = actions

  def step_wait(self) -> VecEnvStepReturn:
    if self.actions is None:
      raise RuntimeError("step_async() must be called before step_wait().")

    obs, rewards, terminated, truncated, info_dict = self.env.step(self.actions)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.broadcast_to(
      np.asarray(truncated, dtype=bool), terminated.shape
    )
    dones = np.logical_or(terminated, truncated)
    rewards = np.asarray(rewards, dtype=np.float32)

    infos = []
    for i in range(self.num_envs):
      info = {
        key: value[i]
        for key, value in info_dict.items()
        if isinstance(value, np.ndarray) and len(value) == self.num_envs
      }
      info["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
      if dones[i]:
        info["terminal_observation"] = obs[i].copy()
        reset_obs, reset_info = self.env.reset(np.array([i], dtype=np.int32))
        obs[i] = reset_obs[0]
        self.reset_infos[i] = reset_info
      infos.append(info)

    self.actions = None
    return obs, rewards, dones, infos

  def reset(self) -> VecEnvObs:
    obs, info = self.env.reset()
    self.reset_infos = [copy.deepcopy(info) for _ in range(self.num_envs)]
    return obs

  def close(self) -> None:
    close = getattr(self.env, "close", None)
    if callable(close):
      close()

  def get_attr(
    self, attr_name: str, indices: VecEnvIndices = None
  ) -> list[typing.Any]:
    indices_ = self._get_indices(indices)
    value = getattr(self.env, attr_name)
    return [value for _ in indices_]

  def set_attr(
    self, attr_name: str, value: typing.Any, indices: VecEnvIndices = None
  ) -> None:
    # A native batch has one shared Python owner, so the attribute is shared
    # by every selected logical environment.
    if self._get_indices(indices):
      setattr(self.env, attr_name, value)

  def env_method(
    self,
    method_name: str,
    *method_args: typing.Any,
    indices: VecEnvIndices = None,
    **method_kwargs: typing.Any,
  ) -> list[typing.Any]:
    indices_ = self._get_indices(indices)
    if not indices_:
      return []
    result = getattr(self.env, method_name)(*method_args, **method_kwargs)
    return [result for _ in indices_]

  def env_is_wrapped(
    self, wrapper_class: type[gymnasium.Wrapper], indices: VecEnvIndices = None
  ) -> list[bool]:
    return [False for _ in self._get_indices(indices)]
