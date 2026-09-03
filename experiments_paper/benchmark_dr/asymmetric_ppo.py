# SPDX-License-Identifier: MIT
# Copyright (c) 2019 Antonin Raffin
# Modifications copyright (c) 2025-2026 Amal Dev Haridevan
# Adapted from the Stable-Baselines3 custom-policy examples.

from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from stable_baselines3 import PPO

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

""" 
We need to override the features extractor, so that it is identity
The policy and critic in the Actorcritic policy needs the untouched obs, so we can make tyhem asymmetric
"""

class NoOpFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        features_dim = int(np.prod(observation_space.shape))
        super(NoOpFeaturesExtractor, self).__init__(observation_space, features_dim)
        self.layer = nn.Sequential(nn.Flatten())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.layer(observations)
    
class AsymmetricNetwork(nn.Module):
    """
    Asymmetric network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param policy_obs_dim: dimension of the policy obs
    :param privileged_obs_dim: dimension of critic privileged info
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        policy_obs_dim: int,
        privileged_obs_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(AsymmetricNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        feature_dim = policy_obs_dim + privileged_obs_dim
        self.policy_obs_dim = policy_obs_dim
        self.privileged_obs_dim = privileged_obs_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(policy_obs_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, last_layer_dim_pi),
            nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, last_layer_dim_vf),
            nn.Tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Asymmetric obs for policy
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features[:, :self.policy_obs_dim]), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass for the policy network only.
        """
        return self.policy_net(features[:, :self.policy_obs_dim])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass for the value network only, using privileged info.
        """
        return self.value_net(features)
    

class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        policy_obs_dim: int = None,
        privileged_info_dim: int = None,
        *args,
        **kwargs,
    ):
        assert (policy_obs_dim is not None and privileged_info_dim is not None), "requires policy obs dim and privileged info dim"
        self.policy_obs_dim = policy_obs_dim
        self.privileged_info_dim = privileged_info_dim
        super(AsymmetricActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricNetwork(self.policy_obs_dim, self.privileged_info_dim)
