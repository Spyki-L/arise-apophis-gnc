# -*- coding: utf-8 -*-
"""
networks.py - Actor / Critic networks (PyTorch port of ActorNetwork.m &
CriticNetwork.m).

Architecture is preserved exactly from the MATLAB prototype:
  Actor : state -> fc1(400) -relu-> fc2(300) -relu-> mu(n_actions) -tanh-> [-1,1]
  Critic: state -> fc1(400) -relu-> fc2(300);  action -> fc(300);
          add -relu-> Q(1)
The only change is that these are now real, differentiable nn.Modules (the MATLAB
forward() had hand-rolled weights and NO backward pass), so DDPG can actually
learn.  fan-in uniform init follows Lillicrap et al. (2016) DDPG.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fanin_init(layer: nn.Linear):
    fan_in = layer.weight.data.size(1)
    lim = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight, -lim, lim)
    nn.init.uniform_(layer.bias, -lim, lim)


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        _fanin_init(self.fc1); _fanin_init(self.fc2)
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.mu(x))            # actions in [-1, 1]


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_value = nn.Linear(n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        _fanin_init(self.fc1); _fanin_init(self.fc2); _fanin_init(self.action_value)
        nn.init.uniform_(self.q.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        s = F.relu(self.fc1(state))
        s = self.fc2(s)                           # (state branch, pre-activation)
        a = self.action_value(action)             # (action branch)
        x = F.relu(s + a)                          # combine (matches MATLAB add-then-relu)
        return self.q(x)                           # scalar Q
