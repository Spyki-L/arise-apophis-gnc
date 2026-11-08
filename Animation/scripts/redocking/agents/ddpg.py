# -*- coding: utf-8 -*-
"""
ddpg.py - DDPG agent (port of DDPGAgent.m) WITH real learning.

The MATLAB DDPGAgent.learn() computed the critic target and then had the comment
`% (Placeholder: you would do backprop here.)` - i.e. the networks never actually
trained; only the soft target update ran.  This implements the full DDPG update
(Lillicrap et al., 2016):

    critic:  minimise  MSE( Q(s,a), r + gamma * (1-done) * Q'(s', mu'(s')) )
    actor :  maximise  Q(s, mu(s))   (gradient ascent -> minimise -Q)
    targets: Polyak soft update  theta' <- tau*theta + (1-tau)*theta'

Same hyper-parameter names/defaults as run_ddpg.m (alpha, beta, tau, layer sizes,
batch_size, gamma).  Works for the centralised 48->24 swarm policy and, unchanged,
for the 6->3 single-dock policy.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F

from .networks import ActorNetwork, CriticNetwork
from .ou_noise import OUActionNoise
from .replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, input_dims, n_actions, alpha=1e-4, beta=1e-3, gamma=0.99,
                 tau=1e-3, layer1_size=400, layer2_size=300, batch_size=256,
                 buffer_size=1_000_000, ou_theta=0.15, ou_sigma=0.20, ou_dt=1e-2,
                 device=None, use_ou=True):
        self.gamma = gamma; self.tau = tau
        self.batch_size = batch_size; self.n_actions = n_actions
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                              else "cpu"))
        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)

        self.actor = ActorNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.critic = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.target_actor = ActorNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.target_critic = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=beta)
        self.noise = OUActionNoise(np.zeros(n_actions), sigma=ou_sigma,
                                   theta=ou_theta, dt=ou_dt) if use_ou else None
        self.update_network_parameters(tau=1.0)               # hard copy at init

    # -------------------------------------------------- act
    @torch.no_grad()
    def choose_action(self, observation, add_noise=True):
        self.actor.eval()
        s = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu = self.actor(s).cpu().numpy()[0]
        if add_noise and self.noise is not None:
            mu = mu + self.noise()
        return np.clip(mu, -1.0, 1.0)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # -------------------------------------------------- learn (the real thing)
    def learn(self, external_batch=None):
        if external_batch is None:
            if len(self.memory) < self.batch_size:
                return None
            batch = self.memory.sample_buffer(self.batch_size)
        else:
            batch = external_batch
        return self._update_from_batch(*batch)

    def _update_from_batch(self, s, a, r, s2, mask):
        dev = self.device
        s = torch.as_tensor(s, device=dev); a = torch.as_tensor(a, device=dev)
        r = torch.as_tensor(r, device=dev).unsqueeze(1)
        s2 = torch.as_tensor(s2, device=dev); mask = torch.as_tensor(mask, device=dev).unsqueeze(1)

        # ---- critic update ----
        with torch.no_grad():
            a2 = self.target_actor(s2)
            q_next = self.target_critic(s2, a2)
            y = r + self.gamma * mask * q_next
        self.critic.train()
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---- actor update ----
        self.actor.train()
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ---- soft target update ----
        self.update_network_parameters()
        return float(critic_loss.item()), float(actor_loss.item())

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else tau
        for tgt, src in ((self.target_actor, self.actor),
                         (self.target_critic, self.critic)):
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * sp.data)

    # -------------------------------------------------- io
    def save_models(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(actor=self.actor.state_dict(),
                        critic=self.critic.state_dict(),
                        target_actor=self.target_actor.state_dict(),
                        target_critic=self.target_critic.state_dict()), path)

    def load_models(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["target_actor"])
        self.target_critic.load_state_dict(ck["target_critic"])
