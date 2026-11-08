# -*- coding: utf-8 -*-
"""
td3.py - TD3 agent (Fujimoto et al. 2018).  Drop-in replacement for the vanilla
DDPGAgent that DIVERGED on this task (best avg reward -197 -> -730 by end of run).

TD3 fixes that instability with:
  * twin critics + clipped double-Q target  -> kills Q-overestimation,
  * target-policy smoothing (noise on the target action),
  * delayed actor / target updates (policy_delay).

Same 400/300 actor as DDPG, and save_models() writes the SAME checkpoint keys
(actor/critic/target_actor/target_critic) so flight/controller.load_flight_agent
loads a TD3 checkpoint with no changes.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .networks import ActorNetwork, CriticNetwork
    from .replay_buffer import ReplayBuffer
except ImportError:
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from networks import ActorNetwork, CriticNetwork
    from replay_buffer import ReplayBuffer


class TD3Agent:
    def __init__(self, input_dims, n_actions, alpha=1e-4, beta=1e-3, gamma=0.99,
                 tau=5e-3, layer1_size=400, layer2_size=300, batch_size=256,
                 buffer_size=1_000_000, policy_delay=2, target_noise=0.2,
                 noise_clip=0.5, expl_noise=0.2, device=None):
        self.gamma = gamma; self.tau = tau; self.batch_size = batch_size
        self.n_actions = n_actions; self.policy_delay = policy_delay
        self.target_noise = target_noise; self.noise_clip = noise_clip
        self.expl_noise = expl_noise; self._it = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)

        self.actor = ActorNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.target_actor = ActorNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.critic1 = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.critic2 = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.tc1 = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)
        self.tc2 = CriticNetwork(input_dims, layer1_size, layer2_size, n_actions).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.c1_opt = torch.optim.Adam(self.critic1.parameters(), lr=beta)
        self.c2_opt = torch.optim.Adam(self.critic2.parameters(), lr=beta)
        self.noise = None            # exploration handled here (Gaussian), not OU
        self._hard_copy()

    def _hard_copy(self):
        for tgt, src in ((self.target_actor, self.actor), (self.tc1, self.critic1), (self.tc2, self.critic2)):
            tgt.load_state_dict(src.state_dict())

    def set_expl_noise(self, sigma):
        self.expl_noise = float(sigma)

    @torch.no_grad()
    def choose_action(self, observation, add_noise=True):
        self.actor.eval()
        s = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu = self.actor(s).cpu().numpy()[0]
        if add_noise and self.expl_noise > 0:
            mu = mu + np.random.normal(0.0, self.expl_noise, size=self.n_actions)
        return np.clip(mu, -1.0, 1.0)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self, external_batch=None):
        if external_batch is None:
            if len(self.memory) < self.batch_size:
                return None
            batch = self.memory.sample_buffer(self.batch_size)
        else:
            batch = external_batch
        return self._update(*batch)

    def _update(self, s, a, r, s2, mask):
        dev = self.device
        s = torch.as_tensor(s, dtype=torch.float32, device=dev)
        a = torch.as_tensor(a, dtype=torch.float32, device=dev)
        r = torch.as_tensor(r, dtype=torch.float32, device=dev).unsqueeze(1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=dev)
        mask = torch.as_tensor(mask, dtype=torch.float32, device=dev).unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.target_actor(s2) + noise).clamp(-1.0, 1.0)
            q1t = self.tc1(s2, a2); q2t = self.tc2(s2, a2)
            y = r + self.gamma * mask * torch.min(q1t, q2t)

        self.critic1.train(); self.critic2.train()
        q1 = self.critic1(s, a); q2 = self.critic2(s, a)
        c1_loss = F.mse_loss(q1, y); c2_loss = F.mse_loss(q2, y)
        self.c1_opt.zero_grad(); c1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0); self.c1_opt.step()
        self.c2_opt.zero_grad(); c2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0); self.c2_opt.step()

        self._it += 1
        actor_loss_val = 0.0
        if self._it % self.policy_delay == 0:
            self.actor.train()
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.actor_opt.step()
            actor_loss_val = float(actor_loss.item())
            self._soft_update()
        return float(c1_loss.item()), actor_loss_val

    def _soft_update(self):
        for tgt, src in ((self.target_actor, self.actor), (self.tc1, self.critic1), (self.tc2, self.critic2)):
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    # ---- checkpoint IO (flight-loadable: same keys as DDPGAgent) ----
    def save_models(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(dict(actor=self.actor.state_dict(),
                        critic=self.critic1.state_dict(),          # DDPG-compatible key
                        target_actor=self.target_actor.state_dict(),
                        target_critic=self.tc1.state_dict(),
                        critic2=self.critic2.state_dict(),          # TD3 extras
                        target_critic2=self.tc2.state_dict()), path)

    def load_models(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"]); self.critic1.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["target_actor"]); self.tc1.load_state_dict(ck["target_critic"])
        if "critic2" in ck: self.critic2.load_state_dict(ck["critic2"])
        if "target_critic2" in ck: self.tc2.load_state_dict(ck["target_critic2"])
        else: self._hard_copy()


if __name__ == "__main__":
    ag = TD3Agent(12, 6)
    import numpy as np
    for _ in range(600):
        ag.remember(np.random.randn(12), np.random.uniform(-1,1,6), -1.0, np.random.randn(12), False)
    out = ag.learn(); print("TD3 learn returned:", None if out is None else tuple(round(x,3) for x in out))
    ag.save_models("/tmp/_td3.pt"); ag.load_models("/tmp/_td3.pt"); print("TD3 save/load OK")
