# -*- coding: utf-8 -*-
"""
train_sb3.py - PRIMARY trainer: Stable-Baselines3 TD3 + HER on the goal-based
SingleDock env, with a curriculum callback and a flight-native exporter.

Why this is the primary path (per the re-docking plan):
  * SB3's TD3 + HerReplayBuffer is battle-tested (twin critics, target smoothing,
    delayed actor) - it does not suffer the divergence the hand-rolled DDPG did.
  * A custom features extractor feeds the policy ONLY [observation(12), desired_goal(3)]
    -> a 15-D actor that maps 1:1 onto the flight ActorNetwork, so the exported
    checkpoint drops into flight/controller.py with NO glue (12/15 -> 6 contract).
  * env.step's reward == env.compute_reward (SingleDockEnv._reward_core), so SB3's
    real and HER-relabelled transitions are CONSISTENT (the old code mixed a dense
    ~-800/ep reward and sparse 0/-1 on one critic).
  * a curriculum callback anneals difficulty (env.set_difficulty) on rolling success.

Export:  after training, export_sb3_to_flight() writes artifacts/flight_policy.pt in
the DDPGAgent checkpoint format that flight/controller.load_flight_agent expects.
"""
from __future__ import annotations
import os, sys, argparse
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_redock as C


def _make_extractor():
    import torch
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class CaptureExtractor(BaseFeaturesExtractor):
        """Feed the policy [observation, desired_goal] only (flight-native 15-D)."""
        def __init__(self, obs_space):
            d = obs_space.spaces["observation"].shape[0] + obs_space.spaces["desired_goal"].shape[0]
            super().__init__(obs_space, features_dim=d)
        def forward(self, obs):
            return torch.cat([obs["observation"], obs["desired_goal"]], dim=1)
    return CaptureExtractor


def _curriculum_callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class Curriculum(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.stage = 0; self.roll = deque(maxlen=C.CURRICULUM["window"]); self.eps = 0
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.roll.append(int(info.get("captured", False))); self.eps += 1
            cur = C.CURRICULUM
            if (self.stage < len(C.CURRICULUM_STAGES) - 1 and self.eps >= cur["min_eps"]
                    and len(self.roll) == self.roll.maxlen and np.mean(self.roll) >= cur["advance_at"]):
                self.stage += 1
                self.training_env.env_method("set_difficulty", C.CURRICULUM_STAGES[self.stage])
                if self.verbose:
                    print(f"[curriculum] -> stage {self.stage} "
                          f"'{C.CURRICULUM_STAGES[self.stage]['name']}' at ep {self.eps}", flush=True)
                self.roll.clear(); self.eps = 0
            return True
    return Curriculum


def export_sb3_to_flight(model, out_path):
    """Copy the SB3 TD3 actor into a flight DDPGAgent checkpoint (12/15->6)."""
    import torch
    from agents.networks import ActorNetwork, CriticNetwork
    mu = model.policy.actor.mu                       # Sequential(Linear,ReLU,Linear,ReLU,Linear,Tanh)
    in_dim = mu[0].in_features
    actor = ActorNetwork(in_dim, 400, 300, 6); sd = actor.state_dict()
    sd["fc1.weight"] = mu[0].weight.detach().cpu(); sd["fc1.bias"] = mu[0].bias.detach().cpu()
    sd["fc2.weight"] = mu[2].weight.detach().cpu(); sd["fc2.bias"] = mu[2].bias.detach().cpu()
    sd["mu.weight"]  = mu[4].weight.detach().cpu(); sd["mu.bias"]  = mu[4].bias.detach().cpu()
    actor.load_state_dict(sd)
    critic = CriticNetwork(in_dim, 400, 300, 6)      # flight never uses the critic; fill fresh
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(),
                    target_actor=actor.state_dict(), target_critic=critic.state_dict()), out_path)
    print("exported flight policy ->", out_path, f"({in_dim}->6)")
    return out_path


def train(kind="her", steps=2_000_000, algo="td3", out="artifacts/sb3", seed=0, curriculum=True):
    os.makedirs(out, exist_ok=True)
    import numpy as np
    from stable_baselines3 import TD3, DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    try:
        import tensorboard  # noqa
        tb = os.path.join(out, "tb")
    except Exception:
        tb = None
    Algo = {"td3": TD3, "ddpg": DDPG}[algo]
    d = C.TD3

    if kind == "her":
        from stable_baselines3 import HerReplayBuffer
        from envs.single_dock_env import SingleDockGym
        env = SingleDockGym(goal_based=True, dynamics="hcw", seed=seed)
        if curriculum:
            env.core.set_difficulty(C.CURRICULUM_STAGES[0])
        n_act = env.action_space.shape[0]
        noise = NormalActionNoise(np.zeros(n_act), d["expl_noise"] * np.ones(n_act))
        # learning_starts MUST exceed one episode (HER samples whole episodes)
        max_h = max(s["steps"] for s in C.CURRICULUM_STAGES)
        model = Algo("MultiInputPolicy", env,
                     replay_buffer_class=HerReplayBuffer,
                     replay_buffer_kwargs=dict(n_sampled_goal=d["her_k"], goal_selection_strategy="future"),
                     action_noise=noise, gamma=d["gamma"], tau=d["tau"], learning_rate=d["beta"],
                     batch_size=d["batch_size"], buffer_size=d["buffer_size"],
                     learning_starts=max_h + 500,
                     policy_kwargs=dict(features_extractor_class=_make_extractor(), net_arch=[400, 300]),
                     verbose=1, seed=seed, tensorboard_log=tb)
    else:
        from envs.single_dock_env import SingleDockGym
        env = SingleDockGym(goal_based=False, dynamics="hcw", seed=seed)
        if curriculum:
            env.core.set_difficulty(C.CURRICULUM_STAGES[0])
        n_act = env.action_space.shape[0]
        noise = NormalActionNoise(np.zeros(n_act), d["expl_noise"] * np.ones(n_act))
        model = Algo("MlpPolicy", env, action_noise=noise, gamma=d["gamma"], tau=d["tau"],
                     learning_rate=d["beta"], batch_size=d["batch_size"], buffer_size=d["buffer_size"],
                     learning_starts=5000, policy_kwargs=dict(net_arch=[400, 300]),
                     verbose=1, seed=seed, tensorboard_log=tb)

    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    cbs = [CheckpointCallback(save_freq=200_000, save_path=out, name_prefix="sb3_ckpt")]
    if curriculum:
        cbs.append(_curriculum_callback()(verbose=1))
    model.learn(total_timesteps=steps, progress_bar=False, callback=CallbackList(cbs))
    model.save(os.path.join(out, f"sb3_{algo}_{kind}"))
    if kind == "her":
        export_sb3_to_flight(model, os.path.join(out, "flight_policy.pt"))
    print("saved", os.path.join(out, f"sb3_{algo}_{kind}.zip"))
    return model


if __name__ == "__main__":
    CFG = dict(env="her", algo="td3", steps=2_000_000, out="artifacts/sb3", seed=0, curriculum=True)
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("--env", default=CFG["env"], choices=["her", "single"])
        ap.add_argument("--algo", default=CFG["algo"], choices=["td3", "ddpg"])
        ap.add_argument("--steps", type=int, default=CFG["steps"])
        ap.add_argument("--out", default=CFG["out"]); ap.add_argument("--seed", type=int, default=CFG["seed"])
        ap.add_argument("--no-curriculum", action="store_true")
        a = ap.parse_args()
        CFG.update(env=a.env, algo=a.algo, steps=a.steps, out=a.out, seed=a.seed, curriculum=not a.no_curriculum)
    train(kind=CFG["env"], steps=CFG["steps"], algo=CFG["algo"], out=CFG["out"],
          seed=CFG["seed"], curriculum=CFG["curriculum"])
