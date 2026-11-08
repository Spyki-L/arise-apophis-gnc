"""PyTorch DDPG(+HER) agent - faithful port of the MATLAB RL AGENT_1 classes,
with the missing back-propagation implemented (the MATLAB learn() was a stub)."""
from .networks import ActorNetwork, CriticNetwork
from .ou_noise import OUActionNoise
from .replay_buffer import ReplayBuffer, HERReplayBuffer
from .ddpg import DDPGAgent
__all__ = ["ActorNetwork", "CriticNetwork", "OUActionNoise",
           "ReplayBuffer", "HERReplayBuffer", "DDPGAgent"]
