"""RAFTI docking model + Gymnasium re-docking environments."""
from .rafti import RaftiPorts
from .multi_chaser_env import MultiChaserRedock
__all__ = ["RaftiPorts", "MultiChaserRedock"]

def make_gym_env(**kw):
    """Lazy factory for the Gymnasium wrapper (only imports gymnasium on demand)."""
    from .multi_chaser_env import MultiChaserRedockGym
    return MultiChaserRedockGym(**kw)

def make_single_dock_env(**kw):
    from .single_dock_env import SingleDockEnv
    return SingleDockEnv(**kw)
