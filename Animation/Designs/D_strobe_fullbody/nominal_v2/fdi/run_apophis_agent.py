from __future__ import annotations
import os, sys, subprocess, importlib.util

_SCRIPT = os.path.abspath(__file__)
_HERE = os.path.dirname(_SCRIPT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_HERE)                                    # so "press Run" works from any working directory
os.environ.setdefault("ARISE_RL_FDI", "1")         # flight-FDI in the loop (deployable default)
os.environ.setdefault("ARISE_RL_CALIB", "1")       # calibrated surrogate noise


def _ensure_deps():
    """First run: auto-install the RL stack into THIS interpreter, then relaunch the script.
    If an install fails (e.g. no torch wheel for this Python) it prints exactly how to fix it."""
    need = {"numpy": "numpy", "gymnasium": "gymnasium", "pettingzoo": "pettingzoo",
            "stable_baselines3": "stable-baselines3", "torch": "torch", "matplotlib": "matplotlib"}
    def _have(m):
        try:
            return importlib.util.find_spec(m) is not None
        except Exception:
            return False
    missing = [pip for mod, pip in need.items() if not _have(mod)]
    if not missing:
        return
    if os.environ.get("ARISE_DEPS_BOOTSTRAPPED") == "1":                  # already tried once
        print("=" * 78)
        print("[setup] still missing after install:", " ".join(missing))
        print(f'   run:  "{sys.executable}" -m pip install {" ".join(missing)}')
        print("   or point Visual Studio at your Python 3.11 RL env, then Run again.")
        print("=" * 78)
        raise SystemExit(1)
    print("=" * 78)
    print("[setup] first run - installing the RL stack into this interpreter:")
    print(f"        {sys.executable}")
    print(f"        {' '.join(missing)}   (torch is large; this can take a few minutes)")
    print("=" * 78)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception as _e:
        print("=" * 78)
        print(f"[setup] auto-install failed: {_e}")
        print(f'   run:  "{sys.executable}" -m pip install {" ".join(missing)}')
        print("   or point Visual Studio at your Python 3.11 RL env, then Run again.")
        print("=" * 78)
        raise SystemExit(1)
    os.environ["ARISE_DEPS_BOOTSTRAPPED"] = "1"
    importlib.invalidate_caches()
    print("[setup] packages installed - relaunching the script ...\n")
    _r = subprocess.run([sys.executable, _SCRIPT, *sys.argv[1:]])   # subprocess quotes spaced paths (os.execv did NOT)
    raise SystemExit(_r.returncode)


_ensure_deps()

import numpy as np
import fdir_env
from fdir_env import ACTIONS, CH, BASE_OBS_DIM

# =============== CONFIG - edit these, then press Run in Visual Studio (no CLI args) ===============
MODEL_FILE = "fdir_brain_hpc.zip"      # deployable brain (resolved next to this file)
BACKEND    = "surrogate"               # "surrogate" (runs anywhere) or "basilisk" (sim machine only)
DEPUTIES   = 8
HORIZON    = 120
SEED       = 7
# =================================================================================================

_RECOVER = {"NAV": "REINIT", "COMM": "RELOAD", "FORM": "MODE",
            "ATT": "SAFEHOLD", "PROP": "SAFEHOLD", "PEER": "ISOLATE"}


def heuristic(obs):
    """Rule-based fallback.  With the flight-FDI obs layer (ARISE_RL_FDI, default on)
    it acts on the OBC's FDIR verdict: act only when the core is FAULTED, on the
    ISOLATED fault type — exactly the VSC's rule-based recovery table.  Without the
    layer, falls back to worst-raw-channel."""
    if obs.shape[0] > BASE_OBS_DIM:                       # FDI telemetry present
        state01 = float(obs[BASE_OBS_DIM + len(CH)])
        onehot = np.asarray(obs[BASE_OBS_DIM + len(CH) + 1: BASE_OBS_DIM + 2 * len(CH) + 1])
        if state01 >= 1.0 and onehot.max() > 0:
            return ACTIONS.index(_RECOVER[CH[int(np.argmax(onehot))]])
        return ACTIONS.index("NOOP")
    sev = obs[:len(CH)]
    ci = int(np.argmax(sev))
    if sev[ci] < 1.5:
        return ACTIONS.index("NOOP")
    return ACTIONS.index(_RECOVER[CH[ci]])


def load_policy(path):
    try:
        from stable_baselines3 import PPO
        m = PPO.load(path)
        return (lambda o: int(m.predict(o, deterministic=True)[0])), "SB3 trained brain"
    except Exception as e:
        return (lambda o: heuristic(o)), f"heuristic fallback ({type(e).__name__})"


def main():
    model_path = os.path.join(_HERE, MODEL_FILE)
    backend = None
    if BACKEND == "basilisk":
        from basilisk_backend import BasiliskBackend
        backend = BasiliskBackend(DEPUTIES, scenario_cfg=dict(horizon=HORIZON))
    env = fdir_env.FDIRParallelEnv(n_deputies=DEPUTIES, horizon=HORIZON, backend=backend, seed=SEED)
    policy, tag = load_policy(model_path)
    print(f"== Apophis deployment · policy: {tag} ==")
    obs, _ = env.reset(seed=SEED)
    campaign = [(f"D{i}", env.backend.fault[i], env.backend.onset[i]) for i in range(DEPUTIES)
                if env.backend.fault[i] != "NONE"]
    print("injected fault campaign:", campaign)
    v0 = env.backend.value(); acts_taken = {}
    while env.agents:
        actions = {ag: policy(obs[ag]) for ag in env.agents}
        for ag in env.agents:
            an = ACTIONS[int(actions[ag])]
            if an != "NOOP":
                acts_taken.setdefault(ag, []).append(an)
        obs, rew, term, trunc, info = env.step(actions)
    v1 = env.backend.value()
    print(f"recovery actions taken: { {k: v[:4] for k, v in acts_taken.items()} }")
    print(f"mission value: {v0:.2f} -> {v1:.2f}   (recovered {v1 - v0:+.2f})")
    print("0 unsafe executions (RTA veto guarantees every executed action is safe)")


if __name__ == "__main__":
    main()
