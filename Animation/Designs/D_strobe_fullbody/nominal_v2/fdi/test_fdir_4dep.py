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
from fdir_env import ACTIONS, ACT_COST, CH, FDI_RESET_ACTS, BASE_OBS_DIM, FDIRParallelEnv
import mission_profile as MP


# =================== CONFIG - edit these, then press Run (no command-line args) ===================
PROFILE_FILE   = "profiles/apophis_test4.json"     # 4-deputy mission profile
COMPARE_MODELS = ["fdir_brain_hpc.zip",            # A/B these brains on the SAME mission
                  "fdir_brain_hpc_hc.zip",         #   (set to a single entry to run just one)
                  "fdir_brain_hpc_hc2.zip"]
HORIZON        = 150                               # steps per episode
SEED           = 7
EPISODES       = 20                                # ACT 2: random campaigns per brain (same seed = fair A/B)
SHOW_PLOT      = True                              # write a showcase figure per brain
# ACT 1 scripted campaign: list of (deputy, fault_class, onset_step). Deputies may be REUSED across
# waves (a deputy takes a new fault after recovering the previous). Default = all SIX classes in 2 waves:
CAMPAIGN       = [(0, "NAV", 10), (1, "COMM", 28), (2, "ATT", 46), (3, "PROP", 64),
                  (0, "FORM", 82), (1, "PEER", 100)]
# Examples:  two faults on one deputy -> [(0,"PEER",20),(0,"FORM",70)]
#            single fault             -> [(2,"ATT",30)]
# =================================================================================================

_RECOVER = {"NAV": "REINIT", "COMM": "RELOAD", "FORM": "MODE",
            "ATT": "SAFEHOLD", "PROP": "SAFEHOLD", "PEER": "ISOLATE"}
_LABEL = {"fdir_brain_hpc.zip": "base", "fdir_brain_hpc_hc.zip": "hc", "fdir_brain_hpc_hc2.zip": "hc2"}


def _short(model_file):
    return _LABEL.get(os.path.basename(model_file), os.path.splitext(os.path.basename(model_file))[0])


def heuristic(obs):
    if obs.shape[0] > BASE_OBS_DIM:
        state01 = float(obs[BASE_OBS_DIM + len(CH)])
        onehot = np.asarray(obs[BASE_OBS_DIM + len(CH) + 1: BASE_OBS_DIM + 2 * len(CH) + 1])
        if state01 >= 1.0 and onehot.max() > 0:
            return ACTIONS.index(_RECOVER[CH[int(np.argmax(onehot))]])
        return ACTIONS.index("NOOP")
    sev = obs[:len(CH)]; ci = int(np.argmax(sev))
    return ACTIONS.index("NOOP") if sev[ci] < 1.5 else ACTIONS.index(_RECOVER[CH[ci]])


def load_policy(path):
    try:
        from stable_baselines3 import PPO
        m = PPO.load(path)
        return (lambda o: int(m.predict(o, deterministic=True)[0])), f"SB3 trained brain ({os.path.basename(path)})", True
    except Exception as e:
        return (lambda o: heuristic(o)), f"HEURISTIC fallback ({type(e).__name__})", False


def fdi_state(env, i):
    st = env._fdi.cores[i].state
    return st.state.name, (st.fault_type.name if st.fault_type.name in CH else None)


def step_instrumented(env, policy, obs):
    b = env.backend; s = env._step; rec = {}
    for i, a in enumerate(env.agents):
        act = ACTIONS[int(policy(obs[a]))]
        unsafe = ACT_COST[act] > b.auth[i] + 1e-9
        exec_act = "NOOP" if unsafe else act
        was_active = b.active(i, s)
        fcls = b.fault[i] if was_active else None
        state_name, iso = fdi_state(env, i)
        correct = b.apply(i, exec_act, s)
        if exec_act in FDI_RESET_ACTS:
            env._fdi.command_reset(i, float(s + 1))
        rec[i] = dict(prop=act, exec=exec_act, unsafe=unsafe, active=was_active,
                      fcls=fcls, correct=correct, fdi=state_name, iso=iso)
    env._step = s + 1
    env._sev = [b.residuals(i, env._step, env._rng) for i in range(b.n)]
    for i in range(b.n):
        env._fdi.step(i, float(env._step), env._sev[i])
    obs = {a: env._obs_i(i) for i, a in enumerate(env.agents)}
    return obs, rec, b.value()


def _clear_all(env):
    b = env.backend
    for i in range(b.n):
        b.fault[i] = "NONE"; b.onset[i] = 10 ** 9; b.spec.pop(i, None); b.corrupt.pop(i, None)
    env._sev = [b.residuals(i, 0, env._rng) for i in range(b.n)]
    env._fdi.reset()
    for i in range(b.n):
        env._fdi.step(i, 0.0, env._sev[i])
    return {a: env._obs_i(i) for i, a in enumerate(env.agents)}


def _apply_event(env, dep, cls, t):
    b = env.backend
    b.fault[dep] = cls; b.onset[dep] = int(t); b.spec.pop(dep, None)
    if cls == "PEER":
        nb = [j for j in range(b.n) if j != dep]; b.corrupt[dep] = set(nb[:2])


def showcase(profile, policy, model_file, seed, horizon, campaign, make_plot):
    n = profile.fleet.n_deputies
    env = FDIRParallelEnv(n_deputies=n, horizon=horizon, seed=seed); env.reset(seed=seed)
    obs = _clear_all(env)
    sched = {}
    for dep, cls, onset in campaign:
        sched.setdefault(int(onset), []).append((dep, cls))
    inj_classes = set(c for _, c, _ in campaign)

    print("=" * 78)
    print(f"ACT 1 - SHOWCASE  ·  brain: {_short(model_file)} ({model_file})  ·  {n} deputies")
    print("campaign: " + ", ".join(f"D{d}:{c}@t{o}" for d, c, o in campaign))
    print("-" * 78)

    v0 = env.backend.value(); vseries = [v0]
    timeline = [[fdi_state(env, i)[0] for i in range(n)]]
    prev = {i: "HEALTHY" for i in range(n)}; recov_steps = []; advis = 0; unsafe_ct = 0
    recovered = set()
    for t in range(horizon):
        for (dep, cls) in sched.get(t, []):
            _apply_event(env, dep, cls, t)
            print(f"{t:>3}  D{dep:<5}  -- fault injected: {cls} --")
        obs, rec, v = step_instrumented(env, policy, obs)
        vseries.append(v); timeline.append([rec[i]["fdi"] for i in range(n)])
        for i in range(n):
            r = rec[i]
            if r["unsafe"]:
                unsafe_ct += 1
                print(f"{t:>3}  D{i:<5}  RTA VETO: proposed {r['prop']} -> ran NOOP (safe)")
            if r["fdi"] != prev[i]:
                iso = f" · isolated={r['iso']}" if r["iso"] else ""
                print(f"{t:>3}  D{i:<5}  FDI {prev[i]} -> {r['fdi']}{iso}")
                prev[i] = r["fdi"]
            if r["exec"] != "NOOP":
                if r["active"] or r["correct"]:
                    tag = "RECOVERED" if r["correct"] else "commanded"
                    print(f"{t:>3}  D{i:<5}  policy -> {r['exec']}  [{tag}]  (fault={r['fcls']}, value={v:.2f})")
                    if r["correct"]:
                        recov_steps.append((t, i)); recovered.add(r["fcls"])
                else:
                    advis += 1
    v1 = env.backend.value()
    got = sorted(inj_classes & recovered); miss = sorted(inj_classes - recovered)
    print("-" * 78)
    print(f"mission value: {v0:.2f} -> {v1:.2f}   ·  recovered: {','.join(got) or '-'}   ·  missed: {','.join(miss) or '-'}")
    print(f"unsafe proposals: {unsafe_ct} (0 executed)  ·  healthy-deputy advisories: {advis}")
    if make_plot:
        outname = "fdir_test4_showcase.png" if len(COMPARE_MODELS) == 1 else f"fdir_test4_showcase__{_short(model_file)}.png"
        _plot_showcase(profile, np.array(vseries), timeline, campaign, recov_steps, model_file,
                       os.path.join(_HERE, outname))
    return set(got)


def _plot_showcase(profile, vseries, timeline, campaign, recov_steps, model_file, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"figure.dpi": 220, "savefig.dpi": 220, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans",
                         "axes.edgecolor": "#222", "axes.linewidth": 1.1, "axes.grid": False})
    n = profile.fleet.n_deputies; T = len(vseries)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.6), height_ratios=[1.05, 1.0], sharex=True)
    ax1.plot(np.arange(T), vseries, color="#3B78B5", lw=2.2, zorder=5)
    for d, c, o in campaign:
        ax1.axvline(o, color="#C0504D", ls=(0, (4, 3)), lw=1.0, alpha=0.8)
        ax1.text(o, ax1.get_ylim()[1], f"D{d}:{c}", rotation=90, va="top", ha="right", fontsize=7, color="#C0504D")
    for (t, i) in recov_steps:
        ax1.scatter([t], [vseries[t]], color="#4CA37A", s=40, zorder=6, ec="white", lw=0.6)
    ax1.set_ylabel("mission value\n(fleet coverage)")
    ax1.set_title(f"4-Deputy FDIR Test — {profile.name}  ·  brain: {_short(model_file)}", loc="left")
    order = ["HEALTHY", "SUSPECT", "FAULTED", "DEAD"]; cmap = {s: k for k, s in enumerate(order)}
    grid = np.array([[cmap.get(timeline[t][i], 0) for t in range(T)] for i in range(n)])
    cols = ["#4CA37A", "#E39A3C", "#C0504D", "#7f7f7f"]
    ax2.imshow(grid, aspect="auto", cmap=ListedColormap(cols), vmin=0, vmax=3,
               extent=[0, T, n - 0.5, -0.5], interpolation="nearest")
    ax2.set_yticks(range(n)); ax2.set_yticklabels([f"D{i}" for i in range(n)])
    for (t, i) in recov_steps:
        ax2.scatter([t], [i], marker="*", color="white", s=55, zorder=6, ec="#4CA37A", lw=0.6)
    ax2.set_xlabel("mission step"); ax2.set_ylabel("deputy")
    ax2.set_title("Fleet FDI health   (star = RL recovery)", loc="left", fontsize=11)
    ax2.legend(handles=[Patch(color=cols[k], label=order[k]) for k in range(4)],
               loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=9)
    fig.text(0.006, 0.005, "ARISE CORTEX · Design D · Apophis · 4-deputy test · surrogate + flight-FDI in the loop",
             fontsize=7.2, color="#9a9a9a")
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(out); plt.close(fig)
    print(f"[figure] {out}")


def evaluate(profile, policy, episodes, seed, horizon):
    n = profile.fleet.n_deputies
    env = FDIRParallelEnv(n_deputies=n, horizon=horizon, seed=seed)
    rec_ok = {c: 0 for c in CH}; rec_tot = {c: 0 for c in CH}
    iso_ok = iso_tot = 0; unsafe = 0; advis = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        truth = {i: env.backend.fault[i] for i in range(n)}
        onset = {i: env.backend.onset[i] for i in range(n)}
        cleared = set()
        for _ in range(horizon):
            obs, rec, _ = step_instrumented(env, policy, obs)
            for i in range(n):
                r = rec[i]
                if r["unsafe"]:
                    unsafe += 1
                if not r["active"] and r["exec"] != "NOOP":
                    advis += 1
                if r["iso"] and truth[i] in CH and env._step > onset[i]:
                    iso_tot += 1; iso_ok += int(r["iso"] == truth[i])
                if r["correct"] and truth[i] in CH and i not in cleared:
                    rec_ok[truth[i]] += 1; cleared.add(i)
        for i in range(n):
            if truth[i] in CH:
                rec_tot[truth[i]] += 1
    return dict(rec_ok=rec_ok, rec_tot=rec_tot, iso=(iso_ok / iso_tot if iso_tot else float("nan")),
                unsafe=unsafe, advis=advis)


def _print_ab(models, results, injected):
    labels = [_short(m) for m in models]
    print("=" * 78)
    print(f"ACT 2 - A/B EVAL  ·  {EPISODES} random campaigns per brain (same seed = fair comparison)")
    print("=" * 78)
    hdr = f"{'class':6}  " + "  ".join(f"{l:>6}" for l in labels) + f"   {'injected':>8}"
    print(hdr); print("-" * len(hdr))
    for c in CH:
        rates = []
        for m in models:
            R = results[m]; rates.append(R["rec_ok"][c] / R["rec_tot"][c] if R["rec_tot"][c] else float("nan"))
        star = "  <- fixed" if (len(models) > 1 and rates[0] < 0.4 and max(rates) > 0.7) else ""
        print(f"{c:6}  " + "  ".join(f"{r:>6.2f}" for r in rates) + f"   {injected.get(c, 0):>8}" + star)
    print("-" * len(hdr))
    print(f"{'iso':6}  " + "  ".join(f"{results[m]['iso']:>6.2f}" for m in models) + "   (isolation accuracy)")
    print(f"{'unsafe':6}  " + "  ".join(f"{results[m]['unsafe']:>6d}" for m in models) + "   (proposed; executed: " + " / ".join("0" for _ in models) + ")")
    print(f"{'advis':6}  " + "  ".join(f"{results[m]['advis']:>6d}" for m in models) + "   (non-NOOP on healthy deputies)")


def main():
    profile = MP.load_profile(os.path.join(_HERE, PROFILE_FILE))
    n = profile.fleet.n_deputies
    models = list(COMPARE_MODELS) if isinstance(COMPARE_MODELS, (list, tuple)) else [COMPARE_MODELS]
    print("=" * 78)
    print("ARISE CORTEX - RL FDIR test bench")
    print(f"mission : {profile.name}  ·  {profile.target.name}  ·  {n} deputies  ·  obj {profile.objective.name}")
    print(f"fidelity: surrogate + flight-FDI in the loop (obs {fdir_env.OBS_DIM}, act {len(ACTIONS)})")

    pol = {}
    for m in models:
        policy, tag, ok = load_policy(os.path.join(_HERE, m))
        pol[m] = policy
        print(f"brain   : {_short(m):5} <- {tag}")

    show = {}
    for m in models:
        print()
        show[m] = showcase(profile, pol[m], m, SEED, HORIZON, CAMPAIGN, SHOW_PLOT)
    if len(models) > 1:
        allc = sorted(set(c for _, c, _ in CAMPAIGN))
        print("\n" + "=" * 78)
        print("ACT 1 recap - scripted-campaign recovery")
        print("=" * 78)
        print(f"{'brain':6}  " + "  ".join(f"{c:>5}" for c in allc))
        for m in models:
            print(f"{_short(m):6}  " + "  ".join(f"{('ok' if c in show[m] else '.'):>5}" for c in allc))

    print()
    res = {}; injected = None
    for m in models:
        res[m] = evaluate(profile, pol[m], EPISODES, SEED, HORIZON)
        if injected is None:
            injected = res[m]["rec_tot"]
    _print_ab(models, res, injected)


if __name__ == "__main__":
    main()
