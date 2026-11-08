#!/usr/bin/env python3
"""Byzantine f-of-N tolerance sweep across fleet sizes (PEER-CBF consistency)."""
import json, sys, time
from pathlib import Path
import numpy as np
import numpy.linalg as la
sys.path.insert(0, str(Path(__file__).resolve().parent))
from peer_cbf import consistency_residual
from exp_apophis_mc import wilson_ci

def campaign(N, f_grid, n_trials=400, sigma_range=0.1, lie_m=60.0, thr=8.0, seed=0):
    rng = np.random.default_rng(seed + 17 * N)
    out = {}
    for f in f_grid:
        det = fa = n_liar = n_honest = 0
        for _ in range(n_trials):
            ang = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.normal(0, 0.05, N)
            pos = np.stack([1500*np.cos(ang), 1500*np.sin(ang), 200*np.sin(2*ang)], 1)
            rep = pos + rng.normal(0, 0.05, pos.shape)
            liars = rng.choice(N, size=f, replace=False)
            for L in liars:
                u = rng.normal(0, 1, 3); rep[L] = pos[L] + u/(la.norm(u)+1e-9)*lie_m
            mr = np.full((N, N), np.nan)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        mr[i, j] = la.norm(pos[i]-pos[j]) + rng.normal(0, sigma_range)
            res = consistency_residual(rep, mr, sigma_range)
            for d in range(N):
                if d in liars:
                    n_liar += 1; det += int(res[d] > thr)
                else:
                    n_honest += 1; fa += int(res[d] > thr)
        out[int(f)] = dict(Pd=(det/n_liar if n_liar else None),
                           Pd_ci=list(wilson_ci(det, n_liar)) if n_liar else None,
                           Pfa=fa/max(n_honest,1), Pfa_ci=list(wilson_ci(fa, n_honest)),
                           n_liar=n_liar, n_honest=n_honest)
    return out

def main():
    Ns = [int(x) for x in sys.argv[1].split(",")]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    t0 = time.time(); res = {}
    for N in Ns:
        f_grid = sorted(set(list(range(0, N//2 + 2))))
        f_grid = [f for f in f_grid if f <= N - 2]
        res[N] = campaign(N, f_grid, n_trials=n_trials)
        line = " ".join(f"f={f}:Pd={v['Pd'] if v['Pd'] is None else round(v['Pd'],3)}/Pfa={round(v['Pfa'],3)}" for f, v in res[N].items())
        print(f"N={N}: {line}", flush=True)
    out_file = Path(f"../results/byz_N{'_'.join(str(n) for n in Ns)}.json")
    out_file.write_text(json.dumps(dict(res=res, n_trials=n_trials, wall_s=time.time()-t0), indent=1))
    print(f"wrote {out_file.name} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
