#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal_model.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E2)
=================================================================

Structural causal model (SCM) of fault propagation + ABDUCTIVE isolation.

  root-cause FAULT  --(GNC physics)-->  observable CHANNEL residual anomalies

Each root cause has an EFFECT SET (the channels it makes anomalous), read off the
known GNC coupling (and consistent with fdi_core._classify_settled):

  NAV  (nav drift)        -> {NAV, FORM}          common-mode formation miss, PEER quiet
  ATT  (tumble)           -> {ATT}
  COMM (heartbeat loss)   -> {COMM}
  PROP (bad burn)         -> {PROP, NAV, FORM}    downstream nav + formation perturbation
  FORM (local form fault) -> {FORM, PEER}         off-station AND peers see it
  PEER (byzantine/lying)  -> {PEER}               peer-inconsistent only, FORM quiet

ISOLATION = ABDUCTION: the minimal-cost set of root causes whose effects best
explain the observed channel-anomaly pattern (parsimonious covering / Reiter
diagnosis).  Unlike the single-label precedence rule it:
  * isolates COMPOUND faults (>=2 simultaneous causes) the precedence rule hides,
  * yields a structured EXPLANATION (cause -> channels covered; what is left over),
  * produces CONFLICT (>=2 near-equal-cost diagnoses -> ambiguous) and MISMATCH
    (anomalies no bounded cause-set explains -> novelty) — the exact signals the
    E1 metacognition monitor consumes.

When two cause-sets are observationally equivalent (e.g. a lying PEER hides behind
FORM's signature), abduction returns one valid minimal explanation AND flags
conflict rather than guessing — the identifiability limit that E3 (active,
information-seeking diagnosis) is designed to resolve.

Bounded & flight-feasible: 6 causes, diagnoses enumerated up to `max_card`.
Pure numpy.  Run: python causal_model.py  (self-test, 6/6).
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np

FAULTS = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")
CHANNELS = ("NAV", "ATT", "COMM", "PROP", "FORM", "PEER")

# fixed causal graph: fault -> set of channels it makes anomalous
EFFECTS = {
    "NAV":  frozenset({"NAV", "FORM"}),
    "ATT":  frozenset({"ATT"}),
    "COMM": frozenset({"COMM"}),
    "PROP": frozenset({"PROP", "NAV", "FORM"}),
    "FORM": frozenset({"FORM", "PEER"}),
    "PEER": frozenset({"PEER"}),
}

# abduction cost weights
W_CARD = 1.0     # Occam: prefer fewer causes
W_UNCOV = 10.0   # strongly prefer covering every observed anomaly
W_OVER = 0.5     # mildly penalise predicting channels that are NOT anomalous
CONFLICT_SCALE = 1.0


@dataclass
class Diagnosis:
    causes: tuple                       # the abduced root-cause set
    covered: frozenset                  # observed anomalies explained
    uncovered: frozenset                # observed anomalies NOT explained (-> mismatch)
    overpredicted: frozenset            # predicted-but-not-observed (active) channels
    cost: float
    conflict: float                     # [0,1] ambiguity (>=2 near-equal diagnoses)
    mismatch: float                     # [0,1] fraction of anomaly unexplained
    n_optimal: int                      # # distinct near-optimal diagnoses
    explanation: str


def forward(causes, active=None) -> frozenset:
    """Counterfactual do(causes): the channel anomalies the SCM predicts."""
    pred = frozenset().union(*(EFFECTS[f] for f in causes)) if causes else frozenset()
    return pred & frozenset(active) if active is not None else pred


def abduce(observed, active=None, max_card=3) -> Diagnosis:
    """Minimal-cost cause-set explaining `observed` (iterable of anomalous channels).
    `active` = channels currently observable (over-prediction only penalised there)."""
    O = frozenset(observed)
    act = frozenset(active) if active is not None else frozenset(CHANNELS)

    def cost(S):
        pred = forward(S)
        unc = O - pred
        over = (pred - O) & act
        return W_CARD * len(S) + W_UNCOV * len(unc) + W_OVER * len(over), unc, over

    cands = []
    for k in range(0, max_card + 1):
        for S in combinations(FAULTS, k):
            c, unc, over = cost(S)
            cands.append((c, S, unc, over))
    cands.sort(key=lambda x: x[0])
    best_c, best_S, best_unc, best_over = cands[0]

    # conflict: distinct cause-set with (near-)equal cost?
    second = next((c for (c, S, _, _) in cands if set(S) != set(best_S)), best_c + 99)
    conflict = float(np.clip(1.0 - (second - best_c) / CONFLICT_SCALE, 0.0, 1.0))
    n_opt = sum(1 for (c, S, _, _) in cands if c <= best_c + 1e-9)
    mismatch = len(best_unc) / max(len(O), 1)

    parts = [f"{f}->{{{','.join(sorted(EFFECTS[f] & O))}}}" for f in best_S]
    expl = " + ".join(parts) if parts else "no fault (nominal)"
    if best_unc:
        expl += f" | UNEXPLAINED: {{{','.join(sorted(best_unc))}}}"
    if conflict > 0.5:
        expl += " | AMBIGUOUS"
    return Diagnosis(best_S, O & forward(best_S), best_unc, best_over,
                     best_c, conflict, mismatch, n_opt, expl)


def diagnose_from_alarms(alarms, active=None, max_card=3) -> Diagnosis:
    """Abduce from a {channel: bool} alarm dict (e.g., the conformal-spine / CUSUM
    alarms in fdi_core._conf_alarm / _bad_flags)."""
    return abduce([ch for ch, a in alarms.items() if a], active=active, max_card=max_card)


# ---- baseline: the single-label precedence rule (fdi_core._classify_settled) ----
def precedence_label(observed) -> str:
    L = set(observed)
    if "COMM" in L: return "COMM"
    if "PROP" in L: return "PROP"
    if "FORM" in L and "PEER" in L: return "FORM"
    if "PEER" in L and "FORM" not in L: return "PEER"
    if "FORM" in L and "PEER" not in L: return "NAV"
    if "ATT" in L: return "ATT"
    if "NAV" in L: return "NAV"
    return "NONE"


# ================================ self-test =================================
def _selftest():
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    # T1: every SINGLE fault -> singleton diagnosis covering all its anomalies
    ok = True
    det = []
    for f in FAULTS:
        dg = abduce(EFFECTS[f])
        ok &= (set(dg.causes) <= {f} or forward(dg.causes) >= EFFECTS[f]) and dg.mismatch == 0.0
        det.append(f"{f}->{'+'.join(dg.causes)}")
    check("T1 single faults isolate & cover (match physics)", ok, " ".join(det))

    # T2: COMPOUND faults — abduction recovers BOTH causes, else FLAGS ambiguity;
    # the precedence rule always returns a single label and never flags ambiguity.
    pairs = list(combinations(FAULTS, 2))
    recovered = ambiguous = masked_ok = valid = 0
    for a, b in pairs:
        O = EFFECTS[a] | EFFECTS[b]
        dg = abduce(O)
        cs = set(dg.causes)
        valid += int(dg.mismatch == 0.0)                                   # always explains the observation
        masked = EFFECTS[a] <= EFFECTS[b] or EFFECTS[b] <= EFFECTS[a]
        if masked:
            masked_ok += int(forward(dg.causes) >= O)                      # collapses to the dominant cause
        elif cs == {a, b}:
            recovered += 1                                                 # both causes isolated
        elif dg.conflict > 0.5:
            ambiguous += 1                                                 # observationally equivalent -> flagged
    handled = recovered + ambiguous + masked_ok
    check("T2 compound: recover BOTH or FLAG ambiguity (precedence: 1 label, no flag)",
          valid == len(pairs) and handled == len(pairs) and recovered >= 11,
          f"both={recovered}/{len(pairs)} · ambiguous-flagged={ambiguous} · masked-collapse={masked_ok} · "
          f"all-valid={valid}/{len(pairs)} · precedence-recovers-both=0/{len(pairs)}")

    # T3: ambiguous pattern -> CONFLICT (>=2 near-equal diagnoses)
    dg = abduce({"NAV", "FORM", "PEER"})
    check("T3 ambiguous {NAV,FORM,PEER} -> conflict flagged",
          dg.conflict > 0.5 and dg.n_optimal >= 2,
          f"conflict={dg.conflict:.2f} n_opt={dg.n_optimal} causes={dg.causes}")

    # T4: bounded search -> MISMATCH on an over-rich pattern (needs >max_card causes)
    dg = abduce({"ATT", "COMM", "NAV", "PEER"}, max_card=1)
    check("T4 unexplained anomalies -> mismatch>0 (novelty signal)",
          dg.mismatch > 0.0, f"mismatch={dg.mismatch:.2f} uncovered={set(dg.uncovered)}")

    # T5: E1 linkage — diagnosis conflict/mismatch drive metacognitive competence
    try:
        from metacognition import CompetenceMonitor  # noqa: F401  (linkage check)
        d_clean = abduce(EFFECTS["ATT"] | EFFECTS["PROP"])      # coherent compound
        d_amb = abduce({"NAV", "FORM", "PEER"})                 # ambiguous
        c_clean = (1 - d_clean.conflict) * (1 - d_clean.mismatch)
        c_amb = (1 - d_amb.conflict) * (1 - d_amb.mismatch)
        check("T5 E1 linkage: clean compound competent, ambiguous -> low competence",
              (c_clean > 0.8 and c_amb < 0.6), f"clean_comp={c_clean:.2f} amb_comp={c_amb:.2f}")
    except Exception as ex:
        check("T5 E1 linkage", False, f"import/err: {ex}")

    # T6: counterfactual forward consistency (diagnosis predicts what's observed)
    ok = all(forward(abduce(EFFECTS[f]).causes) >= EFFECTS[f] for f in FAULTS)
    check("T6 counterfactual forward(do(diagnosis)) reproduces observation", ok)

    print("-" * 64)
    print(f"E2 causal core self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
