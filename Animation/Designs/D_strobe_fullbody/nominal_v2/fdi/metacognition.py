#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metacognition.py  (Design D / nominal_v2 / fdi — CORTEX-E phase E1)
==================================================================

Metacognitive self-FDIR: the agent puts ITSELF in the fault model.  It estimates
whether its own current belief is trustworthy and ABSTAINS to the certified VSC
(``rta.RTAReason.AC_INCOMPETENT``) when it is out of its depth — with a conformally
calibrated false-abstention rate.

Competence from three interpretable self-signals in [0,1]:
  conflict  disagreement among INDEPENDENT sources (FDI / vision / ISL / causal):
            one coherent fault -> 0; sources split -> ->1.
  mismatch  the leading hypothesis fails to explain the anomalies: a single fault
            trips ~1-2 channels; many channels firing incoherently -> ->1.
  ood       perception out-of-distribution (advisory, from perception.py or a
            conformal novelty score).
  competence = (1-conflict)*(1-mismatch)*(1-ood).

The abstain threshold is conformally calibrated on HEALTHY competence so
``P(abstain | healthy) <= alpha``.  A CLEAR, coherent fault yields HIGH competence
(the agent acts/recovers); abstention fires only on conflict / novelty / OOD — so
metacognition STRENGTHENS the simplex (the AC can recuse itself, not just be
overruled).

Wiring: the executive forwards ``not report.abstain`` to
``rta.RTAExecutive.decide(..., ac_competent=...)`` (E1 adds the AC_INCOMPETENT
verdict).  Pure numpy.  Run: ``python metacognition.py`` (self-test, 6/6).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np


def _conformal_lower_threshold(calib, alpha):
    """Largest h with ``P(x < h) <= alpha`` (finite-sample, marginal): the
    floor(alpha*(n+1))-th order statistic.  -inf if too few samples (never abstain)."""
    x = np.sort(np.asarray(calib, float))
    n = x.size
    if n == 0:
        return -np.inf
    k = int(np.floor(alpha * (n + 1)))
    return -np.inf if k < 1 else float(x[k - 1])


def conflict_score(labels):
    """labels: fault-hypothesis labels from independent sources (None = abstain).
    0 if all agree; ->1 as they split."""
    ls = [l for l in labels if l is not None]
    if len(ls) <= 1:
        return 0.0
    return 1.0 - Counter(ls).most_common(1)[0][1] / len(ls)


def mismatch_score(n_anomalous, n_expected=2, n_channels=6):
    """How incoherent the anomaly pattern is: a single fault trips ~<=n_expected
    channels; more simultaneous independent channels -> mismatch ->1."""
    return float(np.clip((n_anomalous - n_expected) / max(n_channels - n_expected, 1), 0.0, 1.0))


@dataclass
class CompetenceReport:
    competence: float
    conflict: float
    mismatch: float
    ood: float
    abstain: bool
    reason: str


@dataclass
class CompetenceMonitor:
    """Calibrated metacognitive monitor.  ``fit`` on healthy competence samples sets
    the abstention threshold; ``assess`` returns a CompetenceReport whose
    ``abstain`` the executive forwards to ``rta.decide(ac_competent=not abstain)``."""
    alpha: float = 0.02            # target false-abstention rate on healthy
    n_expected: int = 2
    n_channels: int = 6
    _thr: float = -np.inf

    def competence(self, source_labels, n_anomalous, ood=0.0):
        cf = conflict_score(source_labels)
        mm = mismatch_score(n_anomalous, self.n_expected, self.n_channels)
        ood = float(np.clip(ood, 0.0, 1.0))
        return (1.0 - cf) * (1.0 - mm) * (1.0 - ood), cf, mm, ood

    def fit(self, healthy_samples):
        """healthy_samples: iterable of (source_labels, n_anomalous, ood) seen under
        healthy/nominal operation.  Calibrates so P(competence < thr | healthy) <= alpha."""
        cs = [self.competence(*s)[0] for s in healthy_samples]
        self._thr = _conformal_lower_threshold(cs, self.alpha)
        return self

    def assess(self, source_labels, n_anomalous, ood=0.0) -> CompetenceReport:
        c, cf, mm, ood = self.competence(source_labels, n_anomalous, ood)
        abstain = c < self._thr
        reason = (max((("conflict", cf), ("mismatch", mm), ("ood", ood)),
                      key=lambda kv: kv[1])[0] if abstain else "competent")
        return CompetenceReport(c, cf, mm, ood, abstain, reason)


# ================================ self-test =================================
def _healthy_sample(rng):
    # mostly all-agree NONE, occasional transient disagreement / lone anomaly / OOD noise
    labels = ["NONE", "NONE", "NONE", "NONE"]
    if rng.random() < 0.10:
        labels[rng.integers(4)] = rng.choice(["NAV", "ATT", "PROP"])
    n_anom = int(rng.random() < 0.12) + int(rng.random() < 0.03)
    ood = float(rng.uniform(0.0, 0.18))
    return labels, n_anom, ood


def _selftest():
    rng = np.random.default_rng(7)
    mon = CompetenceMonitor(alpha=0.02).fit([_healthy_sample(rng) for _ in range(4000)])
    passed = total = 0

    def check(name, ok, d=""):
        nonlocal passed, total
        total += 1
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   {d}" if d else ""))

    r = mon.assess(["NONE"] * 4, 0, 0.05)
    check("T1 healthy -> competent", (not r.abstain) and r.competence > mon._thr,
          f"c={r.competence:.3f} thr={mon._thr:.3f}")

    r = mon.assess(["NAV", "NAV", "NAV", "NAV"], 1, 0.05)
    check("T2 clear fault -> act (no abstain)", not r.abstain, f"c={r.competence:.3f} reason={r.reason}")

    r = mon.assess(["NAV", "ATT", "PROP", "NONE"], 2, 0.05)
    check("T3 conflicting sources -> abstain(conflict)", r.abstain and r.reason == "conflict",
          f"c={r.competence:.3f}")

    r = mon.assess(["NONE", "NONE", "NONE", "NONE"], 5, 0.05)
    check("T4 incoherent pattern -> abstain(mismatch)", r.abstain and r.reason == "mismatch",
          f"c={r.competence:.3f}")

    r = mon.assess(["NAV", "NAV", "NAV", "NAV"], 1, 0.90)
    check("T5 OOD input -> abstain(ood)", r.abstain and r.reason == "ood", f"c={r.competence:.3f}")

    fa = np.mean([mon.assess(*_healthy_sample(rng)).abstain for _ in range(20000)])
    check("T6 false-abstain rate <= alpha (calibrated)", fa <= mon.alpha + 0.008,
          f"rate={fa*100:.2f}% (target<= {mon.alpha*100:.0f}%)")

    print("-" * 60)
    print(f"E1 metacognition self-test: {passed}/{total} PASS")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
