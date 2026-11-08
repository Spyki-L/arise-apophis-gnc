#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mission_profile.py — the mission-agnostic adapter (F2).

The autonomy core (FDI, FORM-tube, PEER-CBF, ISL scheduler, recovery) references
NOTHING mission-specific. Everything about a given mission — the target, the
science objective, the fleet hardware, the encounter geometry, the precomputed
contingency tables, the surrogate-vs-truth calibration — is injected through a
``MissionProfile`` loaded from JSON. Apophis ships as one profile; any other
swarm mission is a different profile with the same core.

Plug-in surface (the only mission-specific code a new mission must supply):
    TargetModel · ObjectiveModel · FleetModel · EncounterModel · a Surrogate callable

Dependency-free (json + dataclasses) so it is flight-portable.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Any
import json


# ── plug-in data models (generic; no Apophis/XB6 strings in the core) ────────

@dataclass
class TargetModel:
    name: str
    kind: str                 # "shape_spin" | "point"
    n_facets: int = 0
    spin_period_h: float = 0.0
    npa_tumble: bool = False


@dataclass
class ObjectiveModel:
    name: str                 # what science quantity to PROTECT
    metric: str               # surrogate label that scores it
    resolution_tiers_m: list = field(default_factory=list)
    target_value: float = 0.0


@dataclass
class FleetModel:
    n_deputies: int
    bus: str = "generic"
    adcs: str = "generic"
    prop: str = "generic"
    isl: str = "crosslink"
    camera: str = "generic"
    dv_budget_mps: float = 10.0


@dataclass
class EncounterModel:
    epoch_utc: str = ""
    v_rel_kms: float = 0.0
    ca_distance_km: float = 0.0
    form_window_s: float = 3600.0
    mean_motion_radps: float = 1.0e-4


@dataclass
class MissionProfile:
    name: str
    target: TargetModel
    objective: ObjectiveModel
    fleet: FleetModel
    encounter: EncounterModel
    contingency_table: dict = field(default_factory=dict)
    surrogate_correction: float = 1.0     # surrogate -> full-MC calibration factor
    meta: dict = field(default_factory=dict)

    # A Surrogate is any callable( fleet_geometry, profile ) -> coverage score.
    # It is attached at runtime (ground builds it; onboard uses the analytic twin),
    # so it is NOT serialized — only its label lives in objective.metric.
    surrogate: Optional[Callable[..., float]] = field(default=None, repr=False, compare=False)


# ── (de)serialization ─────────────────────────────────────────────────────────

def _build(cls, d: dict):
    """Construct a dataclass from a dict, ignoring unknown keys (forward-compat)."""
    import dataclasses as _dc
    fields = {f.name for f in _dc.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in fields})


def from_dict(d: dict) -> MissionProfile:
    return MissionProfile(
        name=d["name"],
        target=_build(TargetModel, d.get("target", {})),
        objective=_build(ObjectiveModel, d.get("objective", {})),
        fleet=_build(FleetModel, d.get("fleet", {})),
        encounter=_build(EncounterModel, d.get("encounter", {})),
        contingency_table=d.get("contingency_table", {}),
        surrogate_correction=d.get("surrogate_correction", 1.0),
        meta=d.get("meta", {}),
    )


def load_profile(path: str) -> MissionProfile:
    with open(path, "r") as f:
        return from_dict(json.load(f))


def save_profile(profile: MissionProfile, path: str) -> None:
    d = asdict(profile)
    d.pop("surrogate", None)   # runtime callable, not serializable
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ── convenience: derive a per-mission FDI config from a profile ──────────────

def fdi_config_from_profile(profile: MissionProfile):
    """
    Build a (mission-agnostic) FDIConfig sized/tuned from the profile. The core
    code is identical across missions; only these numbers come from the profile.
    Imported lazily so this module stays dependency-free if fdi_core is absent.
    """
    from fdi_core import FDIConfig
    cfg = FDIConfig(
        use_nav_channel=True, use_att_channel=True, use_comm_channel=True,
        use_form_channel=True, use_peer_channel=True,
    )
    # Apophis used nu_nav=1.5/h_nav=15 for the consistent rgazel EKF; a different
    # mission supplies its own via meta, else these principled defaults hold.
    cfg.nu_nav = float(profile.meta.get("nu_nav", 1.5))
    cfg.h_nav = float(profile.meta.get("h_nav", 15.0))
    return cfg


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("apophis", "toy_inspect"):
        p = load_profile(os.path.join(here, "profiles", f"{name}.json"))
        print(f"{name:12s}: target={p.target.name:18s} kind={p.target.kind:10s} "
              f"n_dep={p.fleet.n_deputies} objective={p.objective.name}")
