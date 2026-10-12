"""
Method registry.  A method = (router, splitter) + switches.  MTA-SFL is
DDQN + pseudo-count routing with LinUCB splitting, top-k, and bridge mode.
Baselines and ablations are added here later with the same interface, so they
run through the identical simulator, data, seeds and logging (R1.7, R4.4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    router: str = "ddqn"          # 'ddqn' | 'random'
    splitter: str = "linucb"      # 'linucb'
    pseudo_count: bool = True     # intrinsic motivation (Eq. 23)
    allow_bridge: bool = True     # topological bridge p2 == p3
    topk: bool = True             # top-k sparsification (False -> dense payloads)
    kind: str = "mtasfl"          # simulator class: 'mtasfl' | 'adaptsfl' | 'sflsc3' | 'shsfl'


METHODS = {
    "mtasfl": MethodSpec("mtasfl"),
    # state-of-the-art baselines (run in the identical environment, including the
    # same top-k compression of activations/gradients; topk=False -> dense)
    "adaptsfl": MethodSpec("adaptsfl", router="best_path", splitter="adaptsfl_bcd",
                           pseudo_count=False, allow_bridge=False, topk=True, kind="adaptsfl"),
    "sflsc3": MethodSpec("sflsc3", router="best_path", splitter="sflsc3_bcd",
                         pseudo_count=False, allow_bridge=False, topk=True, kind="sflsc3"),
    "shsfl": MethodSpec("shsfl", router="best_path", splitter="fixed_cut1",
                        pseudo_count=False, allow_bridge=False, topk=True, kind="shsfl"),
    # ablation variants are registered later (CIFAR-10 only) via the switches above.
}

SOTA = ["adaptsfl", "sflsc3", "shsfl"]
ALL_METHODS = ["mtasfl"] + SOTA


def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'. available: {list(METHODS)}")
    return METHODS[name]


def simulation_class(spec: MethodSpec):
    """Simulator implementing a method (all share the same environment)."""
    if spec.kind == "mtasfl":
        from .sim.runner import Simulation
        return Simulation
    if spec.kind == "adaptsfl":
        from .baselines.adaptsfl import AdaptSFL
        return AdaptSFL
    if spec.kind == "sflsc3":
        from .baselines.sflsc3 import SFLSC3
        return SFLSC3
    if spec.kind == "shsfl":
        from .baselines.shsfl import SHSFL
        return SHSFL
    raise KeyError(spec.kind)
