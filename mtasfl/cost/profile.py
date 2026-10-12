"""
Layer-block profile of an L-block DNN, as used by the measurement model
(Sec. IV).  Block l in 1..L has per-sample forward FLOPs F[l], backward FLOPs
B[l] and output activation size D[l] (elements per sample).  Index 0 holds the
input (F[0]=B[0]=0, D[0]=input elements).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class ModelProfile:
    name: str
    fwd: np.ndarray          # (L+1,)  FLOPs per sample
    bwd: np.ndarray          # (L+1,)
    act: np.ndarray          # (L+1,)  elements per sample
    n_params: np.ndarray = field(default=None)  # (L+1,) parameters per block

    def __post_init__(self):
        self.fwd = np.asarray(self.fwd, float)
        self.bwd = np.asarray(self.bwd, float)
        self.act = np.asarray(self.act, float)
        self._cf = np.cumsum(self.fwd)
        self._cb = np.cumsum(self.bwd)
        if self.n_params is None:
            self.n_params = np.zeros_like(self.fwd)

    @property
    def L(self) -> int:
        return len(self.fwd) - 1

    def cfp(self, a: int, b: int) -> float:
        """C_FP(a,b): forward FLOPs of blocks a..b inclusive (0 if a>b)."""
        if a > b:
            return 0.0
        return float(self._cf[b] - self._cf[a - 1])

    def cbp(self, a: int, b: int) -> float:
        if a > b:
            return 0.0
        return float(self._cb[b] - self._cb[a - 1])


def sparse_bits(n_elems: float, ratio: float, w: int = 32) -> float:
    """Payload of top-k sparsified tensor with D=n_elems:  k (w + r) bits,
    r = ceil(log2 D)  (Sec. III-C).  ratio >= 1 means dense (D w bits)."""
    if n_elems <= 0:
        return 0.0
    if ratio >= 1.0:
        return n_elems * w
    k = max(1, math.ceil(ratio * n_elems))
    r = math.ceil(math.log2(max(n_elems, 2)))
    return k * (w + r)


def synthetic_profile(L: int = 8, seed: int = 0) -> ModelProfile:
    """Small synthetic profile for unit tests."""
    rng = np.random.default_rng(seed)
    fwd = np.concatenate([[0], rng.uniform(5e6, 5e7, L)])
    act = np.concatenate([[3072], np.linspace(16384, 64, L).round()])
    params = np.concatenate([[0], rng.uniform(1e4, 5e5, L).round()])
    return ModelProfile("synthetic", fwd, 2 * fwd, act, params)
