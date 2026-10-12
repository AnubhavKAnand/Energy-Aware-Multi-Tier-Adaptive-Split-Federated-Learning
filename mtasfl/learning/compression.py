"""
Top-k sparsification (Sec. III-C).

* Comp / Decomp (Eq. 6-7): keep the k largest-magnitude entries of the
  flattened tensor, k = ceil(ratio * D); the rest are zero-padded.
* Forward activations: NO error feedback (paper, Sec. III-C).
* Backward gradients: error feedback at every transmitting node n:
      g_acc = grad + e_n ;  sparse = Comp(g_acc) ;  e_n <- g_acc - Decomp(sparse)

Gradient tensors flowing backwards are concatenations of per-IoT-device
chunks, while the set of surviving devices (and so the tensor shape) changes
from round to round.  The residual e_n is therefore stored per
(sender, device, cut-layer) chunk; a chunk whose cut layer or shape changed
starts from zero (counted in `resets`).
"""
from __future__ import annotations

import math

import torch


def topk_sparsify(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """Decomp(Comp(x)) : dense tensor with only the top-k entries kept."""
    if ratio >= 1.0:
        return x
    flat = x.reshape(-1)
    k = max(1, math.ceil(ratio * flat.numel()))
    if k >= flat.numel():
        return x
    idx = torch.topk(flat.abs(), k, sorted=False).indices
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out.view_as(x)


def rel_error(x: torch.Tensor, xh: torch.Tensor) -> float:
    den = float(x.pow(2).sum())
    return float((x - xh).pow(2).sum()) / den if den > 0 else 0.0


class ErrorFeedbackBank:
    """Residuals e_n are node-local.  Entries a node has not used for `ttl`
    rounds (e.g. a relay that stopped serving a primary) are evicted, otherwise
    the store grows with every (relay, device) pair ever used."""

    def __init__(self, ttl: int = 20):
        self.store: dict = {}           # key -> (cut, residual, last_round)
        self.resets = 0
        self.ttl = ttl
        self.round = 0

    def _get(self, key, cut, like: torch.Tensor):
        v = self.store.get(key)
        if v is None or v[0] != cut or v[1].shape != like.shape:
            if v is not None:
                self.resets += 1
            return torch.zeros_like(like)
        return v[1]

    def _put(self, key, cut, r):
        self.store[key] = (cut, r.detach(), self.round)

    def end_round(self) -> int:
        """Advance the round counter and evict stale entries; returns #evicted."""
        old = [k for k, v in self.store.items() if self.round - v[2] >= self.ttl]
        for k in old:
            del self.store[k]
        self.round += 1
        return len(old)

    def nbytes(self) -> int:
        return int(sum(v[1].numel() * v[1].element_size() for v in self.store.values()))

    def compress_concat(self, sender, grad: torch.Tensor, devices: list[int], b: int,
                        cut: int, ratio: float) -> torch.Tensor:
        """EF top-k over the whole concatenated payload (one link)."""
        chunks = grad.split(b)
        e = torch.cat([self._get((sender, d), cut, c) for d, c in zip(devices, chunks)])
        acc = grad + e
        sparse = topk_sparsify(acc, ratio)
        res = (acc - sparse).split(b)
        for d, r in zip(devices, res):
            self._put((sender, d), cut, r)
        return sparse

    def compress_per_device(self, sender, grad: torch.Tensor, devices: list[int], b: int,
                            cut: int, ratio: float) -> torch.Tensor:
        """EF top-k separately for each device link (primary -> IoT i)."""
        outs = []
        for d, c in zip(devices, grad.split(b)):
            acc = c + self._get((sender, d), cut, c)
            sp = topk_sparsify(acc, ratio)
            self._put((sender, d), cut, acc - sp)
            outs.append(sp)
        return torch.cat(outs)
