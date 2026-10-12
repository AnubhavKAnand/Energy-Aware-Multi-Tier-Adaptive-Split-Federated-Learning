"""
Per-block profile (FLOPs, activation size, parameters) of a BlockNet, used by
the analytic measurement model (C_FP, C_BP, |delta|).

Forward FLOPs count 2 x MACs of Conv/Linear layers plus element-wise cost of
GroupNorm / ReLU / pooling.  Backward FLOPs are taken as 2 x forward (gradient
w.r.t. activations + gradient w.r.t. weights), the standard approximation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..cost.profile import ModelProfile


def _leaf_flops(m: nn.Module, inp, out) -> float:
    x = inp[0]
    if isinstance(m, nn.Conv2d):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        return 2.0 * out.numel() * k
    if isinstance(m, nn.Conv1d):
        k = m.kernel_size[0] * (m.in_channels // m.groups)
        return 2.0 * out.numel() * k
    if isinstance(m, nn.Linear):
        return 2.0 * out.numel() * m.in_features
    if isinstance(m, nn.GroupNorm):
        return 5.0 * out.numel()
    if isinstance(m, (nn.ReLU,)):
        return 1.0 * out.numel()
    if isinstance(m, (nn.MaxPool2d, nn.MaxPool1d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
        return 1.0 * x.numel()
    return 0.0


@torch.no_grad()
def profile_model(model, in_shape: tuple) -> ModelProfile:
    model = model.eval()
    dev = next(model.parameters()).device
    x = torch.zeros((1, *in_shape), device=dev)
    L = model.L
    fwd = np.zeros(L + 1)
    act = np.zeros(L + 1)
    nparam = np.zeros(L + 1)
    act[0] = x.numel()
    for l in range(1, L + 1):
        blk = model.blocks[l - 1]
        acc = [0.0]
        hooks = [m.register_forward_hook(lambda m, i, o: acc.__setitem__(0, acc[0] + _leaf_flops(m, i, o)))
                 for m in blk.modules() if len(list(m.children())) == 0]
        y = blk(x)
        # residual additions in BasicBlock
        if hasattr(blk, "sc"):
            acc[0] += y.numel()
        for h in hooks:
            h.remove()
        fwd[l] = acc[0]
        act[l] = y[0].numel()
        nparam[l] = sum(p.numel() for p in blk.parameters())
        # functional relu/avgpool inside BasicBlock / Head are not modules
        if hasattr(blk, "c1"):
            fwd[l] += 2 * y.numel()
        if blk.__class__.__name__ == "Head":
            fwd[l] += x.numel()          # global average pooling over the input
        x = y
    model.train()
    return ModelProfile(getattr(model, "name", "model"), fwd, 2.0 * fwd, act, nparam)
