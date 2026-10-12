"""
Multi-tier split federated training (Sec. III-A-2, Eq. 4).

One synchronous round:
  for every completed pipeline j (primary j, relay u, split p, surviving IoT set)
     Tier 1  IoT i      : delta_1,i = h1(x_i)            -> top-k (no EF), per device
     Tier 2  primary j  : concat, delta_2 = h2(.)         -> top-k (no EF)
     Tier 3  relay u    : delta_3 = h3(.)  (identity if p2 == p3, bridge mode)
     Tier 4  cloud      : forward p3+1..L, loss
     backward           : cloud -> relay -> primary -> IoT, each hop top-k + EF
  aggregation (Eq. 4)   : W <- W - eta * sum_j (B_j / B_total) grad_j
Because every pipeline computes on the same global weights W(t) (redistributed
by the cloud at the end of each round), accumulating sum-reduced losses divided
by B_total into a single parameter set realises Eq. (4) exactly, layer by
layer, regardless of which tier computed each layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .compression import ErrorFeedbackBank, rel_error, topk_sparsify


@dataclass
class PipelineJob:
    j: int                       # primary UAV
    relay: int                   # relay UAV
    p: tuple                     # (p1, p2, p3)
    devices: list[int]           # surviving IoT devices


@dataclass
class RoundStats:
    loss: float = float("nan")
    train_acc: float = float("nan")
    n_samples: int = 0
    fwd_comp_err: dict = field(default_factory=dict)   # tier -> mean relative error
    bwd_comp_err: dict = field(default_factory=dict)
    grad_norm: float = 0.0
    ef_resets: int = 0
    per_pipeline_loss: dict = field(default_factory=dict)
    lr: float = 0.0
    ef_evicted: int = 0
    ef_bytes: int = 0
    per_device_update: dict = field(default_factory=dict)


class SplitTrainer:
    def __init__(self, model, data, sampler, cfg, device):
        self.model = model.to(device)
        self.data = data
        self.sampler = sampler
        self.cfg = cfg
        self.device = device
        s = cfg.sfl
        self.opt = torch.optim.SGD(self.model.parameters(), lr=s.lr, momentum=s.momentum,
                                   weight_decay=s.weight_decay)
        self.ef = ErrorFeedbackBank()
        self.L = model.L
        self.rounds_trained = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _backward(out, inp, grad):
        """Back-propagate `grad` from `out` to the leaf `inp`.  When the
        segment is empty (identity: out is inp) the gradient passes through."""
        if out is inp:
            inp.grad = grad if inp.grad is None else inp.grad + grad
        else:
            out.backward(grad)

    def _batch(self, devices, b):
        idx = np.concatenate([self.sampler.next(i, b) for i in devices])
        idx_t = torch.as_tensor(idx, device=self.device)
        x = self.data.prep(self.data.x_train[idx_t], train=True)
        y = self.data.y_train[idx_t]
        return x, y

    # ------------------------------------------------------------------ #
    def train_round(self, jobs: list[PipelineJob]) -> RoundStats:
        st = RoundStats()
        jobs = [jb for jb in jobs if len(jb.devices) > 0]
        if not jobs:
            return st
        b = self.cfg.sfl.batch_per_device
        ratio = self.cfg.sfl.topk_ratio
        B_total = sum(len(jb.devices) for jb in jobs) * b
        m = self.model
        m.train()
        self.opt.zero_grad(set_to_none=True)
        resets0 = self.ef.resets
        errs_f = {1: [], 2: [], 3: []}
        errs_b = {1: [], 2: [], 3: []}
        tot_loss, correct = 0.0, 0

        for jb in jobs:
            p1, p2, p3 = jb.p
            x, y = self._batch(jb.devices, b)
            # ---------------- forward -------------------------------------
            h1 = m.run(x, 1, p1)                                   # tier 1
            with torch.no_grad():
                d1 = torch.cat([topk_sparsify(c, ratio) for c in h1.detach().split(b)])
                errs_f[1].append(rel_error(h1.detach(), d1))
            d1 = d1.requires_grad_()
            h2 = m.run(d1, p1 + 1, p2)                             # tier 2
            with torch.no_grad():
                d2 = topk_sparsify(h2.detach(), ratio)
                errs_f[2].append(rel_error(h2.detach(), d2))
            d2 = d2.requires_grad_()
            h3 = m.run(d2, p2 + 1, p3)                             # tier 3 (identity if bridge)
            with torch.no_grad():
                d3 = h3.detach() if p2 == p3 else topk_sparsify(h3.detach(), ratio)
                errs_f[3].append(rel_error(h3.detach(), d3))
            d3 = d3.requires_grad_()
            logits = m.run(d3, p3 + 1, self.L)                     # tier 4
            loss = F.cross_entropy(logits, y, reduction="sum") / B_total
            # ---------------- backward ------------------------------------
            loss.backward()
            g3 = d3.grad
            g3s = self.ef.compress_concat(("S", jb.j), g3, jb.devices, b, p3, ratio)
            errs_b[3].append(rel_error(g3, g3s))
            self._backward(h3, d2, g3s)
            g2 = d2.grad
            g2s = self.ef.compress_concat(("R", jb.relay, jb.j), g2, jb.devices, b, p2, ratio)
            errs_b[2].append(rel_error(g2, g2s))
            self._backward(h2, d1, g2s)
            g1 = d1.grad
            g1s = self.ef.compress_per_device(("P", jb.j), g1, jb.devices, b, p1, ratio)
            errs_b[1].append(rel_error(g1, g1s))
            h1.backward(g1s)

            with torch.no_grad():
                lj = float(F.cross_entropy(logits, y, reduction="sum"))
                tot_loss += lj
                correct += int((logits.argmax(1) == y).sum())
                st.per_pipeline_loss[jb.j] = lj / len(y)

        # ---------------- Eq. (4) aggregation ---------------------------------
        with torch.no_grad():
            gn = torch.sqrt(sum((p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None),
                                torch.tensor(0.0, device=self.device)))
        # Eq. (4) step size, with optional linear warm-up
        wu = self.cfg.sfl.lr_warmup_rounds
        lr = self.cfg.sfl.lr * (min(1.0, (self.rounds_trained + 1) / wu) if wu > 0 else 1.0)
        for gparam in self.opt.param_groups:
            gparam["lr"] = lr
        self.opt.step()
        self.rounds_trained += 1
        st.lr = lr
        st.loss = tot_loss / B_total
        st.train_acc = correct / B_total
        st.n_samples = B_total
        st.grad_norm = float(gn)
        st.fwd_comp_err = {k: float(np.mean(v)) for k, v in errs_f.items() if v}
        st.bwd_comp_err = {k: float(np.mean(v)) for k, v in errs_b.items() if v}
        st.ef_resets = self.ef.resets - resets0
        st.ef_evicted = self.ef.end_round()
        st.ef_bytes = self.ef.nbytes()
        return st

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, batch: int = 1000) -> dict:
        m = self.model
        m.eval()
        xs, ys = self.data.x_test, self.data.y_test
        ncls = self.data.num_classes
        loss, correct = 0.0, 0
        cls_correct = torch.zeros(ncls, device=self.device)
        cls_total = torch.zeros(ncls, device=self.device)
        for s in range(0, len(ys), batch):
            x = self.data.prep(xs[s:s + batch], train=False)
            y = ys[s:s + batch]
            out = m(x)
            loss += float(F.cross_entropy(out, y, reduction="sum"))
            pred = out.argmax(1)
            correct += int((pred == y).sum())
            cls_total += torch.bincount(y, minlength=ncls).float()
            cls_correct += torch.bincount(y[pred == y], minlength=ncls).float()
        m.train()
        n = len(ys)
        per_class = (cls_correct / cls_total.clamp(min=1)).cpu().numpy()
        return dict(test_acc=correct / n, test_loss=loss / n, per_class_acc=per_class)
