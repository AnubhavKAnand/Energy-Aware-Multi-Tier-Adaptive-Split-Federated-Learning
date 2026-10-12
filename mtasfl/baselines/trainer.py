"""
Split federated training with periodic client-side model aggregation, as used
by AdaptSFL (Alg. 1), SFLSC3 (Eq. 8-12) and S-HSFL (Eq. 23-24).

Model blocks 1..Lf ("front") are device-specific between aggregations:
  * AdaptSFL: client-side sub-model w_c,i (blocks 1..c_i) + server-side
    non-common sub-model h_m,i (blocks c_i+1..Lf), Lf = max_i c_i, i.e. the
    "forged client-specific model" h_c,i of Eq. (7).  The cut c_i only decides
    where each part is computed (cost model); the learning dynamics depend on
    Lf and I only.
  * SFLSC3: Lf = Lc (common cut), client models aggregated every I rounds.
  * S-HSFL: Lf = 1 (cut after conv1), I = 1.
Blocks Lf+1..L are the server-side common sub-model, updated every round with
the device-averaged gradient (AdaptSFL Eq. 15, SFLSC3 Eq. 9).

Every device takes one SGD step per round on its own mini-batch (Eq. 5-6),
the same local-computation budget as every other method in this simulator.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..learning.compression import rel_error, topk_sparsify
from ..learning.split_trainer import RoundStats, SplitTrainer


@dataclass
class BaselineJob:
    j: int
    relay: int
    devices: list[int]
    batch: dict = field(default_factory=dict)   # device -> mini-batch size
    cuts: dict = field(default_factory=dict)    # device -> cut layer (compression point)
    fed: set = field(default_factory=set)       # devices training the full model locally


class PerDeviceSplitTrainer(SplitTrainer):
    def __init__(self, model, data, sampler, cfg, device, front_depth: int = 1):
        super().__init__(model, data, sampler, cfg, device)
        self.Lf = int(front_depth)
        self.copies: dict[int, nn.ModuleList] = {}
        self.updated: set[int] = set()

    # ------------------------------------------------------------------ #
    def set_front_depth(self, Lf: int):
        """Change Lf; only allowed when all device models are synchronised."""
        assert not self.copies, "front depth can only change right after aggregation"
        self.Lf = int(Lf)

    def _front(self, dev: int) -> nn.ModuleList:
        if dev not in self.copies:      # device downloads the current client-side model
            self.copies[dev] = copy.deepcopy(nn.ModuleList(self.model.blocks[: self.Lf]))
        return self.copies[dev]

    def _lr(self) -> float:
        wu = self.cfg.sfl.lr_warmup_rounds
        return self.cfg.sfl.lr * (min(1.0, (self.rounds_trained + 1) / wu) if wu > 0 else 1.0)

    # ------------------------------------------------------------------ #
    def train_round(self, jobs: list[BaselineJob]) -> RoundStats:
        st = RoundStats()
        devs = [(d, jb.batch.get(d, self.cfg.sfl.batch_per_device), jb.cuts.get(d, self.Lf), d in jb.fed)
                for jb in jobs for d in jb.devices]
        ratio = self.cfg.sfl.topk_ratio
        errs_f, errs_b = [], []
        if not devs:
            return st
        m = self.model
        m.train()
        lr = self._lr()
        wd = self.cfg.sfl.weight_decay
        self.opt.zero_grad(set_to_none=True)
        tot_loss, correct, n_samp = 0.0, 0, 0
        upd_norm = {}
        for d, b, cut, is_fed in devs:
            front = self._front(d)
            idx = torch.as_tensor(self.sampler.next(d, int(b)), device=self.device)
            x = self.data.prep(self.data.x_train[idx], train=True)
            y = self.data.y_train[idx]
            compress = (ratio < 1.0) and not is_fed     # federated-mode devices send no activations
            cut = min(int(cut), self.Lf)
            h = x
            for blk in front[:cut]:                     # client-side sub-model w_c,i
                h = blk(h)
            if compress:                                # top-k at the cut, no EF (forward)
                with torch.no_grad():
                    z0 = topk_sparsify(h.detach(), ratio)
                    errs_f.append(rel_error(h.detach(), z0))
                z0.requires_grad_()
                z = z0
            else:
                z = h
            for blk in front[cut:]:                     # server-side non-common h_m,i
                z = blk(z)
            logits = m.run(z, self.Lf + 1, self.L)      # server-side common h_s
            loss = F.cross_entropy(logits, y)
            loss.backward()                             # front: grad F_i ; common: accumulates
            if compress and h.requires_grad:            # top-k + EF on the returned gradient
                g = z0.grad
                gs = self.ef.compress_concat(("S", d), g, [d], len(g), cut, ratio)
                errs_b.append(rel_error(g, gs))
                h.backward(gs)
            with torch.no_grad():
                sq = 0.0
                for p in front.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad + wd * p if wd > 0 else p.grad
                    sq += float(g.pow(2).sum())
                    p.add_(g, alpha=-lr)            # Eq. (6): w_c,i <- w_c,i - gamma grad
                    p.grad = None
                upd_norm[d] = lr * sq ** 0.5         # ||Delta w_i||
                tot_loss += float(loss) * len(y)
                correct += int((logits.argmax(1) == y).sum())
                n_samp += len(y)
            self.updated.add(d)
        # server-side common sub-model: device-averaged gradient (Eq. 15)
        with torch.no_grad():
            gn = 0.0
            for p in m.blocks[self.Lf:].parameters():
                if p.grad is not None:
                    p.grad.div_(len(devs))
                    gn += float(p.grad.pow(2).sum())
        for gparam in self.opt.param_groups:
            gparam["lr"] = lr
        self.opt.step()
        self.rounds_trained += 1
        st.lr = lr
        st.loss = tot_loss / n_samp
        st.train_acc = correct / n_samp
        st.n_samples = n_samp
        st.grad_norm = gn ** 0.5
        st.per_device_update = upd_norm
        if errs_f:
            st.fwd_comp_err = {1: float(np.mean(errs_f))}
            st.bwd_comp_err = {1: float(np.mean(errs_b))} if errs_b else {}
        st.ef_evicted = self.ef.end_round()
        st.ef_bytes = self.copies_bytes() + self.ef.nbytes()
        return st

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _average_front(self, only: set | None = None) -> list[dict] | None:
        devs = [d for d in self.updated if d in self.copies and (only is None or d in only)]
        if not devs:
            return None
        out = []
        for l in range(self.Lf):
            sds = [self.copies[d][l].state_dict() for d in devs]
            out.append({k: torch.stack([s[k].float() for s in sds]).mean(0).to(sds[0][k].dtype)
                        for k in sds[0]})
        return out

    @torch.no_grad()
    def aggregate(self) -> int:
        """Client-side model aggregation (AdaptSFL Eq. 7, SFLSC3 Eq. 12,
        S-HSFL Eq. 23): average the device models updated since the last
        aggregation and redistribute.  Returns the number of devices averaged."""
        avg = self._average_front()
        n = len([d for d in self.updated if d in self.copies])
        if avg is not None:
            for l, sd in enumerate(avg):
                self.model.blocks[l].load_state_dict(sd)
        self.copies.clear()
        self.updated.clear()
        return n

    @torch.no_grad()
    def aggregate_from(self, uploaders: set) -> int:
        """Aggregate only the devices that upload their model in this round;
        local progress of devices that miss the aggregation is discarded
        (they download the new model when they next participate)."""
        avg = self._average_front(uploaders)
        n = len([d for d in self.updated if d in self.copies and d in uploaders])
        if avg is None:
            return 0
        for l, sd in enumerate(avg):
            self.model.blocks[l].load_state_dict(sd)
        self.copies.clear()
        self.updated.clear()
        return n

    def copies_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for c in self.copies.values() for p in c.parameters()))

    @torch.no_grad()
    def evaluate(self, batch: int = 1000) -> dict:
        """Evaluate the averaged model w = (1/N) sum_i w_i (the quantity the
        AdaptSFL / SFLSC3 convergence analysis is about)."""
        avg = self._average_front()
        if avg is None:
            return super().evaluate(batch)
        saved = [copy.deepcopy(self.model.blocks[l].state_dict()) for l in range(self.Lf)]
        for l, sd in enumerate(avg):
            self.model.blocks[l].load_state_dict(sd)
        out = super().evaluate(batch)
        for l, sd in enumerate(saved):
            self.model.blocks[l].load_state_dict(sd)
        return out

    # ------------------------------------------------------------------ #
    def probe_moments(self, devices: list[int], rng: np.random.Generator, n_batches: int = 2) -> dict:
        """Estimate the per-block gradient moments used by the convergence
        bounds of AdaptSFL (G_j^2, sigma_j^2) and SFLSC3 (G_l^2, sigma_l^2,
        Lambda^2) on the current synchronised model."""
        return probe_moments(self.model, self.data, devices, self.sampler.shards, rng,
                             self.cfg.sfl.batch_per_device, self.device, n_batches)

    def state(self) -> dict:
        return dict(opt=self.opt.state_dict(), rounds_trained=self.rounds_trained, Lf=self.Lf,
                    ef={k: (v[0], v[1].detach().to("cpu", torch.float16), v[2]) for k, v in self.ef.store.items()},
                    ef_round=self.ef.round,
                    copies={d: {k: v.detach().cpu() for k, v in c.state_dict().items()}
                            for d, c in self.copies.items()},
                    updated=sorted(self.updated))

    def load_state(self, s: dict):
        self.opt.load_state_dict(s["opt"])
        self.rounds_trained = s["rounds_trained"]
        self.copies.clear()
        self.Lf = s["Lf"]
        for d, sd in s["copies"].items():
            c = copy.deepcopy(nn.ModuleList(self.model.blocks[: self.Lf]))
            c.load_state_dict(sd)
            self.copies[d] = c
        self.updated = set(s["updated"])
        self.ef.store = {k: (v[0], v[1].to(self.device, torch.float32), v[2]) for k, v in s["ef"].items()}
        self.ef.round = s["ef_round"]


def probe_moments(model, data, devices, shards, rng, b, device, n_batches=2) -> dict:
    """Per-block E||g||^2 (G^2), within-device variance (sigma^2) and
    across-device dispersion (Lambda^2) from a few devices' mini-batches."""
    L = model.L
    model.train()
    per_dev = []
    var_terms = []
    for d in devices:
        gs = []
        for _ in range(n_batches):
            pool = shards[d] if shards is not None else np.arange(len(data.y_train))
            idx = torch.as_tensor(rng.choice(pool, size=min(b, len(pool)), replace=False), device=device)
            x = data.prep(data.x_train[idx], train=True)
            y = data.y_train[idx]
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            gs.append([torch.cat([p.grad.reshape(-1) for p in model.blocks[l].parameters()
                                  if p.grad is not None]).detach().clone()
                       if any(p.grad is not None for p in model.blocks[l].parameters())
                       else torch.zeros(1, device=device) for l in range(L)])
        mean_d = [torch.stack([g[l] for g in gs]).mean(0) for l in range(L)]
        per_dev.append((gs, mean_d))
        var_terms.append([float(torch.stack([(g[l] - mean_d[l]).pow(2).sum() for g in gs]).mean())
                          * n_batches / max(n_batches - 1, 1) for l in range(L)])
    model.zero_grad(set_to_none=True)
    G2 = np.zeros(L + 1)
    sig2 = np.zeros(L + 1)
    for l in range(L):
        G2[l + 1] = np.mean([float(g[l].pow(2).sum()) for gs, _ in per_dev for g in gs])
        sig2[l + 1] = np.mean([v[l] for v in var_terms])
    glob = [torch.stack([md[l] for _, md in per_dev]).mean(0) for l in range(L)]
    lam2 = float(np.mean([sum(float((md[l] - glob[l]).pow(2).sum()) for l in range(L)) for _, md in per_dev]))
    return dict(G2=G2, sigma2=sig2, lambda2=lam2)
