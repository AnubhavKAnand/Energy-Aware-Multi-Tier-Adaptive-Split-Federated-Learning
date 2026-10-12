"""
SFLSC3: Split Federated Learning for Low-Altitude Wireless Networks - Joint
Sensing, Communication, Computation, and Control Co-design
(X. Hou, X. Wang, J. Wang, Z. Zhang, J. Du, J. Wang, Y. Ren, arXiv:2504.01443v2,
2026; conference version IEEE ICC 2025).

Implemented components
  * common split point Lc, client-side sub-models aggregated every I rounds
    (Eq. 12), server-side sub-models averaged every round (Eq. 9);
  * Problem P2 (Eq. 37-38):
        min_{I, Lc, b, u}  Xi = sum_m theta E_m(I,Lc,u,b) /
                  ( I ( eps' - Gamma1 Gtilde_{Lc} I^2 - Gamma2 / b - Gamma3/q_s^(M-2) - Gamma4 ) )
        E_m = I e (T_tr,m + T_sma,m + T_s,m) + T_par,m                       (Eq. 20-21)
    with Gamma1..Gamma4 of Eq. (35); BCD over I (Eq. 40), b (Eq. 42), Lc by
    traversal (Eq. 43) and the UAV position (Eq. 44-46);
  * re-optimisation at every aggregation boundary with fresh G_l^2, sigma_l^2,
    Lambda^2 estimates.

Environment mapping: SFLSC3's clients are sensing UAVs; here the clients are
the IoT devices and there is no sensing stage, so q_s = 1 (Gamma3 term is a
constant, T_s = 0) and e = 1 local iteration per round (same local budget as
every method).  The controllable UAV position is the primary UAV's hover point,
set to the rate-optimal point of its cluster (closed form of Eq. 45-46 with the
sensing-angle constraint removed: the centroid of the associated devices).
b in {8, 16, 24, 32} and I in 1..100 follow the paper's evaluation ranges.
eps' is set so the convergence constraint admits the whole search box.
The optimiser sees the environment's straggler rule (as MTA-SFL's masks do):
admissible (Lc, b) keep the largest number of devices able to meet the local
deadline, and Xi sums E_m over those devices.
"""
from __future__ import annotations

import numpy as np

from .base import BaselineSimulation


class SFLSC3(BaselineSimulation):
    initial_front_depth = 1

    def init_baseline(self):
        self.Lc = 1
        self.b = int(self.cfg.sfl.batch_per_device)
        self.need_opt = True
        self.targets: dict[int, np.ndarray] = {}
        self.last_xi = None

    def baseline_state(self):
        return dict(Lc=self.Lc, b=self.b, need_opt=self.need_opt, targets=self.targets,
                    last_xi=self.last_xi, Lf=self.front_depth)

    def load_baseline_state(self, s):
        self.Lc, self.b, self.need_opt = s["Lc"], s["b"], s["need_opt"]
        self.targets, self.last_xi, self._Lf_fast = s["targets"], s["last_xi"], s["Lf"]

    # ------------------------------------------------------------------ #
    def plan(self, ctx, recs):
        if self.need_opt and recs:
            self._optimise(ctx, recs)
            self.need_opt = False
        for rec in recs:
            n = len(rec["devs"])
            rec["cuts"] = np.full(n, self.Lc, int)
            rec["bsz"] = np.full(n, self.b, int)

    def after_aggregation(self, ctx):
        if self.full:
            self.moments.update(self._probe(self.cfg.sflsc3.probe_devices))
        self.need_opt = True

    def _after_mobility(self, ctx):
        """Primary UAVs hover at the position chosen by the optimiser."""
        if not self.cfg.sflsc3.optimise_position:
            return
        for j, xy in self.targets.items():
            self.net.uav_xyz[j, :2] = xy

    # ------------------------------------------------------------------ #
    def _optimise(self, ctx, recs):
        sc = self.cfg.sflsc3
        net, prof, L = self.net, self.prof, self.L
        w = self.cfg.sfl.float_bits
        eta, beta = self.cfg.sfl.lr, sc.beta
        G2, sig2 = self.moments["G2"], self.moments["sigma2"]
        lam2, theta = self.moments["lambda2"], self.moments["theta"]
        Gcum = np.cumsum(G2)
        M = sum(len(r["devs"]) for r in recs)
        # Eq. (35) with q_s = 1
        G1 = 16 * beta ** 3 * eta ** 4 + 4 * M * beta ** 2 * eta ** 3
        G2c = beta * eta ** 2 * float(sig2[1:].sum()) / max(M, 1)
        G4 = M * eta * lam2 + 2 * beta * eta ** 2 * lam2 / max(M, 1)
        b_choices = np.array(sc.b_choices, float)
        eps_p = G4 + G2c / b_choices.min() + G1 * float(Gcum[L - 1]) * (sc.I_cap + 1) ** 2

        # per-device quantities: compute speed and serial path rate (Eq. 15-19)
        f_m, inv_r, all_devs = [], [], []
        for rec in recs:
            j, u, devs = rec["j"], rec["relay"], rec["devs"]
            r_ju, _, r_bh = self._path_rates(ctx, j, u, ctx["Q"][u])
            f_m.append(net.f_iot[devs])
            inv_r.append(1.0 / np.maximum(ctx["r_ul_e"][devs], 1e-9) + 1.0 / max(r_ju, 1e-9) + 1.0 / max(r_bh, 1e-9))
            all_devs.append(devs)
        f_m, inv_r, all_devs = np.concatenate(f_m), np.concatenate(inv_r), np.concatenate(all_devs)
        flc = np.array([prof.cfp(1, c) + prof.cbp(1, c) for c in range(L + 1)])   # per sample
        delta = np.array([self.param_bits(1, c) for c in range(L + 1)])
        # environment feasibility (shared straggler rule): devices expected to meet the
        # local deadline for each (Lc, b); admissible choices keep the maximum number
        # of participants (no voluntary dropping), Xi sums over those devices
        Fb = {float(bb): self.feasibility(ctx, all_devs, int(bb)) for bb in b_choices}
        n_feas = {(c, bb): int(Fb[bb][:, c].sum()) for bb in Fb for c in range(1, L)}
        n_best = max(n_feas.values())

        def xi(I, Lc, b):
            if n_feas[(Lc, b)] < n_best or n_best == 0:
                return np.inf
            den = eps_p - G1 * Gcum[Lc] * I ** 2 - G2c / b - G4
            if den <= 0:
                return np.inf
            a = Fb[b][:, Lc]
            s_inv_f, s_inv_r = float((1.0 / f_m[a]).sum()), float(inv_r[a].sum())
            sum_E = I * (b * flc[Lc] * s_inv_f + b * prof.act[Lc] * w * s_inv_r) + delta[Lc] * s_inv_r
            return theta * sum_E / (I * den)

        I, Lc, b = self.I, self.Lc, float(self.b)
        if not np.isfinite(xi(I, Lc, b)):    # warm start must be admissible
            Lc, b = min(((c, bb) for (c, bb), v in n_feas.items() if v == n_best), key=lambda x: (x[1], x[0]))
        best = xi(I, Lc, b)
        for _ in range(sc.bcd_iters):
            prev = (I, Lc, b)
            vals = [xi(i, Lc, b) for i in range(1, sc.I_cap + 1)]          # P3 (Eq. 40)
            I = int(np.argmin(vals)) + 1
            vals = [xi(I, Lc, bb) for bb in b_choices]                       # P4 (Eq. 42)
            b = float(b_choices[int(np.argmin(vals))])
            vals = [xi(I, c, b) for c in range(1, L)]                        # P5 traversal (Eq. 43)
            Lc = int(np.argmin(vals)) + 1
            best = xi(I, Lc, b)
            if (I, Lc, b) == prev:
                break
        # UAV deployment (P6, Eq. 44-46): rate-optimal hover point of each cluster
        if sc.optimise_position:
            tether = self.cfg.topology.iot_cluster_std_m
            for rec in recs:
                j = rec["j"]
                c = net.iot_xy[rec["devs"]].mean(0)
                off = c - net.hotspots[j]
                dist = np.linalg.norm(off)
                if dist > tether:
                    c = net.hotspots[j] + off * tether / dist
                self.targets[j] = c
        self.I, self.Lc, self.b = int(I), int(Lc), int(b)
        self.last_xi = float(best) if np.isfinite(best) else None
        self._set_front_depth(self.Lc)

    def round_fields(self, ctx):
        return dict(xi=self.last_xi, Lc=int(self.Lc), b=int(self.b))
