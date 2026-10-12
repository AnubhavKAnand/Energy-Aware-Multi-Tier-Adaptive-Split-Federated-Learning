"""
AdaptSFL: Adaptive Split Federated Learning in Resource-Constrained Edge
Networks (Z. Lin, G. Qu, W. Wei, X. Chen, K. K. Leung, IEEE/ACM ToN 33(6), 2025).

Implemented components
  * per-device model splitting mu (cut c_i for device i), client-side model
    aggregation (MA) of the forged client-specific models every I rounds,
    server-side common sub-model updated every round (Alg. 1, Eq. 4-7);
  * Problem (28):  min_{I,mu}  Theta(I,mu) = R(I,mu)/I * T(I,mu)
        R   = 2 theta / ( gamma ( eps - beta gamma sum_j sigma_j^2 / N
                                  - 1{I>1} 4 beta^2 gamma^2 I^2 Gtilde_{Lc} ) )   (Cor. 1)
        T   = I * T_split(mu) + T_MA(mu)                                       (Eq. 26)
        Gtilde_{Lc} = sum_{j<=Lc} G_j^2,  Lc = max_i c_i                       (Eq. 29)
  * Alg. 3 (BCD): I-step = exact integer search (Thm. 2), mu-step = exact
    evaluation of the objective with coordinate descent over per-device cuts
    (stands in for the Dinkelbach MILFP solver of Alg. 2; same objective);
  * re-optimisation at every aggregation boundary with freshly estimated
    G_j^2, sigma_j^2 (paper: "Algorithm 3 can be conducted every I rounds").

Environment mapping: devices = IoT devices, edge/fed server = cloud, UAVs
forward.  T_split / T_MA are evaluated with the shared latency model.
eps (target accuracy) is set so the convergence constraint admits I up to
I_cap = 25, the interval range used in the paper's experiments.
The optimiser sees the environment's straggler rule (as MTA-SFL's masks do):
devices that cannot meet the local deadline with any cut are excluded from the
max in Eq. (26) (they are dropped anyway), and a device that can participate is
never assigned a cut that would drop it.
"""
from __future__ import annotations

import numpy as np

from .base import BaselineSimulation


class AdaptSFL(BaselineSimulation):
    initial_front_depth = 1

    def init_baseline(self):
        self.cut_map: dict[int, int] = {}
        self.c_default = 1
        self.need_opt = True
        self.last_theta = None
        self.bcd_iters_used = 0

    def baseline_state(self):
        return dict(cut_map=self.cut_map, c_default=self.c_default, need_opt=self.need_opt,
                    last_theta=self.last_theta, Lf=self.front_depth)

    def load_baseline_state(self, s):
        self.cut_map, self.c_default, self.need_opt = s["cut_map"], s["c_default"], s["need_opt"]
        self.last_theta = s["last_theta"]
        self._Lf_fast = s["Lf"]

    # ------------------------------------------------------------------ #
    def plan(self, ctx, recs):
        if self.need_opt and recs:
            self._optimise(ctx, recs)
            self.need_opt = False
        b = self.cfg.sfl.batch_per_device
        for rec in recs:
            devs = rec["devs"]
            rec["cuts"] = np.array([self.cut_map.get(int(d), self.c_default) for d in devs], int)
            rec["bsz"] = np.full(len(devs), b, int)

    def after_aggregation(self, ctx):
        if self.full:
            self.moments.update(self._probe(self.cfg.adaptsfl.probe_devices))
        self.need_opt = True

    # ------------------------------------------------------------------ #
    def _optimise(self, ctx, recs):
        ac = self.cfg.adaptsfl
        L = self.L
        b = self.cfg.sfl.batch_per_device
        gamma, beta = self.cfg.sfl.lr, ac.beta
        G2, sig2, theta = self.moments["G2"], self.moments["sigma2"], self.moments["theta"]
        Gcum = np.cumsum(G2)
        N = sum(len(r["devs"]) for r in recs)
        A = beta * gamma * float(sig2[1:].sum()) / max(N, 1)
        Bc = 4 * beta ** 2 * gamma ** 2
        eps = A + Bc * (ac.I_cap + 1) ** 2 * float(Gcum[L - 1])
        cuts_all = np.arange(1, L)

        def R(I, Lc):
            den = eps - A - (Bc * I ** 2 * Gcum[Lc] if I > 1 else 0.0)
            return 2 * theta / (gamma * den) if den > 0 else np.inf

        # environment feasibility (shared straggler rule): F[i, c]; devices with
        # no feasible cut will be dropped whatever the decision -> excluded
        Fs = [self.feasibility(ctx, r["devs"], b) for r in recs]
        act = [F.any(1) for F in Fs]

        def pipe_times(pi, cuts):
            rec, a = recs[pi], act[pi]
            if not a.any():
                return 0.0, 0.0
            ts = self.estimated_pipeline(ctx, rec, cuts, np.full(len(cuts), b), active=a).t_total
            up = np.array([self.param_bits(1, c) for c in cuts])
            tm = self.estimated_ma(ctx, rec, up, float(up[a].max()), active=a)
            return ts, tm

        def admissible(pi, cuts):           # never drop a device that could participate
            return bool(np.all(Fs[pi][np.arange(len(cuts)), cuts] | ~act[pi]))

        def theta_of(I, Tsp, Tma, Lc):
            return R(I, Lc) / I * (I * max(Tsp) + max(Tma))

        # current per-device cuts (warm start)
        cuts = [np.array([self.cut_map.get(int(d), self.c_default) for d in r["devs"]], int) for r in recs]
        for pi in range(len(recs)):         # repair warm start: move each active device to a feasible cut
            for k in range(len(cuts[pi])):
                if act[pi][k] and not Fs[pi][k, cuts[pi][k]]:
                    cuts[pi][k] = int(np.argmax(Fs[pi][k, 1:]) + 1)
        I = self.I
        times = [pipe_times(pi, c) for pi, c in enumerate(cuts)]
        Tsp = [t[0] for t in times]
        Tma = [t[1] for t in times]
        Lc = max(int(c.max()) for c in cuts)
        best = theta_of(I, Tsp, Tma, Lc)
        it = 0
        for it in range(1, ac.bcd_iters + 1):
            prev = best
            # ---- I-step (Theorem 2: integer minimiser) ----
            vals = [theta_of(i, Tsp, Tma, Lc) for i in range(1, ac.I_cap + 1)]
            I = int(np.argmin(vals)) + 1
            best = vals[I - 1]
            # ---- mu-step: best uniform cut as candidate start ----
            for c in cuts_all:
                cu = [np.full(len(r["devs"]), c) for r in recs]
                if not all(admissible(pi, x) for pi, x in enumerate(cu)):
                    continue
                tt = [pipe_times(pi, x) for pi, x in enumerate(cu)]
                v = theta_of(I, [t[0] for t in tt], [t[1] for t in tt], int(c))
                if v < best - 1e-12:
                    best, cuts = v, cu
                    Tsp, Tma = [t[0] for t in tt], [t[1] for t in tt]
            Lc = max(int(c.max()) for c in cuts)
            # ---- mu-step: coordinate descent over individual devices ----
            for _ in range(ac.cd_passes):
                changed = False
                for pi, rec in enumerate(recs):
                    for k in range(len(rec["devs"])):
                        cur = cuts[pi][k]
                        for c in cuts_all:
                            if c == cur or (act[pi][k] and not Fs[pi][k, c]):
                                continue
                            trial = cuts[pi].copy()
                            trial[k] = c
                            ts, tm = pipe_times(pi, trial)
                            Tsp2, Tma2 = list(Tsp), list(Tma)
                            Tsp2[pi], Tma2[pi] = ts, tm
                            Lc2 = max([int(x.max()) for j_, x in enumerate(cuts) if j_ != pi]
                                      + [int(trial.max())])
                            v = theta_of(I, Tsp2, Tma2, Lc2)
                            if v < best - 1e-12:
                                best, cuts[pi], Tsp, Tma, Lc, cur = v, trial, Tsp2, Tma2, Lc2, c
                                changed = True
                if not changed:
                    break
            if abs(prev - best) <= 1e-9 * max(abs(best), 1.0):
                break
        self.bcd_iters_used = it
        # ---- apply ----
        self.cut_map = {int(d): int(c) for r, cs in zip(recs, cuts) for d, c in zip(r["devs"], cs)}
        vals, cnts = np.unique(np.concatenate(cuts), return_counts=True)
        self.c_default = int(vals[np.argmax(cnts)])
        self.I = int(I)
        self.last_theta = float(best) if np.isfinite(best) else None
        self._set_front_depth(max(Lc, self.c_default))

    def round_fields(self, ctx):
        cuts = list(self.cut_map.values())
        return dict(theta=self.last_theta, c_default=int(self.c_default),
                    cut_hist=np.bincount(cuts, minlength=self.L).tolist() if cuts else None,
                    bcd_iters=int(self.bcd_iters_used))
