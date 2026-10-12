"""
Common machinery for the baselines in the shared MTA-SFL environment.

Identical to MTA-SFL (inherited from Simulation): topology, mobility, channel
realisations, association (Eq. 1), straggler rule, link-failure test, T_max
deadline, energy model, battery / depletion handling, data partition, model,
learning-rate schedule and every logged metric.

Method-specific (implemented by subclasses through `plan` / `after_aggregation`):
cut layers, batch size, client-side aggregation interval I, device selection.

Shared adaptation rules for two-tier baselines in the 4-tier network:
  * their server = the cloud; primary and relay UAVs forward only (no compute)
  * relay = the cloud-connected neighbour with the largest bottleneck rate
    min(R_{j,u}, R_{u,S}) (none of the baselines routes over UAVs)
  * the same top-k sparsification of activations / gradients as MTA-SFL
    (one shared setting for all methods, so differences come from routing,
    splitting and aggregation; MethodSpec.topk=False gives dense payloads)
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..cost import cost_model as cm
from ..env import channel as chm
from ..sim.runner import Simulation, a2a_rate
from .cost import add_overhead, device_tier1, forward_only_cost
from .trainer import BaselineJob, PerDeviceSplitTrainer, probe_moments


class BaselineSimulation(Simulation):
    initial_front_depth = 1

    # ================================================================== #
    #  construction                                                       #
    # ================================================================== #
    def _build_trainer(self):
        return PerDeviceSplitTrainer(self.model, self.data, self.sampler, self.cfg, self.device,
                                     front_depth=self.initial_front_depth)

    def _build_method(self):
        self.arms = None
        self._Lf_fast = self.initial_front_depth
        self.rng_base = np.random.default_rng(self.cfg.seed + 40_000)
        self.I = 1
        self.rounds_since_agg = 0
        self.synced_devs: set[int] = set()
        self.n_aggregations = 0
        self.decision_ms = 0.0
        self.moments = self._probe(n_dev=4)
        self.init_baseline()

    def init_baseline(self):
        """Subclass state."""

    def _method_meta(self) -> dict:
        return dict(n_arms=None, agent_overhead=dict(
            onboard_agents="none (decisions taken at the cloud / base station)",
            decision="see per-round 'timing.decision_ms'"))

    def _method_state(self) -> dict:
        return dict(I=self.I, rounds_since_agg=self.rounds_since_agg, synced=sorted(self.synced_devs),
                    n_aggregations=self.n_aggregations, moments=self.moments,
                    rng_base=self.rng_base.bit_generator.state, sub=self.baseline_state())

    def _load_method_state(self, s: dict):
        self.I, self.rounds_since_agg = s["I"], s["rounds_since_agg"]
        self.synced_devs = set(s["synced"])
        self.n_aggregations, self.moments = s["n_aggregations"], s["moments"]
        self.rng_base.bit_generator.state = s["rng_base"]
        self.load_baseline_state(s["sub"])

    def baseline_state(self) -> dict:
        return {}

    def load_baseline_state(self, s: dict):
        pass

    def _trainer_state(self) -> dict:
        return self.trainer.state()

    def _load_trainer_state(self, s: dict):
        self.trainer.load_state(s)

    def _agent_flops_energy(self, recs):
        return 0.0, 0.0

    # ================================================================== #
    #  helpers                                                            #
    # ================================================================== #
    @property
    def front_depth(self) -> int:
        return self.trainer.Lf if self.full else self._Lf_fast

    def _set_front_depth(self, Lf: int):
        if self.full:
            self.trainer.set_front_depth(Lf)
        self._Lf_fast = int(Lf)

    def _probe(self, n_dev: int) -> dict:
        """Gradient moments for the convergence bounds (AdaptSFL / SFLSC3)."""
        n_dev = max(2, n_dev)
        devs = list(self.rng_base.choice(self.cfg.topology.n_iot, size=n_dev, replace=False))
        shards = self.shards if self.full else None
        dev = self.device if self.full else "cpu"
        m = probe_moments(self.model, self.data, devs, shards, self.rng_base,
                          self.cfg.sfl.batch_per_device, dev)
        m["theta"] = math.log(self.data.num_classes)     # f(w0) - f* ~ ln C at initialisation
        return m

    def _path_rates(self, ctx, j, u, share):
        cfg, snap = self.cfg, ctx["snap"]
        snr = snap["snr_a2a"][j, u]
        r_ju = a2a_rate(snr, cfg.channel.w_a2a_ul_hz, share)
        r_uj = a2a_rate(snr * cfg.channel.w_a2a_ul_hz / cfg.channel.w_a2a_dl_hz, cfg.channel.w_a2a_dl_hz, share)
        return r_ju, r_uj, snap["bh_rate"][u]

    def param_bits(self, a: int, b: int) -> float:
        return cm.param_bits(self.prof, a, b, self.cfg.sfl.float_bits)

    # ================================================================== #
    #  Phase II/III: relay rule + method plan                             #
    # ================================================================== #
    def decide(self, ctx) -> list:
        net, snap, asc, alive = self.net, ctx["snap"], ctx["asc"], ctx["alive"]
        t0 = time.perf_counter()
        recs = []
        for j in range(net.P):
            devs = np.where(asc.a == j)[0]
            rec = dict(j=j, K=int(len(devs)), relay=None, split=None, status="inactive")
            recs.append(rec)
            if len(devs) == 0 or not alive[j]:
                continue
            base = net.u_adj(j, snap) & snap["cloud_ok"]
            prev = self.prev_relay[j]
            u = None
            if base.any():
                cand = np.where(base)[0]
                bott = [min(self._path_rates(ctx, j, c, 1)[0], snap["bh_rate"][c]) for c in cand]
                u = int(cand[int(np.argmax(bott))])
                rec.update(relay_life_s=float(snap["life"][j, u]), relay_qload=float(snap["q_load"][u]),
                           relay_cc_pre=float(snap["cc"][u]),
                           relay_snr_db=float(10 * np.log10(max(snap["snr_a2a"][j, u], 1e-12))))
            rec.update(devs=devs, n_adj=int(net.u_adj(j, snap).sum()), n_cloud=int(base.sum()),
                       n_feasible_relays=int(base.sum()), relay=u, prev_relay=prev,
                       churn=bool(u is not None and prev is not None and u != prev),
                       forced=bool(prev is not None and not base[prev]))
            if u is None:
                rec["status"] = "aborted_no_route"
        Q = np.zeros(net.U, int)
        for rec in recs:
            if rec.get("relay") is not None:
                Q[rec["relay"]] += 1
        ctx["Q"] = Q
        self.agg_round = (self.rounds_since_agg + 1) >= self.I
        self.plan(ctx, [r for r in recs if r.get("relay") is not None])
        self.agg_round = (self.rounds_since_agg + 1) >= self.I      # I may change in plan()
        for rec in recs:
            if rec.get("relay") is None:
                continue
            if len(rec["devs"]) == 0:          # no device of this cluster selected (S-HSFL)
                rec.update(status="inactive", relay=None, split=None, churn=False, forced=False)
                continue
            c = int(np.max(rec["cuts"]))
            rec["split"] = (c, c, c)          # 4-tier notation: UAVs forward only
            rec["is_bridge"] = False          # forwarding is not an MTA bridge decision
            rec.setdefault("extra", {}).update(cuts=[int(x) for x in rec["cuts"]],
                                               b=[int(x) for x in rec["bsz"]],
                                               n_fed=int(np.sum(rec.get("fed", 0))),
                                               I=int(self.I), agg_round=bool(self.agg_round))
        Q = np.zeros(net.U, int)               # relay load of the pipelines that actually run
        for rec in recs:
            if rec.get("relay") is not None:
                Q[rec["relay"]] += 1
        ctx["Q"] = Q
        self.decision_ms = 1e3 * (time.perf_counter() - t0)
        ctx["timing"]["decision_ms"] = self.decision_ms
        return recs

    def plan(self, ctx, recs):
        """Set rec['devs'] (participants), rec['cuts'], rec['bsz'], rec['fed']."""
        raise NotImplementedError

    def feasibility(self, ctx, devs, b) -> np.ndarray:
        """F[i, c]: device i is expected to meet the local tier-1 deadline with
        cut c and batch b (same straggler rule + 15 % margin as MTA-SFL's masks).
        Column 0 is unused (cuts are 1..L-1)."""
        cfg, net = self.cfg, self.net
        lim = cfg.sfl.iot_deadline_frac * cfg.t_max() / (1.0 + cfg.sfl.mask_margin)
        F = np.zeros((len(devs), self.L), bool)
        for c in range(1, self.L):
            tci, tmi, tgdl, _, _ = device_tier1(self.prof, np.full(len(devs), c), np.full(len(devs), b),
                                                net.f_iot[devs], ctx["r_ul_e"][devs], ctx["r_dl_e"][devs],
                                                cfg.sfl.float_bits, ratio=cfg.sfl.topk_ratio)
            F[:, c] = (tci + tmi + tgdl) <= lim
        return F

    def estimated_pipeline(self, ctx, rec, cuts, bsz, fed=None, active=None):
        """Expected training-stage cost of one pipeline (for the optimisers)."""
        j, u, devs = rec["j"], rec["relay"], rec["devs"]
        cuts, bsz = np.asarray(cuts), np.asarray(bsz)
        if active is not None:              # devices expected to be dropped do not delay the round
            devs, cuts, bsz = devs[active], cuts[active], bsz[active]
            fed = None if fed is None else np.asarray(fed)[active]
        r_ju, r_uj, r_bh = self._path_rates(ctx, j, u, ctx["Q"][u])
        return forward_only_cost(self.prof, cuts, bsz, self.net.f_iot[devs], ctx["r_ul_e"][devs],
                                 ctx["r_dl_e"][devs], r_ju, r_uj, r_bh, r_bh, self.cfg.hw.cloud_f,
                                 self.cfg, fed=fed)

    def estimated_ma(self, ctx, rec, up_bits, dn_bits, active=None):
        """Expected client-side model aggregation time of one pipeline."""
        j, u, devs = rec["j"], rec["relay"], rec["devs"]
        if active is not None:
            devs, up_bits = devs[active], np.asarray(up_bits)[active]
            if len(devs) == 0:
                return 0.0
        r_ju, r_uj, r_bh = self._path_rates(ctx, j, u, ctx["Q"][u])
        return cm.sync_cost(up_bits, dn_bits, 0.0, 0.0, ctx["r_ul_e"][devs], ctx["r_dl_e"][devs],
                            r_ju, r_uj, r_bh, r_bh, self.cfg.hw)[0]

    # ================================================================== #
    #  realised execution                                                 #
    # ================================================================== #
    def execute_pipeline(self, ctx, rec) -> bool:
        cfg, net = self.cfg, self.net
        T_MAX = cfg.t_max()
        w = cfg.sfl.float_bits
        g_min = chm.db2lin(cfg.channel.gamma_min_db)
        j, u, devs = rec["j"], rec["relay"], rec["devs"]
        cuts, bsz = np.asarray(rec["cuts"], int), np.asarray(rec["bsz"], float)
        fed = np.asarray(rec.get("fed", np.zeros(len(devs), bool)), bool)
        r_ul_r, r_dl_r, sinr_ul_r = ctx["r_ul_r"], ctx["r_dl_r"], ctx["sinr_ul_r"]
        dropped, attempt_e_iot = ctx["dropped"], ctx["attempt_e_iot"]
        ok_iot = net.iot_alive()[devs] & (sinr_ul_r[devs] >= g_min)
        tci, tmi, tgdl, bits, fl = device_tier1(self.prof, cuts, bsz, net.f_iot[devs], r_ul_r[devs],
                                                r_dl_r[devs], w, fed, ctx["jit_iot"][devs], cfg.sfl.topk_ratio)
        ok_dead = (tci + tmi + tgdl) <= cfg.sfl.iot_deadline_frac * T_MAX
        surv = ok_iot & ok_dead
        rec["drop_outage"] = int((~ok_iot).sum())
        rec["drop_deadline"] = int((ok_iot & ~ok_dead).sum())
        fail_idx = devs[~surv]
        attempt_e_iot[fail_idx] = (cfg.hw.kappa_iot * net.f_iot[fail_idx] ** 2 * fl[~surv]
                                   + cfg.hw.iot_ptx_w * tmi[~surv]
                                   + cfg.hw.iot_pcir_w * (tci[~surv] + tmi[~surv]))
        dropped[fail_idx] = True
        sdevs = devs[surv]
        rec["n_survive"] = int(len(sdevs))
        if not self._link_ok(ctx, j, u):
            rec["status"] = "link_failure"
            dropped[sdevs] = True
            return False
        if len(sdevs) == 0:
            rec["status"] = "no_survivors"
            return False
        snr = ctx["snr_a2a_r"][j, u]
        share = ctx["Q"][u]
        r_ju = a2a_rate(snr, cfg.channel.w_a2a_ul_hz, share)
        r_uj = a2a_rate(snr * cfg.channel.w_a2a_ul_hz / cfg.channel.w_a2a_dl_hz, cfg.channel.w_a2a_dl_hz, share)
        r_bh = ctx["snap"]["bh_rate"][u]
        sc, sb, sf = cuts[surv], bsz[surv], fed[surv]
        pc = forward_only_cost(self.prof, sc, sb, net.f_iot[sdevs], r_ul_r[sdevs], r_dl_r[sdevs],
                               r_ju, r_uj, r_bh, r_bh, cfg.hw.cloud_f, cfg, fed=sf,
                               jit_iot=ctx["jit_iot"][sdevs])
        cloud_flops = pc.cloud_flops
        # ---- model traffic: client-side aggregation / model download -----
        if cfg.sfl.model_sync:
            client_bits = np.array([self.param_bits(1, self.L if f_ else c) for c, f_ in zip(sc, sf)])
            if self.agg_round:
                up, dn = client_bits, float(client_bits.max())
                ts, es_i, es_j, es_u, sbits = cm.sync_cost(up, dn, 0.0, 0.0, r_ul_r[sdevs], r_dl_r[sdevs],
                                                           r_ju, r_uj, r_bh, r_bh, cfg.hw)
                add_overhead(pc, ts, es_i, es_j, es_u, sbits)
            else:
                need = np.array([d not in self.synced_devs for d in sdevs])
                if need.any():
                    dn = float(client_bits[need].max())
                    ts, es_n, es_j, es_u, sbits = cm.sync_cost(np.zeros(int(need.sum())), dn, 0.0, 0.0,
                                                               r_ul_r[sdevs][need], r_dl_r[sdevs][need],
                                                               r_ju, r_uj, r_bh, r_bh, cfg.hw)
                    es_i = np.zeros(len(sdevs))
                    es_i[need] = es_n
                    add_overhead(pc, ts, es_i, es_j, es_u, sbits)
        cloud_flops += self.extra_overhead(ctx, rec, sdevs, sc, sf, pc, r_ul_r[sdevs], r_dl_r[sdevs],
                                           r_ju, r_uj, r_bh)
        rec.update(pc=pc, sdevs=sdevs, rates=dict(r_ju=r_ju, r_uj=r_uj, r_bh=r_bh,
                                                  r_ul_min=float(r_ul_r[sdevs].min()),
                                                  r_ul_mean=float(r_ul_r[sdevs].mean()),
                                                  r_dl_min=float(r_dl_r[sdevs].min()),
                                                  r_dl_mean=float(r_dl_r[sdevs].mean())))
        rec["bits_tx"] = 2.0 * (float(np.sum(pc.bits["fwd1"])) + pc.bits["fwd2"] + pc.bits["fwd3"])
        rec["bits_dense"] = 2.0 * 3 * float(np.sum(np.where(sf, 0.0, sb * self.prof.act[sc] * w)))
        rec["cloud_flops"] = cloud_flops
        rec["sbsz"], rec["scuts"], rec["sfed"] = sb, sc, sf
        rec["status"] = "deadline_miss" if pc.t_total > T_MAX else "completed"
        return True

    def extra_overhead(self, ctx, rec, sdevs, cuts, fed, pc, r_ul, r_dl, r_ju, r_uj, r_bh) -> float:
        """Method-specific post-training overhead (S-HSFL verification).
        Returns extra cloud FLOPs."""
        return 0.0

    # ================================================================== #
    #  Phase IV / V                                                       #
    # ================================================================== #
    def train(self, ctx, executed):
        done = [r for r in executed if r["status"] == "completed"]
        st = None
        if self.full and done:
            jobs = [BaselineJob(r["j"], r["relay"], [int(i) for i in r["sdevs"]],
                                {int(i): int(b) for i, b in zip(r["sdevs"], r["sbsz"])},
                                {int(i): int(c) for i, c in zip(r["sdevs"], r["scuts"])},
                                {int(i) for i, f in zip(r["sdevs"], r["sfed"]) if f}) for r in done]
            st = self.trainer.train_round(jobs)
        ctx["st"] = st
        uploaders = {int(i) for r in done for i in r["sdevs"]}
        ctx["aggregated"] = 0
        if self.agg_round and uploaders:
            if self.full:
                ctx["aggregated"] = self.trainer.aggregate_from(uploaders)
            else:
                ctx["aggregated"] = len(uploaders)
            self.n_aggregations += 1
            self.rounds_since_agg = 0
            self.synced_devs = set(uploaders)
            self.after_aggregation(ctx)
        else:
            self.rounds_since_agg += 1
            self.synced_devs |= uploaders
        return st

    def after_aggregation(self, ctx):
        """Re-optimisation hook (AdaptSFL / SFLSC3 run their BCD here)."""

    def learn(self, ctx, recs):
        """Method-specific online updates (S-HSFL bandit statistics)."""

    def _method_round_fields(self, ctx) -> dict:
        d = dict(baseline=dict(I=int(self.I), Lf=int(self.front_depth), agg_round=bool(self.agg_round),
                               n_aggregated=int(ctx.get("aggregated", 0)),
                               n_aggregations=int(self.n_aggregations)))
        d["baseline"].update(self.round_fields(ctx))
        return d

    def round_fields(self, ctx) -> dict:
        return {}
