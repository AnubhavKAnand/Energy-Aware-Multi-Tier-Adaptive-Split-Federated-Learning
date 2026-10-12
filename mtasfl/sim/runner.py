"""
MTA-SFL joint execution (Algorithm 1).

Per round t
  Phase 0   failures / battery-swap returns
  Phase I   capacity-aware OFDMA association (Eq. 1), expected rates (Eq. 2-3)
  Phase II  DDQN relay selection per primary UAV on A = U_adj ∩ U_cloud ∩ U_safe
  Phase III LinUCB split selection per route on P_feasible (Eq. 31)
            realised execution: small-scale fading, compute jitter, IoT
            stragglers / outages, aerial link loss, deadline T_max (Eq. 17b)
  Phase IV  split training of completed pipelines + Eq. (4) aggregation
  Phase V   agent updates (Eq. 22-26, 32-34), battery update (Eq. 16),
            depletion -> offline + swap, mobility, logging of every metric
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import torch

from ..agents.ddqn import DDQNAgent, build_state
from ..agents.linucb import LinUCBAgent, context_bounds, D_CTX
from ..config import Config, export_param_table
from ..cost import cost_model as cm
from ..cost.vectorized import ArmEstimator, all_arms
from ..env import channel as chm
from ..env.network import Network
from ..learning.data import DATASET_INFO, DeviceSampler, load_dataset, partition
from ..learning.models import build_model
from ..learning.profiler import profile_model
from ..learning.split_trainer import PipelineJob, SplitTrainer
from ..methods import MethodSpec
from .logger import RunLogger, jain


def _nanmean(xs) -> float:
    v = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def a2a_rate(snr_full, w_hz, share):
    """Rate on an aerial link whose bandwidth W is shared by `share` pipelines
    (FDMA): W/Q log2(1 + SNR_W * Q)  (noise scales with the sub-band)."""
    q = max(int(share), 1)
    return (w_hz / q) * np.log2(1.0 + snr_full * q)


class Simulation:
    def __init__(self, cfg: Config, method: MethodSpec, out_dir: str, smoke: bool = False,
                 data_root: str = "./data", device: str = "cpu", print_every: int = 100,
                 ckpt_every: int = 500, resume: bool = True, tag: str = "", tagged: bool = False):
        if not method.topk:
            cfg = cfg.override({"sfl.topk_ratio": 1.0})
        self.cfg, self.method, self.smoke = cfg, method, smoke
        self.out_dir, self.device = out_dir, device
        self.print_every, self.ckpt_every = print_every, ckpt_every
        self.tag = tag or f"{cfg.dataset}/{method.name}/s{cfg.seed}"
        self.prefix = f"[{self.tag}] " if tagged else ""     # parallel workers: tag every line
        seed = cfg.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng_net = np.random.default_rng(seed)
        self.rng_fade = np.random.default_rng(seed + 10_000)
        self.rng_misc = np.random.default_rng(seed + 20_000)

        # ---------------- learning side --------------------------------
        ds = cfg.dataset
        in_shape, ncls, target = DATASET_INFO[ds]
        self.target_acc = target
        self.full = cfg.mode == "full"
        # fast mode needs only tensor shapes (model profile), never the data itself
        self.data = load_dataset(ds, data_root, smoke=smoke or not self.full)
        self.model = build_model(ds, ncls, in_shape)
        self.prof = profile_model(self.model, in_shape)
        self.L = self.prof.L
        if self.full:
            self.data.to(device)
            self.shards = partition(self.data.y_train.cpu().numpy(), cfg.topology.n_iot,
                                    cfg.sfl.partition, cfg.sfl.dirichlet_alpha, self.rng_misc)
            self.sampler = DeviceSampler(self.shards, np.random.default_rng(seed + 30_000))
            self.trainer = self._build_trainer()
            y = self.data.y_train.cpu().numpy()
            self.dev_label_dist = np.stack([np.bincount(y[s], minlength=ncls) / max(len(s), 1)
                                            for s in self.shards])

        # ---------------- network side ---------------------------------
        self.net = Network(cfg, self.rng_net)
        self._build_method()

        # ---------------- bookkeeping ----------------------------------
        self.prev_relay = [None] * self.net.P
        self.depleted = np.zeros(self.net.U, bool)
        self.participation = np.zeros(cfg.topology.n_iot)
        self.cum = dict(energy_uav=0.0, energy_iot=0.0, energy_prop=0.0, energy_comp=0.0,
                        energy_comm=0.0, energy_cir=0.0, regret=0.0, depletions=0,
                        churn=0, forced_reselect=0, aborted=0, deadline_miss=0, link_fail=0,
                        pipelines=0, completed=0, bridge=0, drops_iot=0, assoc_iot=0,
                        latency_viol=0, energy_viol=0, time_sim=0.0, samples=0)
        self.first_depletion = None
        self.best_acc, self.last_eval = 0.0, {}
        self.rounds_to_target = None
        self.time_to_target = self.energy_to_target = self.bits_to_target = None
        self.window = []
        self.start_round = 0

        # ---------------- resume ----------------------------------------
        ckpt = os.path.join(out_dir, "checkpoint.pt")
        resume_round = None
        if resume and os.path.exists(ckpt):
            resume_round = self._load_checkpoint(ckpt)
        self.log = RunLogger(out_dir, resume_round)
        if resume_round is None:
            if cfg.agent_pretrain_rounds > 0 and type(self) is Simulation:
                self._pretrain_agents(cfg.agent_pretrain_rounds)
            self._write_meta()

    # ================================================================== #
    #  method hooks (overridden by baselines, see mtasfl/baselines/)      #
    # ================================================================== #
    def _build_trainer(self):
        return SplitTrainer(self.model, self.data, self.sampler, self.cfg, self.device)

    def _build_method(self):
        """MTA-SFL: DDQN relay agents + LinUCB split agents (one per primary)."""
        cfg, method = self.cfg, self.method
        arms = all_arms(self.L, cfg.linucb.min_p1)
        if not method.allow_bridge:
            arms = arms[arms[:, 1] < arms[:, 2]]
        self.arms = arms
        self.est = ArmEstimator(self.prof, arms, cfg)
        self.ddqn = [DDQNAgent(j, self.net.U, cfg, cfg.seed, "cpu", method.pseudo_count)
                     for j in range(self.net.P)]
        bounds = context_bounds(cfg)
        self.lin = [LinUCBAgent(j, arms, cfg, bounds) for j in range(self.net.P)]

    def _pretrain_agents(self, rounds: int):
        """Train the DDQN / LinUCB agents offline in fast mode on an independent
        topology (seed + 1000), then deploy them here and keep learning online."""
        import contextlib
        import io
        t0 = time.time()
        pcfg = self.cfg.override({"mode": "fast", "seed": self.cfg.seed + 1000, "agent_pretrain_rounds": 0})
        pdir = os.path.join(self.out_dir, "pretrain")
        sim = Simulation(pcfg, self.method, pdir, smoke=True, resume=False, print_every=10 ** 9,
                         ckpt_every=10 ** 9)
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(rounds)
        self._load_method_state(sim._method_state())
        print(f"[pretrain] {self.tag}: agents pre-trained for {rounds} fast-mode rounds "
              f"on an independent topology in {time.time() - t0:.0f}s")

    def _method_meta(self) -> dict:
        ag = self.ddqn[0]
        return dict(n_arms=len(self.arms), agent_overhead=dict(
            ddqn=ag.memory_bytes(), ddqn_flops_per_decision=ag.flops_per_decision(),
            linucb=self.lin[0].memory_bytes(),
            linucb_flops_per_decision_all_arms=self.lin[0].flops_per_decision(),
            context_dim=D_CTX))

    def _method_state(self) -> dict:
        return dict(ddqn=[a.state_dict() for a in self.ddqn], lin=[a.state_dict() for a in self.lin])

    def _load_method_state(self, s: dict):
        for a, d in zip(self.ddqn, s["ddqn"]):
            a.load_state_dict(d)
        for a, d in zip(self.lin, s["lin"]):
            a.load_state_dict(d)

    def _trainer_state(self) -> dict:
        tr = self.trainer
        return dict(opt=tr.opt.state_dict(), rounds_trained=tr.rounds_trained,
                    ef={k: (v[0], v[1].detach().to("cpu", torch.float16), v[2])
                        for k, v in tr.ef.store.items()},
                    ef_round=tr.ef.round)

    def _load_trainer_state(self, s: dict):
        tr = self.trainer
        tr.opt.load_state_dict(s["opt"])
        tr.rounds_trained = s.get("rounds_trained", 0)
        tr.ef.store = {k: (v[0], v[1].to(self.device, torch.float32), v[2]) for k, v in s["ef"].items()}
        tr.ef.round = s.get("ef_round", 0)

    def _after_mobility(self, ctx):
        """Hook for methods that control UAV placement (no-op for MTA-SFL)."""

    def _method_round_fields(self, ctx) -> dict:
        """Extra per-round log fields of a method (none for MTA-SFL)."""
        return {}

    @staticmethod
    def _is_bridge(r) -> bool:
        return bool(r.get("is_bridge", r["split"][1] == r["split"][2]))

    def _agent_flops_energy(self, recs) -> tuple[float, float]:
        """Onboard decision-making compute per round (R1.8/R4.3)."""
        cfg, net = self.cfg, self.net
        agent_flops, e_agents = 0.0, 0.0
        for r in recs:
            if r.get("relay") is None and r["status"] == "inactive":
                continue
            ag = self.ddqn[r["j"]]
            fl = ag.flops_per_decision() * (1 + 5 * ag.dc.batch)
            if r.get("n_feasible_splits"):
                fl += self.lin[r["j"]].flops_per_decision(r["n_feasible_splits"])
            agent_flops += fl
            e_agents += cfg.hw.kappa_uav * net.f_uav[r["j"]] ** 2 * fl
        return agent_flops, e_agents

    # ================================================================== #
    def _write_meta(self):
        cfg = self.cfg
        self.log.write_json("config.json", cfg.to_dict())
        meta = dict(
            dataset=cfg.dataset, method=self.method.__dict__, seed=cfg.seed, smoke=self.smoke,
            t_max_s=cfg.t_max(), model=self.prof.name, L=self.L,
            profile=dict(fwd_flops=self.prof.fwd, bwd_flops=self.prof.bwd,
                         act_elems=self.prof.act, params=self.prof.n_params),
            **self._method_meta(),
            target_acc=self.target_acc,
            param_table=export_param_table(cfg),
            positions=dict(iot=self.net.iot_xy, hotspots=self.net.hotspots, gateway=self.net.gw_xyz),
        )
        self.log.write_json("meta.json", meta)

    # ================================================================== #
    def _ground(self, asc, rng=None):
        return self.net.ground_rates(asc, rng)

    def _est_inputs(self, j, u, devs, r_ul, r_dl, snap, share, pflight):
        net, cfg = self.net, self.cfg
        snr_full = snap["snr_a2a"][j, u]
        r_ju = a2a_rate(snr_full, cfg.channel.w_a2a_ul_hz, share)
        r_uj = a2a_rate(snr_full * cfg.channel.w_a2a_ul_hz / cfg.channel.w_a2a_dl_hz,
                        cfg.channel.w_a2a_dl_hz, share)
        r_bh = snap["bh_rate"][u]
        b = np.full(len(devs), float(cfg.sfl.batch_per_device))
        return dict(b=b, f_iot=net.f_iot[devs], r_ul_i=r_ul[devs], r_dl_i=r_dl[devs],
                    f_j=net.f_uav[j], r_ju=r_ju, r_uj=r_uj, f_u=net.f_uav[u],
                    r_bh_ul=r_bh, r_bh_dl=r_bh, f_s=cfg.hw.cloud_f,
                    e_res_i=net.E_iot[devs], e_res_j=net.E_uav[j], e_res_u=net.E_uav[u],
                    pflight_j=pflight[j], pflight_u=pflight[u])

    def _cost_vec(self, est, cc_u):
        oc = self.cfg.obj
        return oc.lam1 * est["t_norm"] + oc.lam2 * est["e_ratio"] - oc.lam3 * cc_u

    # ================================================================== #
    def step(self) -> dict:
        """One synchronous round (Algorithm 1).  Baselines override decide /
        execute_pipeline / train / learn; everything else is shared."""
        ctx = self._begin_round()
        recs = self.decide(ctx)
        executed = self.execute(ctx, recs)
        T_MAX = self.cfg.t_max()
        # synchronous round duration (slowest completed pipeline, capped by T_max)
        # + model synchronisation, which runs after training (outside T_max)
        if executed:
            t_round = min(max(r["pc"].t_total for r in executed), T_MAX)
            t_round += max(r["pc"].t_sync for r in executed)
        else:
            t_round = T_MAX
        self._score(ctx, executed, t_round)
        t_train0 = time.perf_counter()
        st = self.train(ctx, executed)
        ctx["timing"]["train_ms"] = 1e3 * (time.perf_counter() - t_train0)
        for r in executed:
            if r["status"] != "completed":
                ctx["dropped"][r["sdevs"]] = True
        t_upd0 = time.perf_counter()
        self.learn(ctx, recs)
        ctx["timing"]["agent_update_ms"] = 1e3 * (time.perf_counter() - t_upd0)
        return self._end_round(ctx, recs, executed, t_round, st)

    # ------------------------------------------------------------------ #
    def _begin_round(self) -> dict:
        cfg, net = self.cfg, self.net
        t = net.t
        ctx = dict(t=t, t_wall0=time.perf_counter(), timing=dict(ddqn_decide_ms=0.0, linucb_decide_ms=0.0))
        # ---------------- Phase 0: swaps & failures ----------------------
        returned = self.depleted & (net.failed_until <= t)
        net.E_uav[returned] = cfg.hw.uav_battery_j
        self.depleted[returned] = False
        injected = net.inject_failures()
        # ---------------- Phase I: association ---------------------------
        asc = net.associate()
        r_ul_e, r_dl_e, sinr_ul_e, _ = self._ground(asc)
        snap = net.snapshot()
        pflight = cm.flight_power(net.speeds(), net.is_primary, cfg)
        ctx.update(returned=returned, injected=injected, asc=asc, r_ul_e=r_ul_e, r_dl_e=r_dl_e,
                   sinr_ul_e=sinr_ul_e,
                   snap=snap, pflight=pflight, alive=snap["alive"])
        return ctx

    # ------------------------------------------------------------------ #
    def decide(self, ctx) -> list:
        """MTA-SFL Phase II (DDQN routing) and Phase III (LinUCB splitting)."""
        cfg, net, M = self.cfg, self.net, self.method
        asc, snap, pflight, alive = ctx["asc"], ctx["snap"], ctx["pflight"], ctx["alive"]
        r_ul_e, r_dl_e = ctx["r_ul_e"], ctx["r_dl_e"]
        # ---------------- Phase II: DDQN routing -------------------------
        recs = []
        cand_cache = {}
        for j in range(net.P):
            devs = np.where(asc.a == j)[0]
            rec = dict(j=j, K=int(len(devs)), relay=None, split=None, status="inactive")
            recs.append(rec)
            if len(devs) == 0 or not alive[j]:
                continue
            feat = build_state(j, snap, net, cfg)
            base = net.u_adj(j, snap) & snap["cloud_ok"]
            safe = np.zeros(net.U, bool)
            for u in np.where(base)[0]:
                share = snap["q_load"][u] + (0 if self.prev_relay[j] == u else 1)
                inp = self._est_inputs(j, u, devs, r_ul_e, r_dl_e, snap, share, pflight)
                est = self.est.estimate(**inp)
                cand_cache[(j, u)] = est
                safe[u] = est["feasible"].any()
            mask = base & safe
            ag = self.ddqn[j]
            ag.close_pending(feat, mask)
            if M.router == "random":
                u = int(self.rng_misc.choice(np.where(mask)[0])) if mask.any() else None
            else:
                u = ag.act(feat, mask)
            ctx["timing"]["ddqn_decide_ms"] += 1e3 * ag.t_decide
            prev = self.prev_relay[j]
            if u is not None:
                rec.update(relay_life_s=float(snap["life"][j, u]), relay_qload=float(snap["q_load"][u]),
                           relay_cc_pre=float(snap["cc"][u]),
                           relay_snr_db=float(10 * np.log10(max(snap["snr_a2a"][j, u], 1e-12))))
            rec.update(devs=devs, feat=feat, mask=mask, n_adj=int(net.u_adj(j, snap).sum()),
                       n_cloud=int(base.sum()), n_feasible_relays=int(mask.sum()),
                       relay=u, prev_relay=prev,
                       churn=bool(u is not None and prev is not None and u != prev),
                       forced=bool(prev is not None and not mask[prev]))
            if u is None:
                rec["status"] = "aborted_no_route"

        # actual relay load this round (Q_load for next round's state)
        Q = np.zeros(net.U, int)
        for rec in recs:
            if rec.get("relay") is not None:
                Q[rec["relay"]] += 1

        # ---------------- Phase III: LinUCB splitting --------------------
        for rec in recs:
            u = rec.get("relay")
            if u is None:
                continue
            j, devs = rec["j"], rec["devs"]
            inp = self._est_inputs(j, u, devs, r_ul_e, r_dl_e, snap, Q[u], pflight)
            est = self.est.estimate(**inp)
            i_star = devs[int(np.argmin(r_ul_e[devs]))]
            x_raw = np.array([net.E_iot[i_star], net.E_uav[j], net.E_uav[u],
                              net.f_iot[i_star], net.f_uav[j], net.f_uav[u],
                              r_ul_e[i_star], inp["r_ju"], inp["r_bh_ul"],
                              r_dl_e[i_star], inp["r_uj"], inp["r_bh_dl"]])
            lin = self.lin[j]
            x_bar = lin.normalise(x_raw)
            k = lin.act(x_bar, est["feasible"])
            ctx["timing"]["linucb_decide_ms"] += 1e3 * lin.t_decide
            rec.update(x_bar=x_bar, arm=k, n_feasible_splits=int(est["feasible"].sum()),
                       ucb_width=lin.last_width, alpha=lin.last_alpha)
            if k is None:
                rec["status"] = "aborted_no_split"
                continue
            rec["split"] = tuple(int(v) for v in self.arms[k])
            rec["est_T"] = float(est["T"][k])
            # ----- empirical regret vs expected-cost oracle (Eq. 19) -----
            best = np.inf
            for (jj, uu), ce in cand_cache.items():
                if jj != j or not rec["mask"][uu]:
                    continue
                c = self._cost_vec(ce, snap["cc"][uu])
                c = np.where(ce["feasible"], c, np.inf)
                best = min(best, float(c.min()))
            ce = cand_cache.get((j, u))
            if ce is not None and np.isfinite(best):
                cu = self._cost_vec(ce, snap["cc"][u])
                c_sel = float(cu[k])
                best_given_u = float(np.where(ce["feasible"], cu, np.inf).min())
                rec["oracle_cost"] = best
                rec["inst_regret"] = c_sel - best
                # decomposition of Eq. (50): routing regret + split (allocation) regret
                rec["regret_route"] = best_given_u - best if np.isfinite(best_given_u) else None
                rec["regret_split"] = c_sel - best_given_u if np.isfinite(best_given_u) else None

        ctx["Q"] = Q
        return recs

    # ------------------------------------------------------------------ #
    def _realise(self, ctx):
        """Draw the realised channel / compute state of this round (shared by
        all methods, so every method faces the same realisation process)."""
        net, cfg = self.net, self.cfg
        r_ul_r, r_dl_r, sinr_ul_r, sinr_dl_r = self._ground(ctx["asc"], self.rng_fade)
        self._sinr_r = (sinr_ul_r, sinr_dl_r)
        snr_a2a_r = net.a2a_snr(rng=self.rng_fade)
        sig = cfg.hw.compute_jitter_sigma
        jit_iot = self.rng_fade.lognormal(0, sig, net.N)
        jit_uav = self.rng_fade.lognormal(0, sig, net.U)
        ctx.update(r_ul_r=r_ul_r, r_dl_r=r_dl_r, sinr_ul_r=sinr_ul_r, sinr_dl_r=sinr_dl_r,
                   snr_a2a_r=snr_a2a_r, jit_iot=jit_iot, jit_uav=jit_uav,
                   dropped=np.zeros(net.N, bool), attempt_e_iot=np.zeros(net.N))

    def execute(self, ctx, recs) -> list:
        self._realise(ctx)
        executed = []
        for rec in recs:
            if rec.get("split") is None:
                if rec["status"] not in ("inactive",):
                    ctx["dropped"][rec.get("devs", [])] = True
                continue
            if self.execute_pipeline(ctx, rec):
                executed.append(rec)
        return executed

    def _link_ok(self, ctx, j, u) -> bool:
        cfg = self.cfg
        return bool(ctx["alive"][u] and not ctx["injected"][u]
                    and ctx["snr_a2a_r"][j, u] >= chm.db2lin(cfg.channel.gamma_th_db)
                    and self.rng_fade.random() >= cfg.failure.link_fade_prob)

    def execute_pipeline(self, ctx, rec) -> bool:
        """MTA-SFL realised execution of one pipeline (multi-tier split + top-k).
        Returns True if the pipeline reached the cloud (completed or deadline miss)."""
        cfg, net = self.cfg, self.net
        T_MAX = cfg.t_max()
        g_min = chm.db2lin(cfg.channel.gamma_min_db)
        b = cfg.sfl.batch_per_device
        r_ul_r, r_dl_r, sinr_ul_r = ctx["r_ul_r"], ctx["r_dl_r"], ctx["sinr_ul_r"]
        jit_iot, jit_uav, snap = ctx["jit_iot"], ctx["jit_uav"], ctx["snap"]
        dropped, attempt_e_iot = ctx["dropped"], ctx["attempt_e_iot"]
        j, u, p, devs = rec["j"], rec["relay"], rec["split"], rec["devs"]
        ok_iot = net.iot_alive()[devs] & (sinr_ul_r[devs] >= g_min)
        tci, tmi, bits1, fl1 = cm.iot_upload(self.prof, p[0], np.full(len(devs), b), net.f_iot[devs],
                                             r_ul_r[devs], cfg.sfl.topk_ratio, cfg.sfl.float_bits,
                                             jit_iot[devs])
        tgdl = bits1 / np.maximum(r_dl_r[devs], 1e-9)
        # tier-1 straggler test: compute + upload + gradient return vs local deadline
        ok_dead = (tci + tmi + tgdl) <= cfg.sfl.iot_deadline_frac * T_MAX
        surv = ok_iot & ok_dead
        rec["drop_outage"] = int((~ok_iot).sum())
        rec["drop_deadline"] = int((ok_iot & ~ok_dead).sum())
        # stragglers / outages still spent compute + transmission energy
        fail_idx = devs[~surv]
        attempt_e_iot[fail_idx] = (cfg.hw.kappa_iot * net.f_iot[fail_idx] ** 2 * fl1[~surv]
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
        snr_a2a_r = ctx["snr_a2a_r"]
        share = ctx["Q"][u]
        r_ju = a2a_rate(snr_a2a_r[j, u], cfg.channel.w_a2a_ul_hz, share)
        r_uj = a2a_rate(snr_a2a_r[j, u] * cfg.channel.w_a2a_ul_hz / cfg.channel.w_a2a_dl_hz,
                        cfg.channel.w_a2a_dl_hz, share)
        pc = cm.pipeline_cost(self.prof, p, np.full(len(sdevs), float(b)), net.f_iot[sdevs],
                              r_ul_r[sdevs], r_dl_r[sdevs], net.f_uav[j], r_ju, r_uj,
                              net.f_uav[u], snap["bh_rate"][u], snap["bh_rate"][u],
                              cfg.hw.cloud_f, cfg,
                              jitter=dict(iot=jit_iot[sdevs], j=jit_uav[j], u=jit_uav[u]))
        rec.update(pc=pc, sdevs=sdevs, rates=dict(r_ju=r_ju, r_uj=r_uj, r_bh=snap["bh_rate"][u],
                                                  r_ul_min=float(r_ul_r[sdevs].min()),
                                                  r_ul_mean=float(r_ul_r[sdevs].mean()),
                                                  r_dl_min=float(r_dl_r[sdevs].min()),
                                                  r_dl_mean=float(r_dl_r[sdevs].mean())))
        Bp = float(len(sdevs) * b)
        w = cfg.sfl.float_bits
        # forward + backward payload on all 3 hops (IoT-j, j-u, u-S) vs dense
        rec["bits_tx"] = 2.0 * (float(np.sum(pc.bits["fwd1"])) + pc.bits["fwd2"] + pc.bits["fwd3"])
        rec["bits_dense"] = 2.0 * Bp * w * (self.prof.act[p[0]] + self.prof.act[p[1]] + self.prof.act[p[2]])
        rec["cloud_flops"] = (self.prof.cfp(p[2] + 1, self.L) + self.prof.cbp(p[2] + 1, self.L)) * Bp
        rec["status"] = "deadline_miss" if pc.t_total > T_MAX else "completed"
        return True

    # ------------------------------------------------------------------ #
    def _score(self, ctx, executed, t_round):
        """Eq. (10)-(17): finalise energy and evaluate the global objective
        (logged for every method; DDQN/LinUCB rewards used by MTA-SFL only)."""
        cfg, net, snap, pflight = self.cfg, self.net, ctx["snap"], ctx["pflight"]
        for rec in executed:
            pc = rec["pc"]
            t_prop = t_round if cfg.obj.prop_time == "round" else pc.t_total
            cm.finalize(pc, t_prop, pflight[rec["j"]], pflight[rec["relay"]], cfg)
            tn, er, vt, ve = cm.objective_terms(pc, net.E_iot[rec["sdevs"]], net.E_uav[rec["j"]],
                                                net.E_uav[rec["relay"]], cfg)
            cc_u = float(snap["cc"][rec["relay"]])
            rec.update(t_norm=tn, e_ratio=er, viol_t=bool(vt), viol_e=bool(ve), cc=cc_u,
                       cost=cm.global_cost(tn, er, cc_u, cfg),
                       r_ext=cm.ddqn_extrinsic_reward(tn, er, cc_u, vt or ve, cfg),
                       r_lin=cm.linucb_reward(tn, er, cfg))

    def train(self, ctx, executed):
        """Phase IV: split training of completed pipelines + Eq. (4)."""
        jobs = [PipelineJob(r["j"], r["relay"], r["split"], [int(i) for i in r["sdevs"]])
                for r in executed if r["status"] == "completed"]
        if self.full and jobs:
            return self.trainer.train_round(jobs)
        return None

    def learn(self, ctx, recs):
        """Phase V: DDQN (Eq. 22-26) and LinUCB (Eq. 32-34) updates."""
        cfg, snap = self.cfg, ctx["snap"]
        psi = cfg.obj.psi_penalty
        for rec in recs:
            j, u = rec["j"], rec.get("relay")
            if u is None:
                continue
            ag = self.ddqn[j]
            if "r_ext" in rec:
                r_ext, terminal = rec["r_ext"], rec["status"] != "completed"
            else:   # route chosen but no split / link failure / no survivors
                r_ext, terminal = rec.get("cc", float(snap["cc"][u])) * cfg.obj.lam3 - psi, True
                rec["r_ext"] = r_ext
            r_int = ag.intrinsic(rec["feat"], u)
            rec["r_int"] = r_int
            ag.record(rec["feat"], u, r_ext + r_int, terminal)
            ag.train_step()
            rec["ddqn_loss"] = ag.last_loss
            rec["ddqn_q"] = ag.last_q
            if "r_lin" in rec:                          # executed: observe reward
                self.lin[j].update(rec["arm"], rec["x_bar"], rec["r_lin"])

    # ------------------------------------------------------------------ #
    def _end_round(self, ctx, recs, executed, t_round, st) -> dict:
        cfg, net = self.cfg, self.net
        t, asc, snap, pflight = ctx["t"], ctx["asc"], ctx["snap"], ctx["pflight"]
        dropped, attempt_e_iot = ctx["dropped"], ctx["attempt_e_iot"]
        Q = ctx.get("Q", np.zeros(net.U, int))
        # ---------------- battery update (Eq. 16) -------------------------
        e_uav = np.zeros(net.U)
        alive_now = net.uav_alive()
        e_uav[alive_now] = (pflight[alive_now] + cfg.hw.uav_pcir_w) * t_round
        e_comp_uav = e_comm_uav = 0.0
        e_iot = cfg.hw.iot_psleep_w * t_round * np.ones(net.N)
        e_iot += attempt_e_iot
        for rec in executed:
            pc, j, u = rec["pc"], rec["j"], rec["relay"]
            e_uav[j] += pc.ecomp_j + pc.ecomm_j
            e_uav[u] += pc.ecomp_u + pc.ecomm_u
            e_comp_uav += pc.ecomp_j + pc.ecomp_u
            e_comm_uav += pc.ecomm_j + pc.ecomm_u
            e_iot[rec["sdevs"]] = pc.e_total_i
        net.E_uav -= e_uav
        net.E_iot = np.maximum(net.E_iot - e_iot, 0.0)
        crit = cfg.hw.critical_energy_frac * cfg.hw.uav_battery_j
        newly = (net.E_uav <= crit) & ~self.depleted
        if newly.any():
            self.depleted |= newly
            net.failed_until[newly] = t + 1 + int(np.ceil(cfg.hw.recharge_time_s / cfg.t_max()))
            net.E_uav[newly] = np.maximum(net.E_uav[newly], 0.0)
            if self.first_depletion is None:
                self.first_depletion = t

        # ---------------- bookkeeping, mobility --------------------------
        for rec in recs:
            if rec["status"] != "inactive" or rec.get("relay") is not None:
                self.prev_relay[rec["j"]] = rec.get("relay")
        assoc = asc.a >= 0
        net.prev_dropped = dropped | ~assoc
        part = np.zeros(net.N, bool)
        for r in executed:
            if r["status"] == "completed":
                part[r["sdevs"]] = True
        self.participation += part
        net.q_load = Q.astype(float)
        net.step_mobility(dt=max(t_round, 0.5))
        self._after_mobility(ctx)
        net.t += 1

        # ---------------- metrics ----------------------------------------
        timing = dict(ctx["timing"])
        timing["round_ms"] = 1e3 * (time.perf_counter() - ctx["t_wall0"])
        rec = self._round_record(t, recs, executed, t_round, st, asc, part, dropped,
                                 e_uav, e_iot, e_comp_uav, e_comm_uav, pflight, newly,
                                 ctx["returned"], ctx["injected"], snap, timing)
        extra = self._method_round_fields(ctx)
        if extra:
            rec.update(extra)
        return rec

    # ================================================================== #
    def _round_record(self, t, recs, executed, t_round, st, asc, part, dropped, e_uav, e_iot,
                      e_comp_uav, e_comm_uav, pflight, newly, returned, injected, snap, timing):
        cfg, net, c = self.cfg, self.net, self.cum
        sinr_r = self._sinr_r
        alive = net.uav_alive()
        e_prop = float(((pflight + cfg.hw.uav_pcir_w) * t_round)[alive | newly].sum())
        e_prop_only = float((pflight * t_round)[alive | newly].sum())
        pipes = []
        for r in recs:
            d = dict(j=r["j"], K=r["K"], status=r["status"], relay=r.get("relay"),
                     prev_relay=r.get("prev_relay"), split=r.get("split"),
                     bridge=bool(r.get("split") and self._is_bridge(r)),
                     n_adj=r.get("n_adj"), n_cloud=r.get("n_cloud"),
                     n_feasible_relays=r.get("n_feasible_relays"),
                     n_feasible_splits=r.get("n_feasible_splits"),
                     churn=r.get("churn"), forced_reselect=r.get("forced"),
                     n_survive=r.get("n_survive"), drop_outage=r.get("drop_outage"),
                     drop_deadline=r.get("drop_deadline"),
                     est_T=r.get("est_T"), cc=r.get("cc"), t_norm=r.get("t_norm"),
                     e_ratio=r.get("e_ratio"), cost=r.get("cost"), viol_t=r.get("viol_t"),
                     viol_e=r.get("viol_e"), r_ext=r.get("r_ext"), r_int=r.get("r_int"),
                     r_lin=r.get("r_lin"), inst_regret=r.get("inst_regret"),
                     oracle_cost=r.get("oracle_cost"), regret_route=r.get("regret_route"),
                     regret_split=r.get("regret_split"), ucb_width=r.get("ucb_width"),
                     alpha=r.get("alpha"), ddqn_loss=r.get("ddqn_loss"), ddqn_q=r.get("ddqn_q"),
                     rates=r.get("rates"), arm=r.get("arm"), x_bar=r.get("x_bar"),
                     relay_life_s=r.get("relay_life_s"), relay_qload=r.get("relay_qload"),
                     relay_snr_db=r.get("relay_snr_db"), bits_tx=r.get("bits_tx"),
                     bits_dense=r.get("bits_dense"))
            if "extra" in r:
                d["extra"] = r["extra"]
            if "pc" in r:
                pc = r["pc"]
                if pc.t_sync > 0:
                    d.update(T_sync=pc.t_sync, bits_sync=pc.bits.get("sync", 0.0))
                d.update(T_total=pc.t_total, T_iot_max=float(pc.t_iot.max()),
                         Tcomp_j=pc.tcomp_j, Tcomm_j=pc.tcomm_j, Tcomp_u=pc.tcomp_u,
                         Tcomm_u=pc.tcomm_u, Tcomp_s=pc.tcomp_s, Tcomm_s=pc.tcomm_s,
                         E_j=pc.e_total_j, E_u=pc.e_total_u, E_iot=float(pc.e_total_i.sum()),
                         E_parts=pc.e_parts, bits=dict(fwd1=float(np.sum(pc.bits["fwd1"])),
                                                       fwd2=pc.bits["fwd2"], fwd3=pc.bits["fwd3"]),
                         E_res_relay=float(net.E_uav[r["relay"]]))
            pipes.append(d)

        n_pipe = sum(1 for r in recs if r["status"] != "inactive")
        done = [r for r in executed if r["status"] == "completed"]
        bits_tx = float(sum(r["bits_tx"] for r in done))
        bits_dense = float(sum(r["bits_dense"] for r in done))
        bits_sync = float(sum(r["pc"].bits.get("sync", 0.0) for r in done))
        e_cloud = float(sum(r["cloud_flops"] for r in done)) * cfg.hw.cloud_j_per_flop
        # onboard agent compute (decision + update) -> energy, R1.8/R4.3
        agent_flops, e_agents = self._agent_flops_energy(recs)
        on = asc.a >= 0
        if on.any():
            su = 10 * np.log10(np.maximum(sinr_r[0][on], 1e-12))
            sd = 10 * np.log10(np.maximum(sinr_r[1][on], 1e-12))
            sinr_stats = (float(su.mean()), float(np.percentile(su, 10)),
                          float(sd.mean()), float(np.percentile(sd, 10)))
        else:
            sinr_stats = (None, None, None, None)
        n_comp = sum(1 for r in executed if r["status"] == "completed")
        n_bridge = sum(1 for r in executed if self._is_bridge(r))
        regrets = [r["inst_regret"] for r in recs if r.get("inst_regret") is not None]
        n_assoc = int((asc.a >= 0).sum())
        n_drop = int(dropped.sum())
        e_uav_round = float(e_uav.sum())
        e_iot_round = float(e_iot.sum())
        e_comp_iot = sum(float(r["pc"].ecomp_i.sum()) for r in executed)
        e_comm_iot = sum(float(r["pc"].ecomm_i.sum()) for r in executed)
        churn = sum(1 for r in recs if r.get("churn"))
        forced = sum(1 for r in recs if r.get("forced"))
        aborted = sum(1 for r in recs if str(r["status"]).startswith("aborted"))
        c.update(energy_uav=c["energy_uav"] + e_uav_round, energy_iot=c["energy_iot"] + e_iot_round,
                 energy_prop=c["energy_prop"] + e_prop_only,
                 energy_comp=c["energy_comp"] + e_comp_uav + e_comp_iot,
                 energy_comm=c["energy_comm"] + e_comm_uav + e_comm_iot,
                 regret=c["regret"] + float(np.sum(regrets)) if regrets else c["regret"],
                 depletions=c["depletions"] + int(newly.sum()), churn=c["churn"] + churn,
                 forced_reselect=c["forced_reselect"] + forced, aborted=c["aborted"] + aborted,
                 deadline_miss=c["deadline_miss"] + sum(1 for r in executed if r["status"] == "deadline_miss"),
                 link_fail=c["link_fail"] + sum(1 for r in recs if r["status"] == "link_failure"),
                 pipelines=c["pipelines"] + n_pipe, completed=c["completed"] + n_comp,
                 bridge=c["bridge"] + n_bridge, drops_iot=c["drops_iot"] + n_drop,
                 assoc_iot=c["assoc_iot"] + n_assoc,
                 latency_viol=c["latency_viol"] + sum(1 for r in executed if r.get("viol_t")),
                 energy_viol=c["energy_viol"] + sum(1 for r in executed if r.get("viol_e")),
                 time_sim=c["time_sim"] + t_round, samples=c["samples"] + (st.n_samples if st else 0),
                 bits_tx=c.get("bits_tx", 0.0) + bits_tx, energy_cloud=c.get("energy_cloud", 0.0) + e_cloud,
                 bits_total=c.get("bits_total", 0.0) + bits_tx + bits_sync,
                 energy_agents=c.get("energy_agents", 0.0) + e_agents)
        rec = dict(
            round=t, t_round=t_round, sim_time_s=c["time_sim"],
            # --- latency
            T_total_mean=float(np.mean([r["pc"].t_total for r in executed])) if executed else None,
            T_total_max=float(np.max([r["pc"].t_total for r in executed])) if executed else None,
            # --- energy (this round)
            energy_swarm_uav_J=e_uav_round, energy_iot_J=e_iot_round,
            energy_total_J=e_uav_round + e_iot_round,
            energy_prop_J=e_prop_only, energy_hover_cir_J=e_prop,
            energy_comp_J=e_comp_uav + e_comp_iot, energy_comm_J=e_comm_uav + e_comm_iot,
            energy_comp_uav_J=e_comp_uav, energy_comm_uav_J=e_comm_uav,
            energy_excl_prop_J=e_uav_round + e_iot_round - e_prop_only,
            # --- battery / lifetime
            uav_alive=int(alive.sum()), uav_depleted_now=int(newly.sum()),
            uav_returned_now=int(returned.sum()), uav_failed_injected=int(injected.sum()),
            uav_Eres_min=float(net.E_uav.min()), uav_Eres_mean=float(net.E_uav.mean()),
            relay_Eres_min=float(net.E_uav[~net.is_primary].min()),
            iot_Eres_min=float(net.E_iot.min()), iot_Eres_mean=float(net.E_iot.mean()),
            # --- association / straggler / fairness
            n_assoc=n_assoc, n_unassoc=int(net.N - n_assoc), n_dropped=n_drop,
            n_participating=int(part.sum()),
            drop_rate=n_drop / max(n_assoc, 1),
            assoc_load_jain=jain(asc.K[asc.K > 0]) if (asc.K > 0).any() else None,
            participation_jain=jain(self.participation),
            # --- pipelines / constraints
            n_pipelines=n_pipe, n_completed=n_comp, n_aborted=aborted,
            n_deadline_miss=sum(1 for r in executed if r["status"] == "deadline_miss"),
            n_link_fail=sum(1 for r in recs if r["status"] == "link_failure"),
            latency_violations=sum(1 for r in executed if r.get("viol_t")),
            energy_violations=sum(1 for r in executed if r.get("viol_e")),
            n_bridge=n_bridge,
            cost_mean=float(np.mean([r["cost"] for r in executed])) if executed else None,
            # --- routing resilience
            # --- communication volume (accuracy-vs-communication, R1.5)
            bits_tx=bits_tx, bits_dense=bits_dense, cum_bits_tx=c["bits_tx"],
            compression_ratio=(bits_tx / bits_dense) if bits_dense > 0 else None,
            bits_sync=bits_sync, bits_total=bits_tx + bits_sync, cum_bits_total=c["bits_total"],
            t_sync_max=max((r["pc"].t_sync for r in executed), default=0.0),
            # --- PHY: realised ground SINR (interference model, R4.1)
            sinr_ul_db_mean=sinr_stats[0], sinr_ul_db_p10=sinr_stats[1],
            sinr_dl_db_mean=sinr_stats[2], sinr_dl_db_p10=sinr_stats[3],
            # --- reported-only energy outside the objective boundary (R2.5, R3.C7)
            energy_cloud_comp_J=e_cloud, energy_agents_J=e_agents, agent_flops=agent_flops,
            churn=churn, forced_reselect=forced, beacon_bytes=net.beacon_bytes_round,
            mean_degree=float(snap["A"].sum(1).mean()), cc_mean=float(snap["cc"].mean()),
            # --- learning-agent behaviour
            inst_regret_sum=float(np.sum(regrets)) if regrets else None,
            regret_route_sum=float(np.nansum([r.get("regret_route") or 0.0 for r in recs])),
            regret_split_sum=float(np.nansum([r.get("regret_split") or 0.0 for r in recs])),
            cum_regret=c["regret"],
            timing=timing,
            pipelines=pipes,
        )
        if st is not None:
            rec.update(train_loss=st.loss, train_acc=st.train_acc, n_samples=st.n_samples,
                       grad_norm=st.grad_norm, fwd_comp_err=st.fwd_comp_err,
                       bwd_comp_err=st.bwd_comp_err, ef_resets=st.ef_resets, lr=st.lr,
                       ef_evicted=st.ef_evicted, ef_mem_bytes=st.ef_bytes)
        return rec

    # ================================================================== #
    def evaluate(self, t) -> dict:
        ev = self.trainer.evaluate()
        pc = ev["per_class_acc"]
        dev_acc = self.dev_label_dist @ pc               # accuracy on each device's distribution
        srt = np.sort(dev_acc)
        out = dict(round=t, sim_time_s=self.cum["time_sim"], samples=self.cum["samples"],
                   test_acc=ev["test_acc"], test_loss=ev["test_loss"], per_class_acc=pc,
                   device_acc_mean=float(dev_acc.mean()), device_acc_std=float(dev_acc.std()),
                   device_acc_worst10=float(srt[:max(1, len(srt) // 10)].mean()),
                   device_acc_jain=jain(dev_acc),
                   cum_energy_uav_J=self.cum["energy_uav"], cum_energy_iot_J=self.cum["energy_iot"])
        if ev["test_acc"] > self.best_acc:
            self.best_acc = ev["test_acc"]
        if self.rounds_to_target is None and ev["test_acc"] >= self.target_acc:
            self.rounds_to_target = t
            self.time_to_target = self.cum["time_sim"]
            self.energy_to_target = self.cum["energy_uav"] + self.cum["energy_iot"]
            self.bits_to_target = self.cum.get("bits_total", 0.0)
            out["reached_target"] = True
        self.last_eval = out
        return out

    # ================================================================== #
    def summary(self, t_done, finished=False) -> dict:
        c = self.cum
        n = max(t_done, 1)
        return dict(
            dataset=self.cfg.dataset, method=self.method.name, seed=self.cfg.seed,
            partition=self.cfg.sfl.partition, rounds_done=t_done, finished=finished,
            mean_round_latency_s=c["time_sim"] / n,
            total_energy_uav_J=c["energy_uav"], total_energy_iot_J=c["energy_iot"],
            total_energy_J=c["energy_uav"] + c["energy_iot"],
            total_energy_prop_J=c["energy_prop"],
            total_energy_excl_prop_J=c["energy_uav"] + c["energy_iot"] - c["energy_prop"],
            energy_comp_J=c["energy_comp"], energy_comm_J=c["energy_comm"],
            energy_per_round_J=(c["energy_uav"] + c["energy_iot"]) / n,
            final_test_acc=self.last_eval.get("test_acc"), best_test_acc=self.best_acc,
            final_test_loss=self.last_eval.get("test_loss"),
            device_acc_jain=self.last_eval.get("device_acc_jain"),
            device_acc_worst10=self.last_eval.get("device_acc_worst10"),
            target_acc=self.target_acc, rounds_to_target=self.rounds_to_target,
            sim_time_to_target_s=self.time_to_target, energy_to_target_J=self.energy_to_target,
            bits_to_target=self.bits_to_target, total_bits_tx=c.get("bits_tx", 0.0),
            total_bits_sync=c.get("bits_total", 0.0) - c.get("bits_tx", 0.0),
            total_bits=c.get("bits_total", 0.0),
            total_energy_cloud_comp_J=c.get("energy_cloud", 0.0),
            total_energy_agents_J=c.get("energy_agents", 0.0),
            completion_rate=c["completed"] / max(c["pipelines"], 1),
            drop_rate=c["drops_iot"] / max(c["assoc_iot"], 1),
            latency_violation_rate=c["latency_viol"] / max(c["pipelines"], 1),
            energy_violation_rate=c["energy_viol"] / max(c["pipelines"], 1),
            deadline_miss=c["deadline_miss"], link_failures=c["link_fail"], aborted=c["aborted"],
            bridge_rate=c["bridge"] / max(c["completed"] + c["deadline_miss"], 1),
            uav_depletions=c["depletions"], first_depletion_round=self.first_depletion,
            churn_per_round=c["churn"] / n, forced_reselect=c["forced_reselect"],
            cum_regret=c["regret"], avg_regret=c["regret"] / n,
            participation_jain=jain(self.participation),
        )

    # ================================================================== #
    def run(self, rounds: int):
        hdr = (f"{'round':>6} {'acc':>6} {'loss':>7} {'T_rnd':>6} {'E_rnd(kJ)':>9} {'drop%':>6} "
               f"{'viol%':>6} {'bridge%':>7} {'regret':>7} {'dead':>4} {'s/rnd':>6}")
        print(f"\n=== {self.tag} | rounds {self.start_round}->{rounds} | T_max={self.cfg.t_max()}s "
              f"| arms={len(self.arms) if self.arms is not None else '-'} | out={self.out_dir}")
        print(self.prefix + hdr)
        w0 = time.perf_counter()
        for t in range(self.start_round, rounds):
            rec = self.step()
            if self.full and ((t + 1) % self.cfg.sfl.eval_every == 0 or t == rounds - 1):
                ev = self.evaluate(t)
                self.log.log_eval(ev)
                rec["test_acc"] = ev["test_acc"]
                rec["test_loss"] = ev["test_loss"]
            self.log.log_round(rec)
            self.window.append(rec)
            if (t + 1) % self.print_every == 0 or t == rounds - 1:
                self._print(t, w0)
                w0 = time.perf_counter()
                self.window = []
                self.log.write_json("summary.json", self.summary(t + 1))
            if (t + 1) % self.ckpt_every == 0:
                self._save_checkpoint(t + 1)
        s = self.summary(rounds, finished=True)
        self.log.write_json("summary.json", s)
        self._save_checkpoint(rounds)
        self.log.close()
        return s

    def _print(self, t, w0):
        W = self.window
        if not W:
            return
        npipe = max(sum(r["n_pipelines"] for r in W), 1)
        viol = sum(r["latency_violations"] + r["energy_violations"] + r["n_aborted"] for r in W) / npipe
        ncomp = max(sum(r["n_completed"] + r["n_deadline_miss"] for r in W), 1)
        acc = self.last_eval.get("test_acc", float("nan"))
        loss = _nanmean([r.get("train_loss") for r in W]) if self.full else float("nan")
        print(f"{self.prefix}{t + 1:>6} {acc:>6.3f} {loss:>7.4f} {np.mean([r['t_round'] for r in W]):>6.2f} "
              f"{np.mean([r['energy_total_J'] for r in W]) / 1e3:>9.3f} "
              f"{100 * np.mean([r['drop_rate'] for r in W]):>6.1f} {100 * viol:>6.1f} "
              f"{100 * sum(r['n_bridge'] for r in W) / ncomp:>7.1f} "
              f"{_nanmean([r['inst_regret_sum'] for r in W]):>7.4f} "
              f"{self.cum['depletions']:>4d} {(time.perf_counter() - w0) / len(W):>6.3f}", flush=True)

    # ================================================================== #
    def _save_checkpoint(self, t_next):
        state = dict(
            t_next=t_next, cum=self.cum, prev_relay=self.prev_relay, depleted=self.depleted,
            participation=self.participation, first_depletion=self.first_depletion,
            best_acc=self.best_acc, last_eval=self.last_eval, rounds_to_target=self.rounds_to_target,
            to_target=(self.time_to_target, self.energy_to_target, self.bits_to_target),
            net=pickle.dumps(self.net),
            rngs=pickle.dumps((self.rng_fade, self.rng_misc)),
            method_state=self._method_state(),
            torch_rng=torch.get_rng_state(),
        )
        if self.full:
            state.update(model=self.model.state_dict(), trainer=self._trainer_state(),
                         sampler=pickle.dumps(self.sampler))
        tmp = os.path.join(self.out_dir, "checkpoint.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(self.out_dir, "checkpoint.pt"))

    def _load_checkpoint(self, path) -> int:
        s = torch.load(path, map_location="cpu", weights_only=False)
        self.cum, self.prev_relay, self.depleted = s["cum"], s["prev_relay"], s["depleted"]
        self.participation, self.first_depletion = s["participation"], s["first_depletion"]
        self.best_acc, self.last_eval, self.rounds_to_target = s["best_acc"], s["last_eval"], s["rounds_to_target"]
        self.time_to_target, self.energy_to_target, self.bits_to_target = s.get("to_target", (None, None, None))
        self.net = pickle.loads(s["net"])
        self.rng_fade, self.rng_misc = pickle.loads(s["rngs"])
        self._load_method_state(s["method_state"])
        torch.set_rng_state(s["torch_rng"])
        if self.full:
            self.model.load_state_dict(s["model"])
            self._load_trainer_state(s["trainer"])
            self.sampler = pickle.loads(s["sampler"])
            self.trainer.sampler = self.sampler
        self.start_round = s["t_next"]
        print(f"[resume] {self.tag} from round {self.start_round}")
        return self.start_round
