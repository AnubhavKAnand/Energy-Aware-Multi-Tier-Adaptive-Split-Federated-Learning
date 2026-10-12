"""
S-HSFL: A Game-Theoretic Enhanced Secure-Hybrid Split-Federated Learning Scheme
for UAV-Assisted Wireless Networks (Q. Gao, X. Zhang, G. Dong, B. Tang, J. Liu,
Drones 10(1):37, 2026).

Implemented components
  * MAB-GT user-equipment selection (Alg. 1):
      - signature verification V_t and trust scores T[n]           (lines 3-4, Eq. 8-9, 14-15)
      - non-cooperative self-selection  U_n = beta_g SNR_n + (1-beta_g)||dw_n|| - alpha E_n >= 0
                                                                     (line 5-6, Eq. 3)
      - coalitions by K-means on [SNR, ||dw||] every T_coalition rounds and
        Shapley contributions phi_n                                 (lines 7-9, Eq. 2, 4-5)
        (v(S) of Eq. 5 is additive, so the Shapley value is exact:
         phi_n = SNR_n + ||dw_n||; no Monte Carlo needed)
      - UCB score  A_n = beta Omega_n + (1-beta) Gamma_n + sigma sqrt(2 ln t / c_n)
                         + gamma_sh phi_n + delta H_n,  A_n <- A_n T[n]     (lines 14-15, Eq. 10-11)
      - top-K selection, EMA updates of Omega (contribution) and Gamma (SNR)
                                                                     (lines 18-20)
  * hybrid split-federated training (Sec. 4.1): of the K selected devices,
    floor(rho K) train in split mode (cut after conv1) and the rest in
    federated mode (full local model) - federated mode is given to the most
    capable devices and only when the full model fits the device's deadline;
    UE-side parameters aggregated every round (Eq. 23);
  * verifiable mechanism (Sec. 4.2): each selected device evaluates the
    challenge set D_chal (proof), signs, uploads proof + signature; the
    server reconstructs the model and recomputes the predictions (Eq. 19-22).
    Its compute, signature and traffic are charged to the round.

No adversarial devices are simulated (all signatures/proofs verify), so the
security mechanisms show up as overhead only - the fair setting for an
efficiency comparison.  Selection fraction K/|candidates| = 0.5 (configurable;
the paper uses K = 10 of N = 100, which with 50 IoT devices would leave ~1
device per UAV cluster).
"""
from __future__ import annotations

import math

import numpy as np

from ..cost import cost_model as cm
from .base import BaselineSimulation
from .cost import add_overhead


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20) -> np.ndarray:
    n = len(X)
    k = max(1, min(k, n))
    C = X[rng.choice(n, size=k, replace=False)]
    lab = np.zeros(n, int)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        newC = np.array([X[lab == c].mean(0) if np.any(lab == c) else C[c] for c in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


class SHSFL(BaselineSimulation):
    initial_front_depth = 1

    def init_baseline(self):
        N = self.cfg.topology.n_iot
        hc = self.cfg.shsfl
        self.initial_front_depth = hc.cut
        self._set_front_depth(hc.cut)
        self.I = 1                                   # UE-side aggregation every round (Eq. 23)
        self.Omega = np.zeros(N)                     # EMA of ||dw_n||
        self.Gamma = np.zeros(N)                     # EMA of normalised SNR
        self.H_sum = np.zeros(N)
        self.H_cnt = np.zeros(N)
        self.count = np.zeros(N)                     # c_n (times selected)
        self.trust = np.ones(N)                      # T[n] / p_n
        self.coalition = np.zeros(N, int)
        self.phi = np.zeros(N)
        self.delta_max = 1e-12
        self.last_delta = np.zeros(N)
        self.t_sel = 0
        self.sel_stats = {}

    def baseline_state(self):
        return {k: getattr(self, k) for k in ("Omega", "Gamma", "H_sum", "H_cnt", "count", "trust",
                                              "coalition", "phi", "delta_max", "last_delta", "t_sel")}

    def load_baseline_state(self, s):
        for k, v in s.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ #
    def plan(self, ctx, recs):
        hc, cfg, net = self.cfg.shsfl, self.cfg, self.net
        self.t_sel += 1
        b = cfg.sfl.batch_per_device
        cand = np.concatenate([r["devs"] for r in recs]) if recs else np.zeros(0, int)
        if cand.size == 0:
            for rec in recs:
                rec.update(cuts=np.zeros(0, int), bsz=np.zeros(0, int), fed=np.zeros(0, bool))
            return
        # ---- observations: SNR, contribution, participation energy -------
        snr_db = 10 * np.log10(np.maximum(ctx["sinr_ul_e"][cand], 1e-12))
        snr = np.clip(snr_db / 40.0, 0.0, 1.0)
        dw = np.clip(self.last_delta[cand] / self.delta_max, 0.0, 1.0)
        w = cfg.sfl.float_bits
        fl = (self.prof.cfp(1, hc.cut) + self.prof.cbp(1, hc.cut)) * b
        bits = b * self.prof.act[hc.cut] * w
        e_part = (cfg.hw.kappa_iot * net.f_iot[cand] ** 2 * fl
                  + cfg.hw.iot_ptx_w * bits / np.maximum(ctx["r_ul_e"][cand], 1e-9))
        cost = hc.alpha * e_part / np.maximum(net.E_iot[cand], 1e-9)          # C_n = alpha * Energy_n
        U = hc.beta_g * snr + (1 - hc.beta_g) * dw - cost                       # Eq. (3)
        verified = np.ones(len(cand), bool)                                      # no adversaries
        P = verified & (U >= 0)                                                  # self-selection
        self.trust[cand[~P]] *= hc.trust_decay                                   # Eq. (8): refusal
        # ---- coalitions + Shapley contribution ---------------------------
        if (self.t_sel - 1) % hc.coalition_period == 0:
            self.coalition[cand] = kmeans(np.stack([snr, dw], 1), hc.n_coalitions, self.rng_base)
        self.phi[cand] = snr + dw                                                # exact Shapley, Eq. (4-5)
        # ---- UCB score (Alg. 1 line 14-15, Eq. 10-11) --------------------
        t = self.t_sel
        om = np.clip(self.Omega[cand] / self.delta_max, 0, 1)
        ga = self.Gamma[cand]
        c_n = self.count[cand]
        bonus = np.where(c_n > 0, hc.sigma_ucb * np.sqrt(2 * math.log(max(t, 2)) / np.maximum(c_n, 1)), np.inf)
        H = np.where(self.H_cnt[cand] > 0, self.H_sum[cand] / np.maximum(self.H_cnt[cand], 1), 0.0)
        A = hc.beta * om + (1 - hc.beta) * ga + bonus + hc.gamma_shapley * self.phi[cand] + hc.delta_hist * H
        A = A * self.trust[cand]
        A[~P] = -np.inf
        K = int(math.ceil(hc.select_frac * len(cand)))
        order = np.argsort(-A, kind="stable")
        sel = [int(cand[i]) for i in order[:K] if A[i] > -np.inf]
        sel_set = set(sel)
        # ---- hybrid modes: federated mode for the most capable devices ----
        n_fed_target = len(sel) - int(math.floor(hc.rho * len(sel)))
        full_fl = (self.prof.cfp(1, self.L) + self.prof.cbp(1, self.L)) * b
        full_bits = self.param_bits(1, self.L)
        T_MAX = cfg.t_max()
        fed_set = set()
        for d in sorted(sel, key=lambda d: -net.f_iot[d]):
            if len(fed_set) >= n_fed_target:
                break
            t_comp = full_fl / net.f_iot[d]
            t_model = full_bits / max(ctx["r_ul_e"][d], 1e-9) + full_bits / max(ctx["r_dl_e"][d], 1e-9)
            if t_comp <= cfg.sfl.iot_deadline_frac * T_MAX and t_model <= T_MAX:
                fed_set.add(d)
        for rec in recs:
            all_devs = rec["devs"]
            keep = np.array([d for d in all_devs if int(d) in sel_set], int)
            rec["n_unselected"] = int(len(all_devs) - len(keep))
            rec["devs"] = keep
            rec["cuts"] = np.full(len(keep), hc.cut, int)
            rec["bsz"] = np.full(len(keep), b, int)
            rec["fed"] = np.array([int(d) in fed_set for d in keep], bool)
            rec["extra"] = dict(n_candidates=int(len(all_devs)), n_selected=int(len(keep)))
        self.selected = sel
        self.sel_snr = dict(zip(cand.tolist(), snr.tolist()))
        self.sel_stats = dict(n_candidates=int(len(cand)), n_self_selected=int(P.sum()),
                              n_selected=int(len(sel)), n_fed=int(len(fed_set)),
                              mean_trust=float(self.trust[cand].mean()))

    # ------------------------------------------------------------------ #
    def extra_overhead(self, ctx, rec, sdevs, cuts, fed, pc, r_ul, r_dl, r_ju, r_uj, r_bh) -> float:
        """Verifiable mechanism (Sec. 4.2): proof generation on D_chal + digital
        signature on the device, proof upload, server-side reconstruction and
        re-computation of the challenge predictions (Eq. 19-22)."""
        hc, cfg, net = self.cfg.shsfl, self.cfg, self.net
        m = hc.challenge_samples
        ncls = self.data.num_classes
        fl_proof = np.array([m * self.prof.cfp(1, self.L if f_ else c) for c, f_ in zip(cuts, fed)])
        t_proof = fl_proof / net.f_iot[sdevs] + hc.sign_time_s
        e_proof = (cfg.hw.kappa_iot * net.f_iot[sdevs] ** 2 * fl_proof + cfg.hw.iot_pcir_w * hc.sign_time_s)
        proof_bits = np.full(len(sdevs), m * ncls * cfg.sfl.float_bits + hc.signature_bits, float)
        ts, es_i, es_j, es_u, sbits = cm.sync_cost(proof_bits, 0.0, 0.0, 0.0, r_ul, r_dl,
                                                   r_ju, r_uj, r_bh, r_bh, cfg.hw)
        cloud_fl = float(len(sdevs) * m * self.prof.cfp(1, self.L))            # reconstruct + predict
        t = float(t_proof.max()) + ts + cloud_fl / cfg.hw.cloud_f
        add_overhead(pc, t, es_i + e_proof, es_j, es_u, sbits, key="verify")
        return cloud_fl

    # ------------------------------------------------------------------ #
    def learn(self, ctx, recs):
        hc = self.cfg.shsfl
        st = ctx.get("st")
        upd = st.per_device_update if st is not None else {}
        completed = {int(d) for r in recs if r["status"] == "completed" for d in r["sdevs"]}
        for d in getattr(self, "selected", []):
            self.count[d] += 1
            dw = float(upd.get(d, 0.0)) if d in completed else 0.0
            if dw > 0:
                self.delta_max = max(self.delta_max, dw)
                self.last_delta[d] = dw
            snr = self.sel_snr.get(d, 0.0)
            self.Omega[d] = hc.lam_l * self.Omega[d] + (1 - hc.lam_l) * dw            # line 19
            self.Gamma[d] = hc.lam_c * self.Gamma[d] + (1 - hc.lam_c) * snr           # line 20
            self.H_sum[d] += hc.omega_hist * snr + (1 - hc.omega_hist) * min(dw / self.delta_max, 1.0)
            self.H_cnt[d] += 1
            if d in completed:                                                         # Eq. (9)
                self.trust[d] = min(self.trust[d] + hc.trust_gain, 1.0)

    def round_fields(self, ctx):
        return dict(self.sel_stats)
