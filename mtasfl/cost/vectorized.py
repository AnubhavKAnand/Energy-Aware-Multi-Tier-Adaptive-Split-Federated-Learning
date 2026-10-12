"""
Vectorised evaluation of the measurement model over many split tuples at once
(same equations as cost_model.pipeline_cost / finalize).  Used for

* the LinUCB feasible set P_feasible(t)  (Eq. 31, mask T_total > T_max),
* the DDQN safe set U_safe(t)            (hard battery + deadline limits),
* the expected-cost oracle for empirical regret (Eq. 19).
"""
from __future__ import annotations

import itertools

import numpy as np

from .profile import ModelProfile


def all_arms(L: int, min_p1: int = 1, max_p3: int | None = None) -> np.ndarray:
    """Every tuple 1 <= p1 <= p2 <= p3 <= L (constraint 17a).  By default
    p3 <= L-1: the cloud always computes at least the final block and the loss
    (Tier 4, Sec. III-A-2(d)); otherwise sparsified logits would reach the loss."""
    max_p3 = L - 1 if max_p3 is None else max_p3
    return np.array([(a, b, c) for a, b, c in itertools.combinations_with_replacement(range(1, L + 1), 3)
                     if a >= min_p1 and c <= max_p3], dtype=int)


def sparse_bits_vec(n_elems, ratio: float, w: int):
    n = np.asarray(n_elems, float)
    if ratio >= 1.0:
        return n * w
    k = np.maximum(1, np.ceil(ratio * n))
    r = np.ceil(np.log2(np.maximum(n, 2)))
    return np.where(n > 0, k * (w + r), 0.0)


class ArmEstimator:
    def __init__(self, prof: ModelProfile, arms: np.ndarray, cfg):
        self.prof, self.arms, self.cfg = prof, arms, cfg
        cf = np.cumsum(prof.fwd)
        cb = np.cumsum(prof.bwd)
        c = cf + cb
        p1, p2, p3 = arms[:, 0], arms[:, 1], arms[:, 2]
        L = prof.L
        self.fl1 = c[p1] - c[0]                 # per sample
        self.fl2 = c[p2] - c[p1]
        self.fl3 = c[p3] - c[p2]
        self.fl4 = c[L] - c[p3]
        self.a1, self.a2, self.a3 = prof.act[p1], prof.act[p2], prof.act[p3]
        self.bridge = p2 == p3
        # parameters held by each tier (model synchronisation), bits
        cp = np.cumsum(prof.n_params) * cfg.sfl.float_bits
        self.d1 = cp[p1] - cp[0]
        self.d2 = cp[p2] - cp[p1]
        self.d3 = cp[p3] - cp[p2]

    def estimate(self, b, f_iot, r_ul_i, r_dl_i, f_j, r_ju, r_uj, f_u, r_bh_ul, r_bh_dl,
                 f_s, e_res_i, e_res_j, e_res_u, pflight_j, pflight_u, t_round=None):
        """Returns dict of arrays over arms: T_total, E_total_u, E_total_j,
        E_total_i (arms x n), e_ratio, t_norm, feasible (17b & 17c)."""
        cfg, hw, sfl, oc = self.cfg, self.cfg.hw, self.cfg.sfl, self.cfg.obj
        ratio, w = sfl.topk_ratio, sfl.float_bits
        b = np.asarray(b, float)
        f_iot = np.asarray(f_iot, float)
        n = b.size
        nA = len(self.arms)
        T_MAX = cfg.t_max()
        # tier 1 (arms x n)
        fl1 = self.fl1[:, None] * b[None, :]
        tci = fl1 / f_iot[None, :]
        bits1 = sparse_bits_vec(self.a1[:, None] * b[None, :], ratio, w)
        tmi = bits1 / np.maximum(r_ul_i, 1e-9)[None, :]
        tgdl = bits1 / np.maximum(r_dl_i, 1e-9)[None, :]
        # expected survivors: devices whose tier-1 link time (compute + upload +
        # gradient return) meets the local deadline; the others are dropped as
        # stragglers (Sec. III-A-2(a)), not the whole route
        mg = 1.0 + sfl.mask_margin
        S = (tci + tmi + tgdl) * mg <= sfl.iot_deadline_frac * T_MAX  # arms x n
        n_surv = S.sum(1)
        B = (S * b[None, :]).sum(1)                                  # arms
        neg = -np.inf
        t_iot = np.where(S, tci + tmi, neg).max(1) if n else np.zeros(nA)
        tgdl_max = np.where(S, tgdl, neg).max(1) if n else np.zeros(nA)
        tmi_max = np.where(S, tmi, neg).max(1) if n else np.zeros(nA)
        t_iot, tgdl_max, tmi_max = [np.where(n_surv > 0, x, 0.0) for x in (t_iot, tgdl_max, tmi_max)]
        # tier 2/3/4 (per-arm surviving batch B)
        bits2 = sparse_bits_vec(self.a2 * B, ratio, w)
        bits3 = sparse_bits_vec(self.a3 * B, ratio, w)
        tcj = self.fl2 * B / f_j
        tmj = bits2 / max(r_ju, 1e-9) + tgdl_max
        tcu = self.fl3 * B / f_u
        tmu = bits3 / max(r_bh_ul, 1e-9) + bits2 / max(r_uj, 1e-9)
        tcs = self.fl4 * B / f_s
        tms = bits3 / max(r_bh_dl, 1e-9)
        T = t_iot + tcj + tmj + tcu + tmu + tcs + tms
        tr = T if t_round is None else np.full_like(T, t_round)
        # energy (surviving devices follow Eq. 12-13)
        ecomp_i = hw.kappa_iot * f_iot[None, :] ** 2 * fl1
        ecomm_i = hw.iot_ptx_w * tmi + hw.iot_prx_w * tgdl * S
        tbusy = tci + tmi + tgdl_max[:, None] * S
        # per-round layer synchronisation (same model as cost_model.sync_cost)
        T_sync = np.zeros(nA)
        es_i = np.zeros_like(tci)
        es_j = np.zeros(nA)
        es_u = np.zeros(nA)
        if sfl.model_sync and n:
            r_ju_, r_uj_ = max(r_ju, 1e-9), max(r_uj, 1e-9)
            r_bu, r_bd = max(r_bh_ul, 1e-9), max(r_bh_dl, 1e-9)
            t_up_i = self.d1[:, None] / np.maximum(r_ul_i, 1e-9)[None, :]
            t_dn_i = self.d1[:, None] / np.maximum(r_dl_i, 1e-9)[None, :]
            t_up_max = np.where(S, t_up_i, 0.0).max(1)
            t_dn_max = np.where(S, t_dn_i, 0.0).max(1)
            U_iot = self.d1 * n_surv
            t_up_ju = (U_iot + self.d2) / r_ju_
            t_up_bh = (U_iot + self.d2 + self.d3) / r_bu
            t_dn_bh = (self.d1 + self.d2 + self.d3) / r_bd
            t_dn_uj = (self.d1 + self.d2) / r_uj_
            T_sync = t_up_max + t_up_ju + t_up_bh + t_dn_bh + t_dn_uj + t_dn_max
            es_i = (hw.iot_ptx_w * t_up_i + hw.iot_prx_w * t_dn_i) * S
            es_j = hw.uav_prx_w * (t_up_max + t_dn_uj) + hw.uav_ptx_w * (t_up_ju + t_dn_max)
            es_u = hw.uav_prx_w * (t_up_ju + t_dn_bh) + hw.uav_ptx_w * (t_up_bh + t_dn_uj)
            tbusy = tbusy + T_sync[:, None] * S
        ecomm_i = ecomm_i + es_i
        ecir_i = hw.iot_pcir_w * tbusy + hw.iot_psleep_w * np.maximum(tr[:, None] - tbusy, 0)
        E_i = ecomp_i + ecomm_i + ecir_i
        ecomp_j = hw.kappa_uav * f_j ** 2 * self.fl2 * B
        ecomm_j = (hw.uav_ptx_w * (bits2 / max(r_ju, 1e-9) + tgdl_max)
                   + hw.uav_prx_w * (tmi_max + bits2 / max(r_uj, 1e-9)))
        ecomp_u = hw.kappa_uav * f_u ** 2 * self.fl3 * B
        ecomm_u = (hw.uav_ptx_w * (bits3 / max(r_bh_ul, 1e-9) + bits2 / max(r_uj, 1e-9))
                   + hw.uav_prx_w * (bits2 / max(r_ju, 1e-9) + bits3 / max(r_bh_dl, 1e-9)))
        E_j = (pflight_j + hw.uav_pcir_w) * tr + ecomp_j + ecomm_j + es_j
        E_u = (pflight_u + hw.uav_pcir_w) * tr + ecomp_u + ecomm_u + es_u
        e_res_i = np.asarray(e_res_i, float)
        r_i = np.minimum(E_i / np.maximum(e_res_i, 1e-9)[None, :], oc.energy_ratio_clip)
        r_j = np.minimum(E_j / max(e_res_j, 1e-9), oc.energy_ratio_clip)
        r_u = np.minimum(E_u / max(e_res_u, 1e-9), oc.energy_ratio_clip)
        # E~ averages over the active set (surviving devices + j + u), Sec. V
        e_ratio = ((r_i * S).sum(1) + r_j + r_u) / (n_surv + 2)
        t_norm = (T + T_sync) / T_MAX          # objective: training + sync
        viol_e = ((E_i > e_res_i[None, :]) & S).any(1) | (E_j > e_res_j) | (E_u > e_res_u)
        # hard limits (17b), (17c) and at least one surviving device
        # no voluntary straggling: a split may not drop more devices than the
        # best split must (otherwise latency could be "reduced" by dropping data)
        feasible = (T / T_MAX * mg <= 1.0) & ~viol_e & (n_surv > 0) & (n_surv >= n_surv.max())
        surv_frac = n_surv / max(n, 1)
        return dict(T=T, t_norm=t_norm, e_ratio=e_ratio, E_u=E_u, E_j=E_j, E_i=E_i,
                    feasible=feasible, viol_e=viol_e, surv_frac=surv_frac, B=B, T_sync=T_sync)
