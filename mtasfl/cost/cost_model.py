"""
Measurement model: multi-tier latency (Sec. IV-A, Eq. 8-9) and energy
(Sec. IV-B, Eq. 10-16), plus the normalised objective (Eq. 17) and the agent
rewards (Eq. 22, 28).

A pipeline j consists of surviving IoT devices i, primary UAV j, relay u and the
cloud S, with split tuple p = (p1, p2, p3).  `p2 == p3` is the topological-
bridge mode: the relay computes nothing (Tcomp = Ecomp = 0) but still receives
and forwards the payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .profile import ModelProfile, sparse_bits


# --------------------------------------------------------------------------- #
def zeng_power(v, hw) -> np.ndarray:
    """Rotary-wing propulsion power P(V) (Zeng, Xu, Zhang, TWC 2019)."""
    v = np.asarray(v, float)
    p_blade = hw.zeng_p0 * (1 + 3 * v ** 2 / hw.zeng_utip ** 2)
    p_ind = hw.zeng_pi * np.sqrt(np.sqrt(1 + v ** 4 / (4 * hw.zeng_v0 ** 4)) - v ** 2 / (2 * hw.zeng_v0 ** 2))
    p_par = 0.5 * hw.zeng_d0 * hw.zeng_rho * hw.zeng_s * hw.zeng_A * v ** 3
    return p_blade + p_ind + p_par


def flight_power(speeds, is_primary, cfg) -> np.ndarray:
    """P_flight per UAV (Eq. 10)."""
    hw, tp = cfg.hw, cfg.topology
    if hw.prop_model == "zeng":
        return zeng_power(speeds, hw)
    v = np.where(is_primary, tp.primary_mean_speed, tp.relay_mean_speed)
    return zeng_power(v, hw)


# --------------------------------------------------------------------------- #
def iot_upload(prof: ModelProfile, p1: int, b, f_iot, r_ul, ratio, w, jitter=None):
    """Tier-1 compute + upload time per device (used for the straggler test)."""
    b = np.asarray(b, float)
    flops = (prof.cfp(1, p1) + prof.cbp(1, p1)) * b
    tcomp = flops / f_iot
    if jitter is not None:
        tcomp = tcomp * jitter
    bits = np.array([sparse_bits(bi * prof.act[p1], ratio, w) for bi in b])
    tcomm = bits / np.maximum(r_ul, 1e-9)
    return tcomp, tcomm, bits, flops


@dataclass
class PipelineCost:
    # latency components
    t_iot: np.ndarray = None        # per surviving device: Tcomp_i + Tcomm_i
    t_grad_dl: np.ndarray = None    # per device gradient downlink time
    tcomp_j: float = 0.0
    tcomm_j: float = 0.0
    tcomp_u: float = 0.0
    tcomm_u: float = 0.0
    tcomp_s: float = 0.0
    tcomm_s: float = 0.0
    t_total: float = 0.0
    # energy components (excluding propulsion / circuit, which need T_round)
    ecomp_i: np.ndarray = None
    ecomm_i: np.ndarray = None
    tbusy_i: np.ndarray = None
    ecomp_j: float = 0.0
    ecomm_j: float = 0.0
    ecomp_u: float = 0.0
    ecomm_u: float = 0.0
    # payloads (bits)
    bits: dict = field(default_factory=dict)
    B: float = 0.0
    bridge: bool = False
    # model synchronisation (outside the T_max training deadline)
    t_sync: float = 0.0
    # filled by finalize()
    e_total_i: np.ndarray = None
    e_total_j: float = 0.0
    e_total_u: float = 0.0
    e_parts: dict = field(default_factory=dict)


def sync_cost(up_iot, dn_iot, up_j, up_u, r_ul_i, r_dl_i, r_ju, r_uj, r_bh_ul, r_bh_dl, hw):
    """Model synchronisation over the 4-tier path (store-and-forward).

    Upload  : every IoT device sends its own layer parameters (up_iot[i] bits)
              IoT -> j -> u -> cloud; the primary adds its own (up_j), the relay
              its own (up_u).
    Download: the cloud broadcasts the aggregated parameters back along
              cloud -> u -> j -> IoT; IoT layers are identical for all devices
              of the pipeline after aggregation, so dn_iot is sent once per hop
              and received by each device on its own sub-channel.
    Returns (time, e_iot[array], e_j, e_u, total_bits)."""
    up_iot = np.asarray(up_iot, float)
    r_ul_i = np.maximum(np.asarray(r_ul_i, float), 1e-9)
    r_dl_i = np.maximum(np.asarray(r_dl_i, float), 1e-9)
    r_ju, r_uj, r_bh_ul, r_bh_dl = (max(x, 1e-9) for x in (r_ju, r_uj, r_bh_ul, r_bh_dl))
    U_iot = float(up_iot.sum())
    t_up_i = up_iot / r_ul_i
    t_dn_i = dn_iot / r_dl_i
    t_up_ju = (U_iot + up_j) / r_ju
    t_up_bh = (U_iot + up_j + up_u) / r_bh_ul
    t_dn_bh = (dn_iot + up_j + up_u) / r_bh_dl
    t_dn_uj = (dn_iot + up_j) / r_uj
    t_up_max = float(t_up_i.max()) if t_up_i.size else 0.0
    t_dn_max = float(t_dn_i.max()) if t_dn_i.size else 0.0
    t = t_up_max + t_up_ju + t_up_bh + t_dn_bh + t_dn_uj + t_dn_max
    e_iot = hw.iot_ptx_w * t_up_i + hw.iot_prx_w * t_dn_i
    e_j = hw.uav_prx_w * (t_up_max + t_dn_uj) + hw.uav_ptx_w * (t_up_ju + t_dn_max)
    e_u = hw.uav_prx_w * (t_up_ju + t_dn_bh) + hw.uav_ptx_w * (t_up_bh + t_dn_uj)
    bits = U_iot + up_j + up_u + dn_iot * (1 + len(up_iot)) + 2 * (up_j + up_u)
    return t, e_iot, e_j, e_u, bits


def param_bits(prof: ModelProfile, a: int, b: int, w: int) -> float:
    """Parameters of blocks a..b inclusive, in bits (0 if a > b)."""
    if a > b:
        return 0.0
    return float(prof.n_params[a:b + 1].sum()) * w


def pipeline_cost(prof: ModelProfile, p, b, f_iot, r_ul_i, r_dl_i,
                  f_j, r_ul_ju, r_dl_uj, f_u, r_bh_ul, r_bh_dl, f_s, cfg,
                  jitter=None) -> PipelineCost:
    """Latency (Eq. 9) and computation/communication energy (Eq. 13-14) of one
    pipeline over its *surviving* devices (arrays b, f_iot, r_ul_i, r_dl_i).

    jitter: optional dict {'iot': array, 'j': float, 'u': float, 's': float}
    multiplying compute times (realised vs estimated)."""
    hw, sfl = cfg.hw, cfg.sfl
    p1, p2, p3 = p
    L = prof.L
    ratio, w = sfl.topk_ratio, sfl.float_bits
    jit = jitter or {}
    b = np.asarray(b, float)
    B = float(b.sum())
    pc = PipelineCost(B=B, bridge=(p2 == p3))

    # ---- Tier 1 -----------------------------------------------------------
    tci, tmi, bits1, fl1 = iot_upload(prof, p1, b, f_iot, r_ul_i, ratio, w, jit.get("iot"))
    gbits1 = bits1  # |grad delta_1,i| has the same shape as delta_1,i
    t_gdl = gbits1 / np.maximum(r_dl_i, 1e-9)
    pc.t_iot = tci + tmi
    pc.t_grad_dl = t_gdl

    # ---- Tier 2 (primary UAV) --------------------------------------------
    fl2 = (prof.cfp(p1 + 1, p2) + prof.cbp(p1 + 1, p2)) * B
    pc.tcomp_j = fl2 / f_j * jit.get("j", 1.0)
    bits2 = sparse_bits(B * prof.act[p2], ratio, w)
    pc.tcomm_j = bits2 / max(r_ul_ju, 1e-9) + (t_gdl.max() if t_gdl.size else 0.0)

    # ---- Tier 3 (relay) ---------------------------------------------------
    fl3 = (prof.cfp(p2 + 1, p3) + prof.cbp(p2 + 1, p3)) * B   # 0 in bridge mode
    pc.tcomp_u = fl3 / f_u * jit.get("u", 1.0)
    bits3 = sparse_bits(B * prof.act[p3], ratio, w)
    pc.tcomm_u = bits3 / max(r_bh_ul, 1e-9) + bits2 / max(r_dl_uj, 1e-9)

    # ---- Tier 4 (cloud) ---------------------------------------------------
    fl4 = (prof.cfp(p3 + 1, L) + prof.cbp(p3 + 1, L)) * B
    pc.tcomp_s = fl4 / f_s * jit.get("s", 1.0)
    pc.tcomm_s = bits3 / max(r_bh_dl, 1e-9)

    # ---- Eq. (9) ----------------------------------------------------------
    t_iot_max = pc.t_iot.max() if pc.t_iot.size else 0.0
    pc.t_total = float(t_iot_max + pc.tcomp_j + pc.tcomm_j + pc.tcomp_u + pc.tcomm_u
                       + pc.tcomp_s + pc.tcomm_s)

    # ---- computation energy  E = kappa f^2 FLOPs -------------------------
    pc.ecomp_i = hw.kappa_iot * f_iot ** 2 * fl1
    pc.ecomp_j = hw.kappa_uav * f_j ** 2 * fl2
    pc.ecomp_u = hw.kappa_uav * f_u ** 2 * fl3

    # ---- communication energy (Eq. 13, Tier-2 eq., Eq. 14) ---------------
    pc.ecomm_i = hw.iot_ptx_w * tmi + hw.iot_prx_w * t_gdl
    t_up_max = tmi.max() if tmi.size else 0.0
    pc.ecomm_j = (hw.uav_ptx_w * (bits2 / max(r_ul_ju, 1e-9) + (t_gdl.max() if t_gdl.size else 0.0))
                  + hw.uav_prx_w * (t_up_max + bits2 / max(r_dl_uj, 1e-9)))
    pc.ecomm_u = (hw.uav_ptx_w * (bits3 / max(r_bh_ul, 1e-9) + bits2 / max(r_dl_uj, 1e-9))
                  + hw.uav_prx_w * (bits2 / max(r_ul_ju, 1e-9) + bits3 / max(r_bh_dl, 1e-9)))
    pc.tbusy_i = tci + tmi + (t_gdl.max() if t_gdl.size else 0.0)
    pc.bits = dict(fwd1=bits1, fwd2=bits2, fwd3=bits3, grad1=gbits1, grad2=bits2, grad3=bits3)

    # ---- per-round layer synchronisation for Eq. (4) ----------------------
    if sfl.model_sync:
        d1 = param_bits(prof, 1, p1, w)
        ts, es_i, es_j, es_u, sbits = sync_cost(np.full(len(b), d1), d1, param_bits(prof, p1 + 1, p2, w),
                                                param_bits(prof, p2 + 1, p3, w), r_ul_i, r_dl_i,
                                                r_ul_ju, r_dl_uj, r_bh_ul, r_bh_dl, hw)
        pc.t_sync = ts
        pc.ecomm_i = pc.ecomm_i + es_i
        pc.ecomm_j += es_j
        pc.ecomm_u += es_u
        pc.tbusy_i = pc.tbusy_i + ts
        pc.bits["sync"] = sbits
    return pc


def finalize(pc: PipelineCost, t_round: float, pflight_j: float, pflight_u: float, cfg):
    """Add propulsion (Eq. 10) and circuit (Eq. 11-12) energy and compute the
    per-node totals (Eq. 15).  `t_round` is the time used for the hovering /
    circuit energy (paper: global T_round)."""
    hw = cfg.hw
    tr = t_round
    t_idle = np.maximum(tr - pc.tbusy_i, 0.0)
    ecir_i = hw.iot_pcir_w * pc.tbusy_i + hw.iot_psleep_w * t_idle
    eprop_j, eprop_u = pflight_j * tr, pflight_u * tr
    ecir_j = ecir_u = hw.uav_pcir_w * tr
    pc.e_total_i = ecir_i + pc.ecomp_i + pc.ecomm_i
    pc.e_total_j = eprop_j + ecir_j + pc.ecomp_j + pc.ecomm_j
    pc.e_total_u = eprop_u + ecir_u + pc.ecomp_u + pc.ecomm_u
    pc.e_parts = dict(
        prop=eprop_j + eprop_u,
        cir=ecir_j + ecir_u + float(ecir_i.sum()),
        comp=pc.ecomp_j + pc.ecomp_u + float(pc.ecomp_i.sum()),
        comm=pc.ecomm_j + pc.ecomm_u + float(pc.ecomm_i.sum()),
        prop_j=eprop_j, prop_u=eprop_u,
        comp_u=pc.ecomp_u, comm_u=pc.ecomm_u, comp_j=pc.ecomp_j, comm_j=pc.ecomm_j,
    )
    return pc


# --------------------------------------------------------------------------- #
def objective_terms(pc: PipelineCost, e_res_i, e_res_j, e_res_u, cfg):
    """Normalised latency T~, energy depletion ratio E~ (Sec. V) and the
    constraint flags of (17b), (17c)."""
    oc, sfl = cfg.obj, cfg.sfl
    # T~ in the objective covers training + synchronisation; the deadline
    # (17b) applies to the training stage T_total only
    t_norm = (pc.t_total + pc.t_sync) / cfg.t_max()
    e = np.concatenate([pc.e_total_i, [pc.e_total_j, pc.e_total_u]])
    r = np.concatenate([np.asarray(e_res_i, float), [e_res_j, e_res_u]])
    ratio = e / np.maximum(r, 1e-9)
    e_ratio = float(np.mean(np.minimum(ratio, oc.energy_ratio_clip)))
    viol_t = pc.t_total / cfg.t_max() > 1.0
    viol_e = bool(np.any(e > r))
    return t_norm, e_ratio, viol_t, viol_e


def global_cost(t_norm, e_ratio, cc_u, cfg) -> float:
    """Eq. (17):  C = lam1 T~ + lam2 E~ - lam3 CC_u."""
    oc = cfg.obj
    return oc.lam1 * t_norm + oc.lam2 * e_ratio - oc.lam3 * cc_u


def ddqn_extrinsic_reward(t_norm, e_ratio, cc_u, violated, cfg) -> float:
    """Eq. (22) with shared terminal penalty Psi on any constraint violation."""
    oc = cfg.obj
    psi = oc.psi_penalty if violated else 0.0
    return oc.lam3 * cc_u - oc.lam2 * e_ratio - oc.lam1 * t_norm - psi


def linucb_reward(t_norm, e_ratio, cfg) -> float:
    """Eq. (28):  r = -(lam1 T~ + lam2 E~).  (Infeasible splits are masked
    out beforehand, Eq. 31, so no terminal penalty enters the bandit.)"""
    oc = cfg.obj
    return -(oc.lam1 * t_norm + oc.lam2 * e_ratio)
