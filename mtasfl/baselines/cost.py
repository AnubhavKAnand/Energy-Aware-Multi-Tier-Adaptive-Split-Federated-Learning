"""
Measurement model for the baselines in the shared 4-tier environment.

All three baselines are two-tier designs (devices <-> one server).  In the
4-tier network their server is the cloud and the UAVs only forward traffic:
  IoT i  --(UL)-->  primary j  --(A2A)-->  relay u  --(backhaul)-->  cloud
with the same store-and-forward structure, rates, energy parameters and
straggler rule as MTA-SFL (cost_model.pipeline_cost with p1 = p2 = p3).
Activations / gradients use the same top-k sparsification as MTA-SFL (all
methods share one compression setting; see methods.py), compressed per device.  Devices may
have different cut layers (AdaptSFL) and batch sizes (SFLSC3), and a device can
train the full model locally (S-HSFL federated-learning mode).
"""
from __future__ import annotations

import numpy as np

from ..cost import cost_model as cm
from ..cost.profile import ModelProfile
from ..cost.vectorized import sparse_bits_vec


def device_tier1(prof: ModelProfile, cuts, bs, f_iot, r_ul, r_dl, w, fed=None, jit=None, ratio=1.0):
    """Per-device compute / upload / gradient-return times, bits and FLOPs."""
    cuts = np.asarray(cuts, int)
    bs = np.asarray(bs, float)
    fed = np.zeros(len(cuts), bool) if fed is None else np.asarray(fed, bool)
    L = prof.L
    fl = np.array([(prof.cfp(1, L if fe else c) + prof.cbp(1, L if fe else c)) * b
                   for c, b, fe in zip(cuts, bs, fed)])
    tci = fl / np.asarray(f_iot, float)
    if jit is not None:
        tci = tci * jit
    # each device's activation tensor is top-k compressed on its own (k(w+r) bits,
    # Sec. III-C); UAVs forward the payloads unchanged.  FL-mode devices send none.
    bits = np.where(fed, 0.0, sparse_bits_vec(bs * prof.act[cuts], ratio, w))
    tmi = bits / np.maximum(np.asarray(r_ul, float), 1e-9)
    tgdl = bits / np.maximum(np.asarray(r_dl, float), 1e-9)
    return tci, tmi, tgdl, bits, fl


def forward_only_cost(prof: ModelProfile, cuts, bs, f_iot, r_ul_i, r_dl_i, r_ju, r_uj,
                      r_bh_ul, r_bh_dl, f_s, cfg, fed=None, jit_iot=None) -> cm.PipelineCost:
    """Training-stage latency and energy of one pipeline (surviving devices)."""
    hw, w = cfg.hw, cfg.sfl.float_bits
    cuts = np.asarray(cuts, int)
    bs = np.asarray(bs, float)
    fed = np.zeros(len(cuts), bool) if fed is None else np.asarray(fed, bool)
    L = prof.L
    tci, tmi, tgdl, bits, fl1 = device_tier1(prof, cuts, bs, f_iot, r_ul_i, r_dl_i, w, fed, jit_iot,
                                             cfg.sfl.topk_ratio)
    B = float(bs.sum())
    pc = cm.PipelineCost(B=B, bridge=True)
    tot = float(bits.sum())
    tg_max = float(tgdl.max()) if tgdl.size else 0.0
    tm_max = float(tmi.max()) if tmi.size else 0.0
    pc.t_iot = tci + tmi
    pc.t_grad_dl = tgdl
    pc.tcomp_j = 0.0
    pc.tcomm_j = tot / max(r_ju, 1e-9) + tg_max
    pc.tcomp_u = 0.0
    pc.tcomm_u = tot / max(r_bh_ul, 1e-9) + tot / max(r_uj, 1e-9)
    fl_s = float(sum((prof.cfp(c + 1, L) + prof.cbp(c + 1, L)) * b
                     for c, b, fe in zip(cuts, bs, fed) if not fe))
    pc.tcomp_s = fl_s / f_s
    pc.tcomm_s = tot / max(r_bh_dl, 1e-9)
    t_iot_max = float(pc.t_iot.max()) if pc.t_iot.size else 0.0
    pc.t_total = float(t_iot_max + pc.tcomm_j + pc.tcomm_u + pc.tcomp_s + pc.tcomm_s)
    pc.ecomp_i = hw.kappa_iot * np.asarray(f_iot, float) ** 2 * fl1
    pc.ecomp_j = pc.ecomp_u = 0.0
    pc.ecomm_i = hw.iot_ptx_w * tmi + hw.iot_prx_w * tgdl
    pc.ecomm_j = (hw.uav_ptx_w * (tot / max(r_ju, 1e-9) + tg_max)
                  + hw.uav_prx_w * (tm_max + tot / max(r_uj, 1e-9)))
    pc.ecomm_u = (hw.uav_ptx_w * (tot / max(r_bh_ul, 1e-9) + tot / max(r_uj, 1e-9))
                  + hw.uav_prx_w * (tot / max(r_ju, 1e-9) + tot / max(r_bh_dl, 1e-9)))
    pc.tbusy_i = tci + tmi + tg_max
    pc.bits = dict(fwd1=bits, fwd2=tot, fwd3=tot, grad1=bits, grad2=tot, grad3=tot)
    pc.cloud_flops = fl_s
    return pc


def add_overhead(pc: cm.PipelineCost, t: float, e_iot, e_j: float, e_u: float, bits: float,
                 key: str = "sync"):
    """Add post-training traffic (model aggregation / verification) that runs
    outside the T_max training deadline."""
    pc.t_sync += t
    pc.ecomm_i = pc.ecomm_i + np.asarray(e_iot, float)
    pc.ecomm_j += e_j
    pc.ecomm_u += e_u
    pc.tbusy_i = pc.tbusy_i + t
    pc.bits[key] = pc.bits.get(key, 0.0) + bits
    if key != "sync":
        pc.bits["sync"] = pc.bits.get("sync", 0.0) + bits
