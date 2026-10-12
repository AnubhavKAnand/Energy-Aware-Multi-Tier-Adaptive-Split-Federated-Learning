import math

import numpy as np
import pytest

from mtasfl.config import Config
from mtasfl.env.network import Network
from mtasfl.env import channel as chm
from mtasfl.cost.profile import synthetic_profile, sparse_bits
from mtasfl.cost import cost_model as cm


def make(seed=0, **ov):
    cfg = Config().override(ov) if ov else Config()
    return cfg, Network(cfg, np.random.default_rng(seed))


# ------------------------------------------------------------------ association
def test_association_capacity_and_feasibility():
    cfg, net = make(topology={"n_iot": 80}, sfl={"k_max_per_uav": 10})
    asc = net.associate()
    assert asc.K.max() <= 10
    assert asc.K.sum() == (asc.a >= 0).sum()
    # total capacity 50 < 80 devices -> exactly 50 associated if all feasible
    snr_db = chm.lin2db(net.iot_snr_assoc())
    feasible_any = (snr_db >= cfg.channel.gamma_min_db).any(1)
    assert (asc.a >= 0).sum() == min(50, feasible_any.sum())
    # sub-channel indices are unique within each cluster
    for j in range(net.P):
        s = asc.subch[asc.a == j]
        assert len(set(s)) == len(s)


def test_association_falls_back_to_other_uav_when_full():
    """R3.C1: a device whose best UAV is full goes to another feasible UAV."""
    cfg, net = make(topology={"n_iot": 30, "n_primary": 2, "n_relay": 3}, sfl={"k_max_per_uav": 15})
    net.iot_xy[:] = net.hotspots[0]          # everybody prefers UAV 0
    net.uav_xyz[1, :2] = net.hotspots[0] + 50
    asc = net.associate()
    assert asc.K[0] == 15 and asc.K[1] == 15


def test_dropped_devices_prioritised():
    cfg, net = make()
    net.prev_dropped[:] = False
    net.prev_dropped[[3, 7, 11]] = True
    asc = net.associate()
    assert set(asc.queue_order[:3]) == {3, 7, 11}


# ------------------------------------------------------------------ interference
def test_orthogonal_blocks_remove_interference():
    cfg, net = make(channel={"reuse_blocks": 100})
    asc = net.associate()
    ul, dl = net.ground_sinr(asc)
    # SNR without interference
    G = chm.a2g_gain(net.iot_xy[:, None, :], net.uav_xyz[None, :net.P, :], cfg.channel)
    on = asc.a >= 0
    K = asc.K[asc.a[on]]
    snr = cfg.hw.iot_ptx_w * G[on, asc.a[on]] / chm.noise_power_w(cfg.channel.w_ul_hz / K, cfg.channel)
    assert np.allclose(ul[on], snr)


def test_full_reuse_degrades_sinr():
    _, n1 = make(seed=3, channel={"reuse_blocks": 1})
    _, n2 = make(seed=3, channel={"reuse_blocks": 100})
    a1, a2 = n1.associate(), n2.associate()
    s1, _ = n1.ground_sinr(a1)
    s2, _ = n2.ground_sinr(a2)
    assert s1[a1.a >= 0].mean() < s2[a2.a >= 0].mean()


# ------------------------------------------------------------------ topology
def test_clustering_coefficient():
    full = np.ones((4, 4), bool)
    np.fill_diagonal(full, False)
    assert np.allclose(Network.clustering_coefficient(full), 1.0)
    star = np.zeros((4, 4), bool)
    star[0, 1:] = star[1:, 0] = True
    cc = Network.clustering_coefficient(star)
    assert cc[0] == 0.0 and np.all(cc[1:] == 0.0)


def test_link_expiration_time():
    dp = np.array([[0.0, 0, 0]])
    dv = np.array([[10.0, 0, 0]])
    t = Network._let(dp, dv, r=100.0, cap=600)
    assert math.isclose(t[0], 10.0)
    assert Network._let(dp, np.zeros((1, 3)), 100.0, 600)[0] == 600


def test_snapshot_shapes():
    cfg, net = make()
    s = net.snapshot()
    assert s["A"].shape == (net.U, net.U)
    assert s["life"].shape == (net.P, net.U)
    assert np.all((s["cc"] >= 0) & (s["cc"] <= 1))
    net.step_mobility()
    assert np.all((net.uav_xyz[:, :2] >= 0) & (net.uav_xyz[:, :2] <= cfg.topology.area_m))


# ------------------------------------------------------------------ cost model
def _pipe(cfg, p, prof):
    b = np.array([16.0, 16.0, 8.0])
    return cm.pipeline_cost(prof, p, b, f_iot=np.array([1e9, 1e9, 5e8]),
                            r_ul_i=np.array([2e6, 1e6, 3e6]), r_dl_i=np.array([2e6, 2e6, 2e6]),
                            f_j=8e9, r_ul_ju=2e7, r_dl_uj=2e7, f_u=6e9,
                            r_bh_ul=5e7, r_bh_dl=5e7, f_s=1e12, cfg=cfg)


def test_latency_hand_check():
    cfg = Config()
    prof = synthetic_profile(L=6)
    p = (1, 3, 5)
    pc = _pipe(cfg, p, prof)
    w, ratio = cfg.sfl.float_bits, cfg.sfl.topk_ratio
    b = np.array([16.0, 16.0, 8.0])
    fiot = np.array([1e9, 1e9, 5e8])
    t1 = (prof.cfp(1, 1) + prof.cbp(1, 1)) * b / fiot + \
        np.array([sparse_bits(x * prof.act[1], ratio, w) for x in b]) / np.array([2e6, 1e6, 3e6])
    B = 40.0
    tj = (prof.cfp(2, 3) + prof.cbp(2, 3)) * B / 8e9
    bits2 = sparse_bits(B * prof.act[3], ratio, w)
    gdl = np.array([sparse_bits(x * prof.act[1], ratio, w) for x in b]).max() / 2e6
    tcj = bits2 / 2e7 + gdl
    tu = (prof.cfp(4, 5) + prof.cbp(4, 5)) * B / 6e9
    bits3 = sparse_bits(B * prof.act[5], ratio, w)
    tcu = bits3 / 5e7 + bits2 / 2e7
    ts = (prof.cfp(6, 6) + prof.cbp(6, 6)) * B / 1e12
    tcs = bits3 / 5e7
    expect = t1.max() + tj + tcj + tu + tcu + ts + tcs
    assert math.isclose(pc.t_total, expect, rel_tol=1e-12)


def test_bridge_mode_zero_compute_nonzero_comm():
    """R2.7: p2 == p3 removes relay computation but NOT communication."""
    cfg = Config()
    prof = synthetic_profile(L=6)
    pc = _pipe(cfg, (1, 3, 3), prof)
    assert pc.bridge and pc.tcomp_u == 0.0 and pc.ecomp_u == 0.0
    assert pc.ecomm_u > 0 and pc.tcomm_u > 0
    pc2 = _pipe(cfg, (1, 3, 5), prof)
    assert pc2.ecomp_u > 0


def test_energy_accounting_closes():
    cfg = Config()
    prof = synthetic_profile(L=6)
    pc = _pipe(cfg, (2, 3, 5), prof)
    cm.finalize(pc, t_round=pc.t_total * 1.3, pflight_j=170.0, pflight_u=180.0, cfg=cfg)
    total = pc.e_total_i.sum() + pc.e_total_j + pc.e_total_u
    parts = sum(pc.e_parts[k] for k in ("prop", "cir", "comp", "comm"))
    assert math.isclose(total, parts, rel_tol=1e-12)


def test_dense_payload():
    assert sparse_bits(1000, 1.0, 32) == 32000
    assert sparse_bits(1000, 0.1, 32) == 100 * (32 + 10)


def test_objective_and_rewards():
    cfg = Config()
    prof = synthetic_profile(L=6)
    pc = _pipe(cfg, (2, 3, 5), prof)
    cm.finalize(pc, pc.t_total, 170.0, 180.0, cfg)
    tn, er, vt, ve = cm.objective_terms(pc, [1e4] * 3, 3e5, 3e5, cfg)
    c = cm.global_cost(tn, er, 0.5, cfg)
    r = cm.ddqn_extrinsic_reward(tn, er, 0.5, vt or ve, cfg)
    if not (vt or ve):
        assert math.isclose(r, -c)
    # infinite-capacity battery not violated, tiny battery is
    _, _, _, ve2 = cm.objective_terms(pc, [1e-6] * 3, 3e5, 3e5, cfg)
    assert ve2


def test_zeng_hover_power():
    hw = Config().hw
    assert math.isclose(float(cm.zeng_power(0.0, hw)), hw.zeng_p0 + hw.zeng_pi, rel_tol=1e-9)


import pytest as _pt


@_pt.mark.parametrize("sync", [False, True])
def test_vectorized_matches_scalar(sync):
    from mtasfl.cost.vectorized import ArmEstimator, all_arms
    cfg = Config().override({"dataset": "mnist", "sfl.iot_deadline_frac": 100.0, "sfl.model_sync": sync})
    prof = synthetic_profile(L=6)
    arms = all_arms(6)
    est = ArmEstimator(prof, arms, cfg)
    b = np.array([16.0, 16.0, 8.0])
    f = np.array([1e9, 1e9, 5e8])
    rul, rdl = np.array([2e6, 1e6, 3e6]), np.array([2e6, 2e6, 2e6])
    out = est.estimate(b, f, rul, rdl, 8e9, 2e7, 2e7, 6e9, 5e7, 5e7, 1e12,
                       [1e4] * 3, 3e5, 3e5, 170.0, 180.0)
    for k in [0, 5, len(arms) - 1]:
        pc = cm.pipeline_cost(prof, tuple(arms[k]), b, f, rul, rdl, 8e9, 2e7, 2e7, 6e9, 5e7, 5e7, 1e12, cfg)
        cm.finalize(pc, pc.t_total, 170.0, 180.0, cfg)
        tn, er, _, _ = cm.objective_terms(pc, [1e4] * 3, 3e5, 3e5, cfg)
        assert math.isclose(out["T"][k], pc.t_total, rel_tol=1e-9)
        assert math.isclose(out["E_u"][k], pc.e_total_u, rel_tol=1e-9)
        assert math.isclose(out["e_ratio"][k], er, rel_tol=1e-9)
        assert math.isclose(out["t_norm"][k], tn, rel_tol=1e-9)
        assert math.isclose(out["T_sync"][k], pc.t_sync, rel_tol=1e-9)
    assert all_arms(6)[:, 2].max() == 5
