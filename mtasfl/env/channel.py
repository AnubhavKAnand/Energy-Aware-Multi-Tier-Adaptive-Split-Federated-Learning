"""
Channel models.

* Air-to-ground (IoT <-> primary UAV): probabilistic LoS model of Al-Hourani
  et al. with free-space path loss + excess loss; small-scale fading is Rician
  (LoS) or Rayleigh (NLoS).
* Air-to-air (UAV <-> UAV): free-space path loss + Rician fading.
* Backhaul (relay UAV -> cloud gateway): dedicated orthogonal channel with a
  deterministic expected rate (Sec. III-A-2(c) of the paper).

"Estimated" quantities use the mean channel (no small-scale fading); "realised"
quantities draw fading at execution time.  The gap between the two is what the
learning agents have to cope with.
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 3e8


def db2lin(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


def fspl_db(d_m, fc_hz):
    d = np.maximum(np.asarray(d_m, dtype=float), 1.0)
    return 20 * np.log10(d) + 20 * np.log10(fc_hz) + 20 * np.log10(4 * np.pi / C_LIGHT)


def noise_power_w(bw_hz, ch) -> np.ndarray:
    n0_w = db2lin(ch.noise_psd_dbm_hz - 30.0 + ch.noise_figure_db)
    return n0_w * np.asarray(bw_hz, dtype=float)


# --------------------------------------------------------------------------- #
def a2g_plos(ground_xy, uav_xyz, ch):
    """LoS probability; ground_xy (...,2), uav_xyz (...,3)."""
    r = np.linalg.norm(ground_xy[..., :2] - uav_xyz[..., :2], axis=-1)
    h = uav_xyz[..., 2]
    theta = np.degrees(np.arctan2(h, np.maximum(r, 1e-6)))
    return 1.0 / (1.0 + ch.a2g_a * np.exp(-ch.a2g_b * (theta - ch.a2g_a)))


def a2g_gain(ground_xy, uav_xyz, ch, rng=None):
    """Linear channel gain h.  rng=None -> expected (mean) gain."""
    d = np.sqrt(np.sum((ground_xy[..., :2] - uav_xyz[..., :2]) ** 2, axis=-1) + uav_xyz[..., 2] ** 2)
    pl0 = fspl_db(d, ch.fc_hz)
    plos = a2g_plos(ground_xy, uav_xyz, ch)
    # mean excess loss (LoS state is position-dependent and slowly varying, so it
    # is averaged rather than redrawn i.i.d. every round)
    pl = pl0 + plos * ch.a2g_eta_los_db + (1 - plos) * ch.a2g_eta_nlos_db
    if rng is None:
        return db2lin(-pl)
    # small-scale Rician fading whose K-factor shrinks with the LoS probability
    k_eff = np.maximum(plos * ch.a2g_rician_k_los, ch.a2g_rician_k_min)
    return db2lin(-pl) * rician_power(k_eff, np.shape(plos), rng)


def logdist_pl_db(d_m, fc_hz, alpha):
    """FSPL at 1 m + 10 alpha log10(d)."""
    d = np.maximum(np.asarray(d_m, dtype=float), 1.0)
    return fspl_db(1.0, fc_hz) + 10 * alpha * np.log10(d)


def a2a_gain(xyz_a, xyz_b, ch, rng=None):
    d = np.linalg.norm(xyz_a - xyz_b, axis=-1)
    g = db2lin(-logdist_pl_db(d, ch.fc_hz, ch.a2a_pl_exp))
    if rng is None:
        return g
    return g * rician_power(ch.a2a_rician_k, np.shape(d), rng)


def rician_power(k, shape, rng):
    """|g|^2 with E[|g|^2]=1 for Rician factor K."""
    s = np.sqrt(k / (k + 1.0))
    sig = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    re = s + sig * rng.standard_normal(shape)
    im = sig * rng.standard_normal(shape)
    return re ** 2 + im ** 2


def backhaul(uav_xyz, gw_xyz, ch):
    """Deterministic backhaul SNR (linear) and rate (bit/s) for each UAV."""
    d = np.linalg.norm(uav_xyz - gw_xyz[None, :], axis=-1)
    g = db2lin(-logdist_pl_db(d, ch.fc_hz, ch.bh_pl_exp) + ch.bh_antenna_gain_db)
    snr = ch.bh_tx_power_w * g / noise_power_w(ch.bh_bandwidth_hz, ch)
    rate = ch.bh_bandwidth_hz * np.log2(1.0 + snr)
    return snr, rate


def shannon(bw_hz, sinr):
    """Eq. (3):  R = W' log2(1 + SINR)."""
    return np.asarray(bw_hz, dtype=float) * np.log2(1.0 + np.maximum(sinr, 0.0))
