"""
Network state of the four-tier MTA-SFL system (Sec. III).

Indexing
--------
* IoT devices      : 0 .. N-1
* UAVs             : 0 .. U-1, of which 0 .. P-1 are primary UAVs (U_pri)
* cloud server     : separate gateway position (backhaul end-point)

Responsibilities
----------------
* Gauss-Markov mobility, quasi-static per round (Sec. III-B)
* capacity-aware SNR association with fairness queue + tie-breaking (Eq. 1)
* OFDMA sub-channel assignment and inter-cluster interference (Eq. 2)
* directional rates (Eq. 3) for UL / DL, ground and aerial links, backhaul
* aerial topology, local clustering coefficient (Eq. 5)
* predicted link lifetime L_life and queue load Q_load (DDQN state, Eq. 21)
* feasible-action sets U_adj, U_cloud (Sec. VI-A-2)
* beacon control-overhead accounting (R1.8 / R3.10)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from . import channel as chm


BEACON_BYTES_PER_FIELD = 4          # float32 fields exchanged by beaconing
BEACON_FIELDS = 8                   # id, pos(3), vel(2), E_res, Q_load


@dataclass
class Association:
    a: np.ndarray            # (N,) primary index or -1 (unassociated)
    subch: np.ndarray        # (N,) OFDMA sub-channel index within cluster
    K: np.ndarray            # (P,) number of devices per primary
    queue_order: np.ndarray  # order in which devices were processed


class Network:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        tp, hw = cfg.topology, cfg.hw
        self.N, self.P = tp.n_iot, tp.n_primary
        self.U = tp.n_primary + tp.n_relay
        A = tp.area_m

        # ---------- IoT placement around P hotspots ----------
        margin = 0.15 * A
        self.hotspots = rng.uniform(margin, A - margin, size=(self.P, 2))
        lab = rng.integers(0, self.P, size=self.N)
        xy = self.hotspots[lab] + rng.normal(0, tp.iot_cluster_std_m, size=(self.N, 2))
        self.iot_xy = np.clip(xy, 0, A)

        # ---------- UAV placement ----------
        self.uav_xyz = np.zeros((self.U, 3))
        self.uav_xyz[: self.P, :2] = self.hotspots
        self.uav_xyz[: self.P, 2] = tp.primary_alt_m
        self.uav_xyz[self.P:, :2] = rng.uniform(0, A, size=(self.U - self.P, 2))
        self.uav_xyz[self.P:, 2] = rng.uniform(*tp.relay_alt_range_m, size=self.U - self.P)
        self.is_primary = np.zeros(self.U, bool)
        self.is_primary[: self.P] = True
        # Gauss-Markov mean velocity vectors
        mean_speed = np.where(self.is_primary, tp.primary_mean_speed, tp.relay_mean_speed)
        self.speed_std = np.where(self.is_primary, tp.primary_speed_std, tp.relay_speed_std)
        heading = rng.uniform(0, 2 * np.pi, self.U)
        self.v_mean = np.stack([np.cos(heading), np.sin(heading)], 1) * mean_speed[:, None]
        self.vel = self.v_mean.copy()

        # ---------- cloud gateway ----------
        ang = rng.uniform(0, 2 * np.pi)
        c = np.array([A / 2, A / 2])
        self.gw_xyz = np.array([*(c + tp.cloud_distance_m * np.array([np.cos(ang), np.sin(ang)])), 30.0])

        # ---------- hardware ----------
        self.f_iot = rng.uniform(*hw.iot_f_range, size=self.N)
        self.f_uav = rng.uniform(*hw.uav_f_range, size=self.U)
        lo = hw.init_energy_frac_lo
        self.E_iot = rng.uniform(lo, 1.0, self.N) * hw.iot_battery_j
        self.E_uav = rng.uniform(lo, 1.0, self.U) * hw.uav_battery_j
        self.E_iot0, self.E_uav0 = self.E_iot.copy(), self.E_uav.copy()

        # ---------- dynamic flags ----------
        self.failed_until = np.zeros(self.U, int)     # injected failures
        self.prev_dropped = np.zeros(self.N, bool)    # association fairness
        self.q_load = np.zeros(self.U)                # last-round relay usage
        self.t = 0
        self.beacon_bytes_round = 0.0

    # ================================================================== #
    #  liveness                                                          #
    # ================================================================== #
    def uav_alive(self) -> np.ndarray:
        crit = self.cfg.hw.critical_energy_frac * self.cfg.hw.uav_battery_j
        return (self.E_uav > crit) & (self.failed_until <= self.t)

    def iot_alive(self) -> np.ndarray:
        return self.E_iot > 0

    # ================================================================== #
    #  mobility                                                          #
    # ================================================================== #
    def step_mobility(self, dt: float | None = None):
        tp = self.cfg.topology
        dt = tp.round_slot_s if dt is None else dt
        a = tp.gm_alpha
        noise = self.rng.standard_normal((self.U, 2)) * self.speed_std[:, None]
        self.vel = a * self.vel + (1 - a) * self.v_mean + np.sqrt(1 - a ** 2) * noise
        new = self.uav_xyz[:, :2] + self.vel * dt
        # primaries are tethered to their hotspot (hover with drift)
        tether = tp.iot_cluster_std_m      # hover over the cluster it serves
        for j in range(self.P):
            off = new[j] - self.hotspots[j]
            if np.linalg.norm(off) > tether:
                self.vel[j] *= -1
                self.v_mean[j] *= -1
                new[j] = self.uav_xyz[j, :2]
        # reflect at area boundary
        A = tp.area_m
        for d in range(2):
            lo, hi = new[:, d] < 0, new[:, d] > A
            out = lo | hi
            self.vel[out, d] *= -1
            self.v_mean[out, d] *= -1
            new[:, d] = np.clip(new[:, d], 0, A)
        self.uav_xyz[:, :2] = new

    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    # ================================================================== #
    #  association (Eq. 1)                                               #
    # ================================================================== #
    def iot_snr_assoc(self) -> np.ndarray:
        """Expected SNR (N,P) on a worst-case per-device share W_UL/K_max."""
        ch, hw = self.cfg.channel, self.cfg.hw
        g = chm.a2g_gain(self.iot_xy[:, None, :], self.uav_xyz[None, : self.P, :], ch)
        bw = ch.w_ul_hz / self.cfg.sfl.k_max_per_uav
        return hw.iot_ptx_w * g / chm.noise_power_w(bw, ch)

    def associate(self) -> Association:
        ch, sfl = self.cfg.channel, self.cfg.sfl
        snr = self.iot_snr_assoc()
        snr_db = chm.lin2db(snr)
        alive_p = self.uav_alive()[: self.P]
        iot_ok = self.iot_alive()
        # fairness queue: previously dropped devices first, rest random
        perm = self.rng.permutation(self.N)
        order = np.concatenate([perm[self.prev_dropped[perm]], perm[~self.prev_dropped[perm]]])
        c_res = np.full(self.P, sfl.k_max_per_uav)
        a = -np.ones(self.N, int)
        subch = -np.ones(self.N, int)
        tie_tol_db = 1e-2
        for i in order:
            if not iot_ok[i]:
                continue
            cand = np.where((c_res > 0) & (snr_db[i] >= ch.gamma_min_db) & alive_p)[0]
            if cand.size == 0:
                continue
            best = cand[snr_db[i, cand] >= snr_db[i, cand].max() - tie_tol_db]
            tied = best[c_res[best] == c_res[best].max()]
            j = self.rng.choice(tied)
            a[i] = j
            subch[i] = sfl.k_max_per_uav - c_res[j]
            c_res[j] -= 1
        K = np.array([(a == j).sum() for j in range(self.P)])
        return Association(a=a, subch=subch, K=K, queue_order=order)

    # ================================================================== #
    #  ground links: SINR with inter-cluster interference (Eq. 2)        #
    # ================================================================== #
    def _overlap(self, asc: Association) -> np.ndarray:
        """W[i,k]: fraction of device i's sub-band overlapped by device k
        (k in a *different* cluster on the same frequency block)."""
        ch = self.cfg.channel
        a, s, K = asc.a, asc.subch, asc.K
        on = a >= 0
        blk = np.where(on, a % ch.reuse_blocks, -1)
        lo = np.where(on, s / np.maximum(K[np.maximum(a, 0)], 1), 0.0)
        hi = np.where(on, (s + 1) / np.maximum(K[np.maximum(a, 0)], 1), 0.0)
        ov = np.clip(np.minimum(hi[:, None], hi[None, :]) - np.maximum(lo[:, None], lo[None, :]), 0, None)
        width = np.maximum(hi - lo, 1e-12)
        W = ov / width[:, None]
        same_blk = (blk[:, None] == blk[None, :]) & on[:, None] & on[None, :]
        diff_cluster = a[:, None] != a[None, :]
        return W * same_blk * diff_cluster

    def ground_sinr(self, asc: Association, rng=None):
        """Returns (sinr_ul, sinr_dl) per device (0 where unassociated).
        rng=None -> expected channel."""
        ch, hw = self.cfg.channel, self.cfg.hw
        a = asc.a
        on = a >= 0
        G = chm.a2g_gain(self.iot_xy[:, None, :], self.uav_xyz[None, : self.P, :], ch, rng)  # (N,P)
        Kd = np.maximum(asc.K[np.maximum(a, 0)], 1)
        idx = np.arange(self.N)
        jj = np.maximum(a, 0)
        sig_ul = hw.iot_ptx_w * G[idx, jj]
        Wov = self._overlap(asc)
        # UL: interferer k (other cluster) received at primary of i
        I_ul = (Wov * (hw.iot_ptx_w * G[:, jj].T)).sum(1)  # G[k, a_i] -> (i,k)
        n_ul = chm.noise_power_w(ch.w_ul_hz / Kd, ch)
        sinr_ul = np.where(on, sig_ul / (n_ul + I_ul), 0.0)
        # DL: primary j' on the same block broadcasting concurrently (full overlap)
        blk_p = np.arange(self.P) % ch.reuse_blocks
        alive_p = self.uav_alive()[: self.P] & (asc.K > 0)
        sig_dl = hw.uav_ptx_w * G[idx, jj]
        interf_mask = (blk_p[None, :] == blk_p[jj][:, None]) & (np.arange(self.P)[None, :] != jj[:, None]) & alive_p[None, :]
        I_dl = (interf_mask * hw.uav_ptx_w * G).sum(1)
        n_dl = chm.noise_power_w(ch.w_dl_hz / Kd, ch)
        sinr_dl = np.where(on, sig_dl / (n_dl + I_dl), 0.0)
        return sinr_ul, sinr_dl

    def ground_rates(self, asc: Association, rng=None):
        ch = self.cfg.channel
        sinr_ul, sinr_dl = self.ground_sinr(asc, rng)
        Kd = np.maximum(asc.K[np.maximum(asc.a, 0)], 1)
        on = asc.a >= 0
        r_ul = np.where(on, chm.shannon(ch.w_ul_hz / Kd, sinr_ul), 0.0)
        r_dl = np.where(on, chm.shannon(ch.w_dl_hz / Kd, sinr_dl), 0.0)
        return r_ul, r_dl, sinr_ul, sinr_dl

    # ================================================================== #
    #  aerial links                                                      #
    # ================================================================== #
    def a2a_snr(self, bw_hz=None, rng=None) -> np.ndarray:
        """(U,U) SNR between UAVs on bandwidth bw (beacon: full W_a2a)."""
        ch, hw = self.cfg.channel, self.cfg.hw
        bw = ch.w_a2a_ul_hz if bw_hz is None else bw_hz
        G = chm.a2a_gain(self.uav_xyz[:, None, :], self.uav_xyz[None, :, :], ch, rng)
        if rng is not None:  # reciprocal fading
            G = np.triu(G, 1)
            G = G + G.T
        snr = hw.uav_ptx_w * G / chm.noise_power_w(bw, ch)
        np.fill_diagonal(snr, 0.0)
        return snr

    def adjacency(self) -> np.ndarray:
        """e_{m,n}(t)=1 iff SINR >= gamma_th (Sec. III-B), alive UAVs only."""
        snr_db = chm.lin2db(self.a2a_snr())
        alive = self.uav_alive()
        A = (snr_db >= self.cfg.channel.gamma_th_db) & alive[:, None] & alive[None, :]
        np.fill_diagonal(A, False)
        return A

    @staticmethod
    def clustering_coefficient(A: np.ndarray) -> np.ndarray:
        """Eq. (5): directed local clustering coefficient."""
        U = A.shape[0]
        cc = np.zeros(U)
        for u in range(U):
            nb = np.where(A[u])[0]
            k = nb.size
            if k >= 2:
                cc[u] = A[np.ix_(nb, nb)].sum() / (k * (k - 1))
        return cc

    def backhaul(self):
        """Deterministic backhaul SNR / rate for every UAV, and U_cloud mask."""
        ch = self.cfg.channel
        snr, rate = chm.backhaul(self.uav_xyz, self.gw_xyz, ch)
        ok = (chm.lin2db(snr) >= ch.bh_snr_min_db) & self.uav_alive()
        return snr, rate, ok

    # ------------------------------------------------------------------ #
    def _range_for_snr(self, thr_db, p_w, bw_hz, alpha, gain_db=0.0) -> float:
        """Distance at which the mean SNR equals thr_db (log-distance PL)."""
        ch = self.cfg.channel
        n_db = chm.lin2db(chm.noise_power_w(bw_hz, ch))
        pl_max = 10 * np.log10(p_w) + gain_db - n_db - thr_db
        return 10 ** ((pl_max - chm.fspl_db(1.0, ch.fc_hz)) / (10 * alpha))

    @staticmethod
    def _let(dp, dv, r, cap):
        """Link expiration time for |dp + dv t| = r (Su et al.)."""
        a = np.sum(dv * dv, -1)
        b = np.sum(dp * dv, -1)
        c = np.sum(dp * dp, -1) - r ** 2
        disc = b ** 2 - a * c
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (-b + np.sqrt(np.maximum(disc, 0))) / np.where(a > 1e-9, a, np.nan)
        t = np.where(a > 1e-9, t, cap)
        t = np.where(c > 0, 0.0, t)            # already out of range
        return np.clip(np.nan_to_num(t, nan=cap), 0, cap)

    def link_lifetime(self, j: int) -> np.ndarray:
        """Predicted route lifetime L_life for primary j via each UAV:
        min(LET(j,u), LET(u, gateway)) under constant-velocity prediction."""
        tp, ch, hw = self.cfg.topology, self.cfg.channel, self.cfg.hw
        r_a2a = self._range_for_snr(ch.gamma_th_db, hw.uav_ptx_w, ch.w_a2a_ul_hz, ch.a2a_pl_exp)
        v3 = np.concatenate([self.vel, np.zeros((self.U, 1))], 1)
        dp = self.uav_xyz - self.uav_xyz[j]
        dv = v3 - v3[j]
        let_a2a = self._let(dp, dv, r_a2a, tp.lifetime_cap_s)
        r_bh = self._range_for_snr(ch.bh_snr_min_db, ch.bh_tx_power_w, ch.bh_bandwidth_hz,
                                   ch.bh_pl_exp, ch.bh_antenna_gain_db)
        let_bh = self._let(self.uav_xyz - self.gw_xyz, v3, r_bh, tp.lifetime_cap_s)
        return np.minimum(let_a2a, let_bh)

    # ================================================================== #
    #  per-round snapshot used by agents                                 #
    # ================================================================== #
    def relay_candidates_mask(self) -> np.ndarray:
        m = self.uav_alive().copy()
        if not self.cfg.topology.primaries_can_relay:
            m &= ~self.is_primary
        return m

    def snapshot(self) -> dict:
        """Topology information refreshed at the start of round t via
        beaconing among adjacent UAVs (Sec. VI-A-1)."""
        A = self.adjacency()
        cc = self.clustering_coefficient(A)
        snr_a2a = self.a2a_snr()
        bh_snr, bh_rate, cloud_ok = self.backhaul()
        alive = self.uav_alive()
        # every alive UAV broadcasts one beacon per neighbour per round
        self.beacon_bytes_round = float(A.sum()) * BEACON_FIELDS * BEACON_BYTES_PER_FIELD
        life = np.stack([self.link_lifetime(j) for j in range(self.P)])  # (P,U)
        return dict(A=A, cc=cc, snr_a2a=snr_a2a, bh_snr=bh_snr, bh_rate=bh_rate,
                    cloud_ok=cloud_ok, alive=alive, life=life,
                    q_load=self.q_load.copy(), relay_mask=self.relay_candidates_mask())

    def u_adj(self, j: int, snap: dict) -> np.ndarray:
        thr = self.cfg.channel.gamma_th_db
        return (chm.lin2db(snap["snr_a2a"][j]) >= thr) & snap["relay_mask"] & (np.arange(self.U) != j)

    # ================================================================== #
    #  failures                                                          #
    # ================================================================== #
    def inject_failures(self):
        fc = self.cfg.failure
        if fc.relay_fail_prob <= 0:
            return np.zeros(self.U, bool)
        cand = ~self.is_primary & (self.failed_until <= self.t)
        hit = cand & (self.rng.random(self.U) < fc.relay_fail_prob)
        self.failed_until[hit] = self.t + fc.fail_duration_rounds
        return hit
