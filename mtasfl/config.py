"""
Central configuration for MTA-SFL.

Every number used anywhere in the simulator lives here, so that the paper's
simulation-parameter table (R2.9, R3.14) can be exported directly from it
(see `export_param_table`).  Values marked [ASSUMED] are not given in the
paper and are literature-backed defaults that can be overridden from YAML.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
#  Topology / mobility                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class TopologyCfg:
    area_m: float = 1000.0                 # square side length            [ASSUMED]
    n_iot: int = 50                        # |N| static IoT devices        [ASSUMED]
    n_primary: int = 5                     # |U_pri|                        [ASSUMED]
    n_relay: int = 15                      # |U| - |U_pri| relay-capable    [ASSUMED]
    iot_cluster_std_m: float = 80.0        # IoT devices drawn around hotspots
    primary_alt_m: float = 100.0           # primary UAV altitude
    relay_alt_range_m: tuple = (120.0, 180.0)
    # Gauss-Markov mobility (relays move; primaries hover with small drift)
    gm_alpha: float = 0.8                  # memory                         [ASSUMED]
    relay_mean_speed: float = 10.0         # m/s                            [ASSUMED]
    relay_speed_std: float = 3.0
    primary_mean_speed: float = 1.0        # m/s hover drift                [ASSUMED]
    primary_speed_std: float = 0.5
    round_slot_s: float = 5.0              # wall-time advanced per round for mobility
    primaries_can_relay: bool = False      # relay candidates = non-primary UAVs
    lifetime_cap_s: float = 600.0          # cap for predicted link lifetime
    # cloud gateway (backhaul end-point); distance measured from area centre
    cloud_distance_m: float = 2000.0       # varied in "cloud distance" sweep


# --------------------------------------------------------------------------- #
#  Channel / PHY                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ChannelCfg:
    fc_hz: float = 2.4e9
    noise_psd_dbm_hz: float = -174.0       # N0
    noise_figure_db: float = 7.0
    # Air-to-ground probabilistic LoS (Al-Hourani, urban)
    a2g_a: float = 9.61
    a2g_b: float = 0.16
    a2g_eta_los_db: float = 1.0
    a2g_eta_nlos_db: float = 20.0
    # Air-to-air: log-distance PL = FSPL(1 m) + 10*alpha*log10(d) + Rician
    a2a_pl_exp: float = 2.6                # gives ~375 m range at defaults [ASSUMED]
    a2a_rician_k: float = 10.0
    a2g_rician_k_los: float = 5.0          # small-scale fading on A2G LoS
    a2g_rician_k_min: float = 1.0          # K-factor floor (static IoT, hovering UAV)
    # Backhaul (relay UAV -> cloud gateway), dedicated orthogonal channel
    bh_bandwidth_hz: float = 20e6
    bh_tx_power_w: float = 1.0
    bh_antenna_gain_db: float = 15.0       # directional gateway antenna
    bh_pl_exp: float = 2.7                 # log-distance exponent       [ASSUMED]
    bh_snr_min_db: float = 0.0             # defines U_cloud(t)
    # Bandwidths (Eq. 3)
    w_ul_hz: float = 20e6                  # W_UL per primary-UAV cluster (802.11ax 20 MHz)
    w_dl_hz: float = 20e6                  # W_DL per primary-UAV cluster
    w_a2a_ul_hz: float = 20e6              # primary -> relay
    w_a2a_dl_hz: float = 20e6              # relay -> primary
    # Inter-cluster interference: clusters are assigned to `reuse_blocks`
    # frequency blocks round-robin.  reuse_blocks >= n_primary  => orthogonal
    # (no inter-cluster interference); 1 => full reuse (worst case, R4.1).
    reuse_blocks: int = 2
    # thresholds
    gamma_min_db: float = 0.0              # IoT decoding threshold  (Eq.1)
    gamma_th_db: float = 10.0              # aerial decoding threshold (Sec III-B)


# --------------------------------------------------------------------------- #
#  Hardware / energy                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class HardwareCfg:
    # effective compute throughput (FLOP/s); latency = FLOPs / f  (Sec IV-A)
    iot_f_range: tuple = (1.0e9, 2.0e9)        # Raspberry-Pi-class [ASSUMED]
    uav_f_range: tuple = (100e9, 200e9)       # Jetson-class      [ASSUMED]
    cloud_f: float = 1e12
    cloud_j_per_flop: float = 3e-11        # reported only (cloud is mains-powered) [ASSUMED]
    # effective switched capacitance:  E = kappa * f^2 * FLOPs, calibrated so
    # that kappa*f^2 ~ 2e-9 J/FLOP (IoT) and ~1.5e-10 J/FLOP (UAV)   [ASSUMED]
    kappa_iot: float = 9e-28
    kappa_uav: float = 6.7e-33
    # RF powers
    iot_ptx_w: float = 0.1
    iot_prx_w: float = 0.05
    uav_ptx_w: float = 0.2
    uav_prx_w: float = 0.2
    # circuit / sleep
    iot_pcir_w: float = 0.1
    iot_psleep_w: float = 1e-3
    uav_pcir_w: float = 5.0
    # propulsion (rotary-wing, Zeng et al. 2019).  P_flight is the average
    # mechanical power (Eq. 10).  'constant' uses the model at mean speed.
    prop_model: str = "constant"           # 'constant' | 'zeng'
    zeng_p0: float = 79.86
    zeng_pi: float = 88.63
    zeng_utip: float = 120.0
    zeng_v0: float = 4.03
    zeng_d0: float = 0.6
    zeng_rho: float = 1.225
    zeng_s: float = 0.05
    zeng_A: float = 0.503
    # batteries (J).  Initial residual energy ~ U[frac_lo, 1] * capacity.
    uav_battery_j: float = 300e3           # ~83 Wh
    recharge_time_s: float = 300.0         # depleted UAV offline (battery swap), then full
    iot_battery_j: float = 20e3
    init_energy_frac_lo: float = 0.4
    critical_energy_frac: float = 0.05     # below -> UAV considered dead
    # per-round compute-time jitter (realised vs estimated), lognormal sigma
    compute_jitter_sigma: float = 0.1


# --------------------------------------------------------------------------- #
#  SFL                                                                         #
# --------------------------------------------------------------------------- #
T_MAX_DEFAULT = {"cifar10": 12.0, "cifar100": 12.0, "mnist": 2.0, "fmnist": 2.0, "ucihar": 1.5}


@dataclass
class SFLCfg:
    k_max_per_uav: int = 12                # K_max (Eq.1)
    batch_per_device: int = 16             # b_i
    t_max_s: float = 0.0                   # T_max deadline (Eq.17b); 0 -> per-dataset default
    iot_deadline_frac: float = 0.5         # local IoT link deadline (compute+upload+grad return) = frac*T_max
    mask_margin: float = 0.15              # safety margin on estimates in masks (Eq.31, U_safe)
    topk_ratio: float = 0.1                # k / D
    float_bits: int = 32                   # w
    # Model synchronisation traffic (layer weights up to the cloud for Eq. (4)
    # aggregation and back to the nodes that run them).  Applied identically to
    # every method: MTA-SFL syncs every round (Eq. 4), baselines follow their own
    # aggregation schedule.  Not counted against the T_max training deadline,
    # but added to the round duration and to the energy.  False -> paper Eq. (9).
    model_sync: bool = True
    lr: float = 0.05
    lr_warmup_rounds: int = 100            # linear warm-up of eta (ResNet-GN stability)
    momentum: float = 0.0                  # Eq. (4) is plain SGD; set >0 to add momentum
    weight_decay: float = 0.0
    rounds: int = 2000
    partition: str = "iid"                 # 'iid' | 'dirichlet'
    dirichlet_alpha: float = 0.5
    eval_every: int = 20


# --------------------------------------------------------------------------- #
#  Objective / agents                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ObjectiveCfg:
    lam1: float = 0.4                      # latency
    lam2: float = 0.4                      # energy
    lam3: float = 0.2                      # clustering coefficient
    psi_penalty: float = 10.0              # Psi = M (terminal penalty)
    # propulsion time used inside E_ratio: 'round' (paper, T_round) | 'pipeline'
    prop_time: str = "round"
    energy_ratio_clip: float = 10.0        # cap on E/E_res inside rewards


@dataclass
class DDQNCfg:
    hidden: tuple = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.9
    batch: int = 64
    replay: int = 20000
    target_sync: int = 50                  # rounds
    warmup: int = 64
    beta: float = 0.5                      # pseudo-count bonus coefficient
    n_bins: int = 5                        # discretisation for N_hat
    eps_start: float = 0.0                 # paper explores via pseudo-counts only
    eps_end: float = 0.0
    eps_decay_rounds: int = 1


@dataclass
class LinUCBCfg:
    delta: float = 0.05
    noise_R: float = 0.1                   # sub-Gaussian scale in alpha_t,delta
    alpha_scale: float = 0.1               # multiplier on alpha_t,delta (tuned, see README)
    # split candidates: block indices; arms = all 1<=p1<=p2<=p3<=L
    min_p1: int = 1


@dataclass
class AdaptSFLCfg:
    """AdaptSFL (Lin et al., IEEE ToN 2025) - Alg. 3 BCD over (I, mu)."""
    I_cap: int = 25                 # search range of I (paper's RMA range 1..25)
    beta: float = 1.0               # smoothness constant of Assumption 1   [ASSUMED]
    bcd_iters: int = 4
    cd_passes: int = 2              # coordinate passes over per-device cuts
    probe_devices: int = 4          # devices used to estimate G_j^2, sigma_j^2


@dataclass
class SFLSC3Cfg:
    """SFLSC3 (Hou et al., 2026) - BCD over (I, Lc, b, UAV position)."""
    I_cap: int = 100                # paper evaluates I in {1, 10, 50, 100}
    b_choices: tuple = (8, 16, 24, 32)   # paper's data-sensing volume range
    beta: float = 1.0                                                     # [ASSUMED]
    bcd_iters: int = 4
    probe_devices: int = 4
    optimise_position: bool = True  # primary UAV hovers at the rate-optimal point


@dataclass
class SHSFLCfg:
    """S-HSFL (Gao et al., Drones 2026) - MAB-GT selection + verification."""
    cut: int = 1                    # split after conv1 (paper Sec. 6.1)
    select_frac: float = 0.5        # K / |candidates|  (paper: K=10 of N=100)
    rho: float = 0.5                # split-learning share of selected (Ks = Kf)
    beta: float = 0.5               # Eq. (6): weight of channel vs contribution
    beta_g: float = 0.5             # Eq. (3) utility weight
    alpha: float = 1.0              # Eq. (3) energy cost weight
    sigma_ucb: float = 0.5          # Alg. 1 line 14 exploration coefficient
    gamma_shapley: float = 0.2
    delta_hist: float = 0.2
    omega_hist: float = 0.5         # Eq. (11)
    lam_l: float = 0.9              # EMA of contribution (Alg. 1 line 19)
    lam_c: float = 0.9              # EMA of SNR (Alg. 1 line 20)
    trust_decay: float = 0.9        # Eq. (8)
    trust_gain: float = 0.05        # Eq. (9)
    coalition_period: int = 10      # T_coalition
    n_coalitions: int = 3
    challenge_samples: int = 32     # |D_chal| (verification proof)
    sign_time_s: float = 0.005      # digital signature on an IoT-class CPU [ASSUMED]
    signature_bits: int = 512


@dataclass
class FailureCfg:
    relay_fail_prob: float = 0.0           # per-UAV per-round crash probability
    fail_duration_rounds: int = 20
    link_fade_prob: float = 0.0            # extra mid-round link loss prob


@dataclass
class Config:
    seed: int = 0
    dataset: str = "cifar10"
    model: str = "auto"
    device: str = "cpu"
    mode: str = "full"                     # 'full' (real training) | 'fast'
    topology: TopologyCfg = field(default_factory=TopologyCfg)
    channel: ChannelCfg = field(default_factory=ChannelCfg)
    hw: HardwareCfg = field(default_factory=HardwareCfg)
    sfl: SFLCfg = field(default_factory=SFLCfg)
    obj: ObjectiveCfg = field(default_factory=ObjectiveCfg)
    ddqn: DDQNCfg = field(default_factory=DDQNCfg)
    linucb: LinUCBCfg = field(default_factory=LinUCBCfg)
    failure: FailureCfg = field(default_factory=FailureCfg)
    adaptsfl: AdaptSFLCfg = field(default_factory=AdaptSFLCfg)
    sflsc3: SFLSC3Cfg = field(default_factory=SFLSC3Cfg)
    shsfl: SHSFLCfg = field(default_factory=SHSFLCfg)
    out_dir: str = "runs/default"
    # Offline pre-training of MTA-SFL's DDQN/LinUCB agents in the simulator
    # (fast mode, different topology seed) before online deployment; 0 = off.
    agent_pretrain_rounds: int = 0

    # ------------------------------------------------------------------ #
    def t_max(self) -> float:
        """Deadline T_max in seconds (dataset default unless set)."""
        if self.sfl.t_max_s > 0:
            return self.sfl.t_max_s
        return T_MAX_DEFAULT.get(self.dataset.split("-")[0], 6.0)

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, d: dict[str, Any]) -> "Config":
        """Return a copy with nested overrides, e.g. {'sfl': {'topk_ratio': 0.05}}
        or dotted keys {'sfl.topk_ratio': 0.05}."""
        new = copy.deepcopy(self)
        for k, v in d.items():
            if "." in k:
                head, *rest = k.split(".")
                d2 = {".".join(rest): v}
                setattr(new, head, _override_dc(getattr(new, head), d2))
            elif dataclasses.is_dataclass(getattr(new, k, None)) and isinstance(v, dict):
                setattr(new, k, _override_dc(getattr(new, k), v))
            else:
                if not hasattr(new, k):
                    raise KeyError(f"unknown config key {k}")
                setattr(new, k, v)
        return new

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return Config().override(d)


def _override_dc(dc, d: dict):
    dc = copy.deepcopy(dc)
    for k, v in d.items():
        if "." in k:
            head, *rest = k.split(".")
            setattr(dc, head, _override_dc(getattr(dc, head), {".".join(rest): v}))
            continue
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key {k} in {type(dc).__name__}")
        cur = getattr(dc, k)
        if isinstance(cur, tuple) and isinstance(v, list):
            v = tuple(v)
        setattr(dc, k, v)
    return dc


def export_param_table(cfg: Config) -> str:
    """Markdown table of every simulation parameter (for the paper)."""
    rows = []
    for sec, val in cfg.to_dict().items():
        if isinstance(val, dict):
            for k, v in val.items():
                rows.append(f"| {sec} | {k} | {v} |")
        else:
            rows.append(f"| general | {sec} | {val} |")
    return "| Section | Parameter | Value |\n|---|---|---|\n" + "\n".join(rows)
