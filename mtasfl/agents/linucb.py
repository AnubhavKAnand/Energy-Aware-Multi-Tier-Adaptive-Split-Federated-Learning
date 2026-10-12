"""
Route-specific LinUCB adaptive split selection (Sec. VI-B, Algorithm 3).

Context x_t,j (Eq. 27, d = 12):
  [E_res,i*, E_res,j, E_res,u, f_i*, f_j, f_u,
   R^UL_i*,j, R^UL_j,u, R^UL_u,S, R^DL_j,i*, R^DL_u,j, R^DL_S,u]
  i* = bottleneck IoT device (largest estimated tier-1 upload time).
Normalisation: x_bar = (1/sqrt(d)) (2 (x - x_min)/(x_max - x_min) - 1)
  => ||x_bar||_2 <= 1  (Lemma 2).
Disjoint linear model per arm p:  UCB = x^T theta_p + alpha_t,d sqrt(x^T A_p^-1 x)
  alpha_t,d = R sqrt(d ln(1 + t/d) + 2 ln(|P|/delta)) + 1   (Lemma 2)
Selection restricted to P_feasible(t) (Eq. 31).  Update Eq. (32)-(34).
"""
from __future__ import annotations

import math
import time

import numpy as np

D_CTX = 12


class LinUCBAgent:
    def __init__(self, j: int, arms: np.ndarray, cfg, bounds: tuple[np.ndarray, np.ndarray]):
        self.j, self.arms, self.cfg = j, arms, cfg
        self.lc = cfg.linucb
        n = len(arms)
        self.A_inv = np.repeat(np.eye(D_CTX)[None], n, 0)       # A_p^-1 (A_p = I_d)
        self.b = np.zeros((n, D_CTX))
        self.theta = np.zeros((n, D_CTX))
        self.n_pulls = np.zeros(n, int)
        self.t = 0
        self.lo, self.hi = bounds
        self.last_width = float("nan")
        self.last_alpha = float("nan")
        self.t_decide = 0.0

    def normalise(self, x_raw: np.ndarray) -> np.ndarray:
        z = (np.clip(x_raw, self.lo, self.hi) - self.lo) / np.maximum(self.hi - self.lo, 1e-12)
        return (2 * z - 1) / math.sqrt(D_CTX)

    def alpha(self) -> float:
        lc = self.lc
        t = max(self.t, 1)
        a = lc.noise_R * math.sqrt(D_CTX * math.log(1 + t / D_CTX)
                                   + 2 * math.log(len(self.arms) / lc.delta)) + 1.0
        return lc.alpha_scale * a

    def act(self, x_bar: np.ndarray, feasible: np.ndarray):
        """Algorithm 3: argmax over P_feasible of the UCB; NULL if empty."""
        t0 = time.perf_counter()
        if not feasible.any():
            self.t_decide = time.perf_counter() - t0
            return None
        idx = np.where(feasible)[0]
        th = self.theta[idx] @ x_bar
        width = np.sqrt(np.einsum("i,nij,j->n", x_bar, self.A_inv[idx], x_bar))
        al = self.alpha()
        ucb = th + al * width
        k = idx[int(np.argmax(ucb))]
        self.last_width = float(width[np.argmax(ucb)])
        self.last_alpha = al
        self.t_decide = time.perf_counter() - t0
        return int(k)

    def update(self, k: int, x_bar: np.ndarray, r: float):
        """Eq. (32)-(34) with a Sherman-Morrison rank-1 update of A^-1."""
        self.t += 1
        Ai = self.A_inv[k]
        Ax = Ai @ x_bar
        self.A_inv[k] = Ai - np.outer(Ax, Ax) / (1.0 + x_bar @ Ax)
        self.b[k] += r * x_bar
        self.theta[k] = self.A_inv[k] @ self.b[k]
        self.n_pulls[k] += 1

    def memory_bytes(self) -> dict:
        n = len(self.arms)
        return dict(arms=n, matrix_bytes=int(n * D_CTX * D_CTX * 8 + 2 * n * D_CTX * 8))

    def flops_per_decision(self, n_feasible=None) -> int:
        n = len(self.arms) if n_feasible is None else n_feasible
        return int(n * (2 * D_CTX * D_CTX + 4 * D_CTX))

    def state_dict(self):
        return dict(A_inv=self.A_inv, b=self.b, theta=self.theta, n=self.n_pulls, t=self.t)

    def load_state_dict(self, d):
        self.A_inv, self.b, self.theta, self.n_pulls, self.t = d["A_inv"], d["b"], d["theta"], d["n"], d["t"]


def context_bounds(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Fixed physical min/max of each context feature (for min-max scaling)."""
    hw, ch = cfg.hw, cfg.channel
    lo = np.array([0, 0, 0, hw.iot_f_range[0], hw.uav_f_range[0], hw.uav_f_range[0],
                   0, 0, 0, 0, 0, 0], float)
    r_g, r_a, r_b = 30e6, 400e6, 300e6
    hi = np.array([hw.iot_battery_j, hw.uav_battery_j, hw.uav_battery_j,
                   hw.iot_f_range[1], hw.uav_f_range[1], hw.uav_f_range[1],
                   r_g, r_a, r_b, r_g, r_a, r_b], float)
    return lo, hi
