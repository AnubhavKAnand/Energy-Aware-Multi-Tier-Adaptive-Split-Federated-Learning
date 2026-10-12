"""
DDQN-based topological relay selection with pseudo-count exploration
(Sec. VI-A, Algorithm 2).

State  s_t,j (Eq. 21): for every UAV slot u the tuple
       [CC_u, E_res,u / E_cap, SINR_j,u (dB, scaled), L_life,u / cap, Q_load,u / |U_pri|]
       -> fixed-size vector of length 5 |U| (non-candidate slots are masked
       out of the action space, not of the state).
Action u_t,j in A = U_adj,j  ∩  U_cloud  ∩  U_safe.
Reward r_total = r_ext (Eq. 22) + beta / sqrt(N_hat(s,u)) (Eq. 23-24).
Target Y = r + gamma Q'(s', argmax_{u' in A'} Q(s',u')) (Eq. 25); no bootstrap
       on terminal transitions (aborted pipeline / empty next action space).
Loss  MSE (Eq. 26); target net synchronised every `target_sync` rounds.
"""
from __future__ import annotations

import time
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn

N_FEAT = 5


class QNet(nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_state(j, snap, net, cfg) -> np.ndarray:
    """Eq. (21) local state of primary j, shape (U, 5)."""
    U = net.U
    snr_db = 10 * np.log10(np.maximum(snap["snr_a2a"][j], 1e-12))
    feat = np.zeros((U, N_FEAT), np.float32)
    feat[:, 0] = snap["cc"]
    feat[:, 1] = np.clip(net.E_uav / cfg.hw.uav_battery_j, 0, 1)
    feat[:, 2] = np.clip((snr_db + 10.0) / 50.0, 0, 1)
    feat[:, 3] = snap["life"][j] / cfg.topology.lifetime_cap_s
    feat[:, 4] = snap["q_load"] / max(net.P, 1)
    feat[~snap["alive"]] = 0.0
    return feat


class DDQNAgent:
    def __init__(self, j: int, n_uav: int, cfg, seed: int, device="cpu", use_pseudo_count=True):
        self.j, self.U, self.cfg = j, n_uav, cfg
        dc = cfg.ddqn
        self.dc = dc
        torch.manual_seed(seed * 1000 + j)
        self.q = QNet(N_FEAT * n_uav, n_uav, dc.hidden).to(device)
        self.qt = QNet(N_FEAT * n_uav, n_uav, dc.hidden).to(device)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=dc.lr)
        self.replay = deque(maxlen=dc.replay)
        self.counts = defaultdict(int)
        self.rng = np.random.default_rng(seed * 7919 + j)
        self.device = device
        self.use_pc = use_pseudo_count
        self.pending = None             # (s, u, r) waiting for s'
        self.steps = 0
        self.last_loss = float("nan")
        self.last_q = float("nan")
        self.t_decide = 0.0
        self.t_train = 0.0

    # ------------------------------------------------------------------ #
    def _key(self, feat, u):
        nb = self.dc.n_bins
        cell = tuple(np.minimum((feat[u] * nb).astype(int), nb - 1))
        return (u,) + cell

    def intrinsic(self, feat, u) -> float:
        """Eq. (23): beta / sqrt(N_hat(s,u)), with N_hat incremented first."""
        if not self.use_pc:
            return 0.0
        k = self._key(feat, u)
        self.counts[k] += 1
        return self.dc.beta / np.sqrt(self.counts[k])

    def epsilon(self) -> float:
        dc = self.dc
        frac = min(1.0, self.steps / max(dc.eps_decay_rounds, 1))
        return dc.eps_start + frac * (dc.eps_end - dc.eps_start)

    # ------------------------------------------------------------------ #
    def act(self, feat: np.ndarray, mask: np.ndarray):
        """Algorithm 2: argmax_{u in A} Q(s,u); NULL if A is empty."""
        t0 = time.perf_counter()
        if not mask.any():
            self.t_decide = time.perf_counter() - t0
            return None
        if self.rng.random() < self.epsilon():
            u = int(self.rng.choice(np.where(mask)[0]))
        else:
            with torch.no_grad():
                q = self.q(torch.as_tensor(feat.reshape(1, -1), device=self.device))[0].cpu().numpy()
            q[~mask] = -np.inf
            u = int(np.argmax(q))
            self.last_q = float(q[u])
        self.t_decide = time.perf_counter() - t0
        return u

    # ------------------------------------------------------------------ #
    def close_pending(self, next_feat, next_mask):
        """Complete the previous transition with s_{t+1} (Algorithm 1, Phase V)."""
        if self.pending is None:
            return
        s, u, r, terminal = self.pending
        done = terminal or not next_mask.any()
        self.replay.append((s.reshape(-1), u, r, next_feat.reshape(-1), next_mask.copy(), done))
        self.pending = None

    def record(self, feat, u, r_total, terminal=False):
        if u is None:   # aborted: NULL action, terminal penalty (Alg.1 l.44)
            return
        self.pending = (feat.copy(), u, float(r_total), terminal)

    def train_step(self):
        t0 = time.perf_counter()
        dc = self.dc
        self.steps += 1
        if len(self.replay) < max(dc.warmup, dc.batch):
            self.t_train = time.perf_counter() - t0
            return
        idx = self.rng.integers(0, len(self.replay), dc.batch)
        batch = [self.replay[i] for i in idx]
        s = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        a = torch.as_tensor([b[1] for b in batch], device=self.device)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack([b[3] for b in batch]), device=self.device)
        m2 = torch.as_tensor(np.stack([b[4] for b in batch]), device=self.device)
        done = torch.as_tensor([b[5] for b in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_online = self.q(s2).masked_fill(~m2, -1e9)
            a2 = q_online.argmax(1)                              # selection: online
            q_next = self.qt(s2).gather(1, a2[:, None])[:, 0]    # evaluation: target
            y = r + dc.gamma * (1 - done) * q_next
        q_sa = self.q(s).gather(1, a[:, None])[:, 0]
        loss = nn.functional.mse_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.detach())
        if self.steps % dc.target_sync == 0:
            self.qt.load_state_dict(self.q.state_dict())
        self.t_train = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    def memory_bytes(self) -> dict:
        p = sum(x.numel() for x in self.q.parameters()) * 4
        per_tr = (2 * N_FEAT * self.U + self.U) * 4 + 16
        return dict(qnet_params=sum(x.numel() for x in self.q.parameters()),
                    qnet_bytes=p, target_bytes=p,
                    replay_bytes_full=per_tr * self.dc.replay,
                    pseudo_count_entries=len(self.counts))

    def flops_per_decision(self) -> int:
        d = [N_FEAT * self.U, *self.dc.hidden, self.U]
        return int(sum(2 * a * b for a, b in zip(d[:-1], d[1:])))

    def state_dict(self):
        # replay packed into compact arrays (float16 states) to keep checkpoints small
        R = list(self.replay)
        packed = None
        if R:
            packed = dict(s=np.stack([r[0] for r in R]).astype(np.float16),
                          a=np.array([r[1] for r in R], np.int16),
                          r=np.array([r[2] for r in R], np.float32),
                          s2=np.stack([r[3] for r in R]).astype(np.float16),
                          m2=np.stack([r[4] for r in R]),
                          d=np.array([r[5] for r in R], bool))
        return dict(q=self.q.state_dict(), qt=self.qt.state_dict(), opt=self.opt.state_dict(),
                    replay=packed, counts=dict(self.counts), steps=self.steps,
                    pending=self.pending, rng=self.rng.bit_generator.state)

    def load_state_dict(self, d):
        self.q.load_state_dict(d["q"])
        self.qt.load_state_dict(d["qt"])
        self.opt.load_state_dict(d["opt"])
        self.replay = deque(maxlen=self.dc.replay)
        P = d["replay"]
        if P is not None:
            for i in range(len(P["a"])):
                self.replay.append((P["s"][i].astype(np.float32), int(P["a"][i]), float(P["r"][i]),
                                    P["s2"][i].astype(np.float32), P["m2"][i], bool(P["d"][i])))
        self.counts = defaultdict(int, d["counts"])
        self.steps = d["steps"]
        self.pending = d["pending"]
        self.rng.bit_generator.state = d["rng"]
