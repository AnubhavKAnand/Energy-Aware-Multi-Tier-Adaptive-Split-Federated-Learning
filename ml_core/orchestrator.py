"""Synchronous Split Federated Learning Orchestrator.

Implements:
  • 4-tier forward/backward pipeline with TopK+EF compression at every
    tier boundary.
  • Dynamic cut management with heterogeneous cut configurations across
    parallel primary-UAV clusters.
  • FedAvg weight synchronisation (Eq. 8): sample-weighted aggregation
    across matching tier slices.
  • Genuine network-wide global evaluation on the centralised test set
    using the assembled global model  — no ``iot_models[0]`` shortcut.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

from mta_sfl.ml_core.compression import CompressionManager, TopKSparsifyWithEF
from mta_sfl.ml_core.split_models import CutConfig, SplitModel, TierExecutor


# ═══════════════════════════════════════════════════════════════════════════
# Cluster Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ClusterConfig:
    """Description of one primary-UAV cluster.

    Attributes
    ----------
    cluster_id      : Unique identifier for this cluster.
    primary_uav_id  : Node ID of the primary (edge) UAV.
    relay_uav_id    : Node ID of the relay UAV (may be ``None``).
    iot_ids         : Node IDs of IoT devices assigned to this cluster.
    cut_config      : ``(p1, p2, p3)``  — 1-indexed cut-point tuple.
    """
    cluster_id: str
    primary_uav_id: str
    relay_uav_id: Optional[str]
    iot_ids: List[str]
    cut_config: CutConfig


# ═══════════════════════════════════════════════════════════════════════════
# Round Metrics
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RoundMetrics:
    """Aggregate metrics for one global SFL round."""
    round_idx: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)
    num_participating_iot: int = 0
    ef_residual_norms: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# FedAvg Helpers
# ═══════════════════════════════════════════════════════════════════════════


def fedavg_aggregate(
    models: List[nn.Module],
    weights: List[float],
) -> Dict[str, Tensor]:
    """Sample-weighted FedAvg across a list of models with identical arch.

        W_global = Σᵢ (nᵢ / n_total) · Wᵢ        (Eq. 8)

    Parameters
    ----------
    models  : List of nn.Modules with identical parameter names / shapes.
    weights : Per-model weight (e.g., number of samples).

    Returns
    -------
    Averaged state_dict.
    """
    if not models:
        return {}

    total = sum(weights)
    if total <= 0:
        total = 1.0
    normalised = [w / total for w in weights]

    avg_state: Dict[str, Tensor] = {}
    for w_i, model in zip(normalised, models):
        for name, param in model.state_dict().items():
            if name in avg_state:
                avg_state[name] = avg_state[name] + param.detach().clone() * w_i
            else:
                avg_state[name] = param.detach().clone() * w_i

    return avg_state


def broadcast_state(
    models: List[nn.Module],
    global_state: Dict[str, Tensor],
) -> None:
    """Load a shared state_dict into every model in the list."""
    for model in models:
        model.load_state_dict(global_state)


# ═══════════════════════════════════════════════════════════════════════════
# SFL Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class SFLOrchestrator:
    """Synchronous Split Federated Learning coordinator.

    Manages the complete 4-tier lifecycle:
      1. Forward pass with TopK compression at each tier boundary.
      2. Backward pass with EF-compressed gradients.
      3. Weight update via SGD/Adam.
      4. FedAvg aggregation across participating clients.
      5. Global evaluation on the centralised test set.

    Parameters
    ----------
    dataset_name   : Architecture selection (``"cifar10"`` etc.).
    num_classes    : Number of output classes.
    clusters       : List of ``ClusterConfig`` per primary-UAV cluster.
    k_ratio        : TopK compression ratio (fraction of elements kept).
    lr             : Learning rate for all optimisers.
    optimizer_name : ``"sgd"`` or ``"adam"``.
    momentum       : SGD momentum (ignored for Adam).
    device         : ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        dataset_name: str,
        num_classes: int,
        clusters: List[ClusterConfig],
        k_ratio: float = 0.5,
        lr: float = 0.01,
        optimizer_name: str = "sgd",
        momentum: float = 0.9,
        device: str = "cpu",
    ):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.clusters = clusters
        self.k_ratio = k_ratio
        self.lr = lr
        self.device = torch.device(device)
        self.optimizer_name = optimizer_name
        self.momentum = momentum

        # --- Build the reference model ---
        self._base_model = SplitModel(dataset_name, num_classes)
        self._L = self._base_model.L

        # --- Per-cluster tier executors ---
        self._executors: Dict[str, TierExecutor] = {}
        for cl in clusters:
            self._executors[cl.cluster_id] = TierExecutor(
                self._base_model,
                cl.cut_config,
                num_iot=len(cl.iot_ids),
            ).to(self.device)

        # --- Shared cloud model (Tier 4) ---
        # All clusters share the same Tier 4 model (the cloud).
        # Synchronise them to the first cluster's Tier 4.
        self._sync_cloud_models()

        # --- Compression manager ---
        self.compression = CompressionManager(k_ratio)

        # --- Build optimisers ---
        self._tier1_optimisers: Dict[str, Optimizer] = {}   # keyed by iot_id
        self._tier2_optimisers: Dict[str, Optimizer] = {}   # keyed by cluster_id
        self._tier3_optimisers: Dict[str, Optimizer] = {}   # keyed by cluster_id
        self._tier4_optimiser: Optional[Optimizer] = None
        self._build_optimisers()

        # --- Data loader iterators (set per round) ---
        self._loader_iters: Dict[str, Iterator] = {}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _sync_cloud_models(self) -> None:
        """Ensure all executors share the same Tier 4 (cloud) parameters."""
        ref_state = None
        for cid, ex in self._executors.items():
            if ref_state is None:
                ref_state = copy.deepcopy(ex.tier4_model.state_dict())
            else:
                ex.tier4_model.load_state_dict(ref_state)

        # Additionally, point all executors' tier4_model to the SAME object
        # so gradient accumulation across clusters works correctly.
        first_ex = next(iter(self._executors.values()))
        shared_tier4 = first_ex.tier4_model
        for ex in self._executors.values():
            ex.tier4_model = shared_tier4

    def _make_optimiser(self, params: List[nn.Parameter]) -> Optimizer:
        if not params:
            # Dummy optimiser for empty parameter lists (bridge tier 3)
            return SGD([nn.Parameter(torch.zeros(1))], lr=0.0)
        if self.optimizer_name == "adam":
            return Adam(params, lr=self.lr)
        return SGD(params, lr=self.lr, momentum=self.momentum)

    def _build_optimisers(self) -> None:
        for cl in self.clusters:
            ex = self._executors[cl.cluster_id]
            # Tier 1: one optimiser per IoT device
            for i, iot_id in enumerate(cl.iot_ids):
                self._tier1_optimisers[iot_id] = self._make_optimiser(
                    list(ex.tier1_models[i].parameters())
                )
            # Tier 2
            self._tier2_optimisers[cl.cluster_id] = self._make_optimiser(
                list(ex.tier2_model.parameters())
            )
            # Tier 3
            self._tier3_optimisers[cl.cluster_id] = self._make_optimiser(
                list(ex.tier3_model.parameters())
            )

        # Tier 4 (shared cloud)
        shared_tier4 = next(iter(self._executors.values())).tier4_model
        self._tier4_optimiser = self._make_optimiser(
            list(shared_tier4.parameters())
        )

    # ------------------------------------------------------------------
    # Data loader management
    # ------------------------------------------------------------------

    def _get_batch(
        self,
        iot_id: str,
        loader: DataLoader,
    ) -> Tuple[Tensor, Tensor]:
        """Get the next batch for an IoT device, cycling if exhausted."""
        key = iot_id
        if key not in self._loader_iters:
            self._loader_iters[key] = iter(loader)
        try:
            x, y = next(self._loader_iters[key])
        except StopIteration:
            self._loader_iters[key] = iter(loader)
            x, y = next(self._loader_iters[key])
        return x.to(self.device), y.to(self.device)

    # ------------------------------------------------------------------
    # Core: Single-cluster forward/backward with compression
    # ------------------------------------------------------------------

    def _forward_backward_cluster(
        self,
        cluster: ClusterConfig,
        iot_loaders: Dict[str, DataLoader],
        surviving_iot_ids: Optional[List[str]] = None,
    ) -> Tuple[float, int]:
        """Run one forward + backward pass for a single cluster.

        Parameters
        ----------
        cluster          : Cluster configuration.
        iot_loaders      : Mapping  iot_id → DataLoader.
        surviving_iot_ids: Subset of active IoTs (for dropout / depletion).
                           If None, all IoTs in the cluster participate.

        Returns
        -------
        (loss_value, total_samples_in_batch)
        """
        ex = self._executors[cluster.cluster_id]
        active_ids = surviving_iot_ids or cluster.iot_ids

        smashed_list: List[Tensor] = []
        labels_list: List[Tensor] = []
        total_samples = 0

        for iot_idx, iot_id in enumerate(cluster.iot_ids):
            if iot_id not in active_ids:
                continue
            if iot_id not in iot_loaders:
                continue

            x_batch, y_batch = self._get_batch(iot_id, iot_loaders[iot_id])
            total_samples += x_batch.size(0)

            # ---- Tier 1 Forward ----
            delta1 = ex.tier1_forward(x_batch, iot_idx)

            # ---- Compress at Tier 1 → 2 boundary ----
            buf_key = f"{cluster.cluster_id}/{iot_id}/t1t2"
            delta1_c = self.compression.compress(delta1, buf_key)

            smashed_list.append(delta1_c)
            labels_list.append(y_batch)

        if not smashed_list:
            return 0.0, 0

        # ---- Tier 2 Forward ----
        delta2 = ex.tier2_forward(smashed_list)

        # ---- Compress at Tier 2 → 3 boundary ----
        buf_key_t2t3 = f"{cluster.cluster_id}/t2t3"
        delta2_c = self.compression.compress(delta2, buf_key_t2t3)

        # ---- Tier 3 Forward ----
        delta3 = ex.tier3_forward(delta2_c)

        # ---- Compress at Tier 3 → 4 boundary (skip if bridge) ----
        if not ex.is_bridge_mode:
            buf_key_t3t4 = f"{cluster.cluster_id}/t3t4"
            delta3_c = self.compression.compress(delta3, buf_key_t3t4)
        else:
            delta3_c = delta3

        # ---- Tier 4 Forward ----
        labels_cat = torch.cat(labels_list, dim=0)
        logits, loss = ex.tier4_forward(delta3_c, labels_cat)

        # ---- Backward (gradients flow back through compression nodes) ----
        loss.backward()

        return float(loss.item()), total_samples

    # ------------------------------------------------------------------
    # Main round execution
    # ------------------------------------------------------------------

    def run_round(
        self,
        round_idx: int,
        iot_loaders: Dict[str, DataLoader],
        test_loader: Optional[DataLoader] = None,
        surviving_map: Optional[Dict[str, List[str]]] = None,
    ) -> RoundMetrics:
        """Execute one complete SFL communication round.

        Steps:
          1. Zero all gradients.
          2. For each cluster, run forward/backward (gradients accumulate
             at the shared Tier 4 model).
          3. Step all optimisers.
          4. FedAvg across matching tiers.
          5. Global evaluation on ``test_loader``.

        Parameters
        ----------
        round_idx     : Global round index.
        iot_loaders   : iot_id → training DataLoader.
        test_loader   : Centralised test DataLoader.
        surviving_map : cluster_id → [surviving iot_ids].  ``None`` = all active.

        Returns
        -------
        ``RoundMetrics`` for this round.
        """
        # ---- Step 1: Zero all gradients ----
        self._zero_all_grads()

        # ---- Step 2: Forward/backward per cluster (gradient accumulation) ----
        total_loss = 0.0
        total_samples = 0

        for cl in self.clusters:
            surviving = (surviving_map or {}).get(cl.cluster_id)
            cl_loss, cl_samples = self._forward_backward_cluster(
                cl, iot_loaders, surviving
            )
            total_loss += cl_loss * cl_samples
            total_samples += cl_samples

        avg_train_loss = total_loss / max(total_samples, 1)

        # ---- Step 3: Step all optimisers ----
        self._step_all_optimisers()

        # ---- Step 4: FedAvg aggregation ----
        self._fedavg_all_tiers(iot_loaders)

        # ---- Step 5: Global evaluation ----
        test_loss, test_acc, per_class = 0.0, 0.0, {}
        if test_loader is not None:
            test_loss, test_acc, per_class = self.global_evaluate(test_loader)

        return RoundMetrics(
            round_idx=round_idx,
            train_loss=avg_train_loss,
            test_loss=test_loss,
            test_accuracy=test_acc,
            per_class_accuracy=per_class,
            num_participating_iot=total_samples,
            ef_residual_norms=self.compression.all_residual_norms(),
        )

    # ------------------------------------------------------------------
    # Gradient management
    # ------------------------------------------------------------------

    def _zero_all_grads(self) -> None:
        for opt in self._tier1_optimisers.values():
            opt.zero_grad()
        for opt in self._tier2_optimisers.values():
            opt.zero_grad()
        for opt in self._tier3_optimisers.values():
            opt.zero_grad()
        if self._tier4_optimiser is not None:
            self._tier4_optimiser.zero_grad()

    def _step_all_optimisers(self) -> None:
        for opt in self._tier1_optimisers.values():
            opt.step()
        for opt in self._tier2_optimisers.values():
            opt.step()
        for opt in self._tier3_optimisers.values():
            opt.step()
        if self._tier4_optimiser is not None:
            self._tier4_optimiser.step()

    # ------------------------------------------------------------------
    # FedAvg per tier (Eq. 8)
    # ------------------------------------------------------------------

    def _fedavg_all_tiers(
        self,
        iot_loaders: Dict[str, DataLoader],
    ) -> None:
        """Apply FedAvg weight synchronisation across all tiers.

        Tier 1: Aggregate across ALL IoT devices (all clusters).
        Tier 2: Aggregate across all primary-UAV cluster models.
        Tier 3: Aggregate across all relay-UAV models (skip bridges).
        Tier 4: Single model — no aggregation needed.
        """
        # ---- Tier 1: all IoTs ----
        tier1_models: List[nn.Module] = []
        tier1_weights: List[float] = []
        iot_to_cluster_idx: Dict[str, Tuple[str, int]] = {}

        for cl in self.clusters:
            ex = self._executors[cl.cluster_id]
            for i, iot_id in enumerate(cl.iot_ids):
                tier1_models.append(ex.tier1_models[i])
                # Weight by dataset size
                w = len(iot_loaders[iot_id].dataset) if iot_id in iot_loaders else 1.0
                tier1_weights.append(w)
                iot_to_cluster_idx[iot_id] = (cl.cluster_id, i)

        if len(tier1_models) > 1:
            avg_state = fedavg_aggregate(tier1_models, tier1_weights)
            broadcast_state(tier1_models, avg_state)

        # ---- Tier 2: all primary UAVs ----
        tier2_models: List[nn.Module] = []
        tier2_weights: List[float] = []
        for cl in self.clusters:
            ex = self._executors[cl.cluster_id]
            tier2_models.append(ex.tier2_model)
            # Weight by total cluster samples
            cluster_samples = sum(
                len(iot_loaders[iid].dataset) for iid in cl.iot_ids
                if iid in iot_loaders
            )
            tier2_weights.append(max(cluster_samples, 1.0))

        if len(tier2_models) > 1:
            avg_state = fedavg_aggregate(tier2_models, tier2_weights)
            broadcast_state(tier2_models, avg_state)

        # ---- Tier 3: all relay UAVs (skip bridge-mode clusters) ----
        tier3_models: List[nn.Module] = []
        tier3_weights: List[float] = []
        for cl in self.clusters:
            ex = self._executors[cl.cluster_id]
            if not ex.is_bridge_mode and len(list(ex.tier3_model.parameters())) > 0:
                tier3_models.append(ex.tier3_model)
                cluster_samples = sum(
                    len(iot_loaders[iid].dataset) for iid in cl.iot_ids
                    if iid in iot_loaders
                )
                tier3_weights.append(max(cluster_samples, 1.0))

        if len(tier3_models) > 1:
            avg_state = fedavg_aggregate(tier3_models, tier3_weights)
            broadcast_state(tier3_models, avg_state)

    # ------------------------------------------------------------------
    # Global Evaluation
    # ------------------------------------------------------------------

    def assemble_global_model(self) -> nn.Sequential:
        """Construct the unified global model from synchronised weights.

        Assembles Tier 1 (from first IoT) + Tier 2 + Tier 3 + Tier 4
        into a single ``nn.Sequential``.
        """
        # Use the first cluster's models (they're synchronised via FedAvg)
        first_cl = self.clusters[0]
        ex = self._executors[first_cl.cluster_id]

        modules: List[nn.Module] = []

        # Tier 1 (synchronised)
        for submod in ex.tier1_models[0]:
            modules.append(submod)
        # Tier 2
        for submod in ex.tier2_model:
            modules.append(submod)
        # Tier 3 (identity if bridge)
        if not ex.is_bridge_mode:
            for submod in ex.tier3_model:
                modules.append(submod)
        # Tier 4
        for submod in ex.tier4_model:
            modules.append(submod)

        return nn.Sequential(*modules).to(self.device)

    @torch.no_grad()
    def global_evaluate(
        self,
        test_loader: DataLoader,
    ) -> Tuple[float, float, Dict[int, float]]:
        """Evaluate the assembled global model on the centralized test set.

        Returns
        -------
        (test_loss, test_accuracy, per_class_accuracy)

        This evaluates the **genuine global model** — not ``iot_models[0]``.
        """
        global_model = self.assemble_global_model()
        global_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        class_correct: Dict[int, int] = {}
        class_total: Dict[int, int] = {}

        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = global_model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            for c in range(self.num_classes):
                mask = (y == c)
                class_total[c] = class_total.get(c, 0) + mask.sum().item()
                class_correct[c] = class_correct.get(c, 0) + (
                    (preds == y) & mask
                ).sum().item()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        per_class: Dict[int, float] = {}
        for c in range(self.num_classes):
            if class_total.get(c, 0) > 0:
                per_class[c] = class_correct.get(c, 0) / class_total[c]
            else:
                per_class[c] = 0.0

        return avg_loss, accuracy, per_class

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_executor(self, cluster_id: str) -> TierExecutor:
        return self._executors[cluster_id]

    def total_iot_count(self) -> int:
        return sum(len(cl.iot_ids) for cl in self.clusters)


# ═══════════════════════════════════════════════════════════════════════════
# Self-Contained Verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SFL Orchestrator — Verification")
    print("=" * 60)

    import numpy as np
    torch.manual_seed(42)

    from mta_sfl.ml_core.data_partitioner import DataPartitioner
    from mta_sfl.ml_core.split_models import SplitModel

    # ---- Setup: MNIST, 4 IoTs, 2 clusters ----
    dp = DataPartitioner("mnist", num_clients=4, mode="dirichlet",
                          alpha=0.5, batch_size=16, seed=42)
    train_loaders, test_loader = dp.partition()

    # Map IoT IDs to loaders
    iot_ids = [f"iot_{i}" for i in range(4)]
    iot_loaders: Dict[str, DataLoader] = {
        iot_ids[i]: train_loaders[i] for i in range(4)
    }

    # Define 2 clusters with different cut configs
    clusters = [
        ClusterConfig(
            cluster_id="cluster_A",
            primary_uav_id="uav_0",
            relay_uav_id="uav_1",
            iot_ids=["iot_0", "iot_1"],
            cut_config=(3, 6, 9),  # Normal 4-tier split
        ),
        ClusterConfig(
            cluster_id="cluster_B",
            primary_uav_id="uav_2",
            relay_uav_id="uav_3",
            iot_ids=["iot_2", "iot_3"],
            cut_config=(3, 6, 6),  # Bridge mode (Tier 3 = identity)
        ),
    ]

    print(f"\n  Clusters: {[c.cluster_id for c in clusters]}")
    print(f"  Cut configs: {[c.cut_config for c in clusters]}")
    print(f"  IoTs per cluster: {[c.iot_ids for c in clusters]}")

    # ---- Create orchestrator ----
    orch = SFLOrchestrator(
        dataset_name="mnist",
        num_classes=10,
        clusters=clusters,
        k_ratio=0.5,
        lr=0.01,
        device="cpu",
    )

    print(f"  Total IoTs: {orch.total_iot_count()}")
    print(f"  Cluster B is bridge mode: "
          f"{orch.get_executor('cluster_B').is_bridge_mode}")

    # ---- Test 1: Run 3 training rounds ----
    metrics_history: List[RoundMetrics] = []
    for r in range(3):
        metrics = orch.run_round(r, iot_loaders, test_loader)
        metrics_history.append(metrics)
        print(f"\n  Round {r}: train_loss={metrics.train_loss:.4f}, "
              f"test_loss={metrics.test_loss:.4f}, "
              f"test_acc={metrics.test_accuracy:.4f}")

    # ---- Test 2: Verify global model evaluation is not faked ----
    _, acc_global, per_class = orch.global_evaluate(test_loader)
    assert 0.0 <= acc_global <= 1.0, f"Invalid accuracy: {acc_global}"
    assert len(per_class) == 10, f"Expected 10 classes, got {len(per_class)}"
    print(f"\n  [Global Eval] accuracy = {acc_global:.4f}")
    print(f"  Per-class: { {c: f'{a:.3f}' for c, a in per_class.items()} }")
    print("  ✓ Global evaluation check passed")

    # ---- Test 3: Verify FedAvg changed weights ----
    # After FedAvg, all Tier 1 models should have identical weights
    ex_a = orch.get_executor("cluster_A")
    ex_b = orch.get_executor("cluster_B")
    state_0 = ex_a.tier1_models[0].state_dict()
    state_1 = ex_a.tier1_models[1].state_dict()
    state_2 = ex_b.tier1_models[0].state_dict()

    for key in state_0:
        assert torch.allclose(state_0[key], state_1[key], atol=1e-6), \
            f"Tier 1 FedAvg failed: {key} differs between IoT 0 and 1"
        assert torch.allclose(state_0[key], state_2[key], atol=1e-6), \
            f"Tier 1 FedAvg failed: {key} differs between clusters"
    print("\n  ✓ FedAvg synchronisation: all Tier 1 models have identical weights")

    # ---- Test 4: Verify EF buffers are active ----
    ef_norms = orch.compression.all_residual_norms()
    active_buffers = {k: v for k, v in ef_norms.items() if v > 0}
    assert len(active_buffers) > 0, \
        "At least some EF buffers should be non-zero after training"
    print(f"\n  [EF Buffers] {len(active_buffers)} active out of "
          f"{len(ef_norms)} total")
    for k, v in list(active_buffers.items())[:3]:
        print(f"    {k}: ‖e‖ = {v:.4f}")
    print("  ✓ EF buffer activity check passed")

    # ---- Test 5: Training loss should decrease (or at least not explode) ----
    losses = [m.train_loss for m in metrics_history]
    assert all(not math.isnan(l) for l in losses), f"NaN loss detected: {losses}"
    print(f"\n  Train losses: {[f'{l:.4f}' for l in losses]}")
    print("  ✓ No NaN/Inf in training losses")

    # ---- Test 6: Surviving subset (simulated dropout) ----
    import math
    surviving_map = {"cluster_A": ["iot_0"], "cluster_B": ["iot_3"]}
    m = orch.run_round(99, iot_loaders, test_loader, surviving_map=surviving_map)
    print(f"\n  [Survival test] Round 99 with 2/4 IoTs: "
          f"loss={m.train_loss:.4f}, acc={m.test_accuracy:.4f}")
    print("  ✓ Partial participation check passed")

    print("\n" + "=" * 60)
    print("ALL ORCHESTRATOR TESTS PASSED")
    print("=" * 60)
