# MTA-SFL — implementation

Energy-Aware Multi-Tier Adaptive Split Federated Learning for UAV-IoT networks
(TGCN revision).  Four tiers — IoT devices → primary UAVs → relay UAVs → cloud —
with a DDQN relay-selection agent per primary UAV (pseudo-count exploration)
and a LinUCB split-selection agent per route, top-k sparsification
(no EF forward / EF backward) and the zero-computation topological bridge.

## Install

```bash
pip install -r requirements.txt
```

Datasets download automatically into `./data` on first use
(MNIST, Fashion-MNIST, CIFAR-10/100 via torchvision; UCI HAR from the UCI repository).

## Run

```bash
python main.py --pilot           # FIRST: ~200 rounds of every method on MNIST + CIFAR-10,
                                 # prints accuracy, time per round and an estimate for the full run
python main.py                   # EVERYTHING: 4 methods x 5 datasets x {IID, non-IID} x seeds {0,1,2}
                                 # = 120 runs of 4000 rounds
python main.py --workers 2       # two runs at once on the same GPU
python summarize.py              # mean ± 95 % CI table of all finished runs -> results/summary_table.csv
```

Subsets: `--methods mtasfl adaptsfl` (or `mtasfl,adaptsfl`), `--datasets cifar10`,
`--partitions dirichlet`, `--seeds 0`, `--rounds 4000`; any parameter with
`--set key=value` (e.g. `--set sfl.topk_ratio=0.05`); `--smoke` = synthetic data, no download.

* Order is **seed-major**: all methods/datasets/partitions for seed 0, then seed 1, then 2.
* GPU is used automatically (`--device cuda:1` to choose).
* Re-running the same command **skips finished runs and resumes** unfinished ones from the
  last checkpoint (every 500 rounds). `--no-resume` starts fresh.
* A failed run is recorded in `results/progress.json` (with its traceback) and the queue continues.
* Terminal: one line every 100 rounds
  `round | test acc | train loss | round latency | energy/round | drop % | violation % | bridge % | regret | UAV depletions | sec/round`.

## Output

```
results/<iid|dirichlet>/<dataset>/<method>/seed<k>/
  rounds.jsonl   one JSON line per round, written + fsync'd the moment the round ends
  eval.jsonl     test metrics every 20 rounds (sfl.eval_every)
  summary.json   running summary (updated every 100 rounds, final at the end)
  config.json    full configuration of the run
  meta.json      model block profile (FLOPs/activations/params), agent memory & FLOPs,
                 simulation-parameter table (markdown), node positions
  checkpoint.pt  resumable state
results/progress.json   live status of all 120 jobs (done / running / failed), ETA
results/index.json      final summaries of all finished runs
results/summary_table.csv   written by summarize.py: mean ± 95 % CI over seeds, ~30 metrics
```

### What is recorded every round (`rounds.jsonl`)

| Reviewer item | Fields |
|---|---|
| Latency (Eq. 9) | `t_round`, `T_total_mean/max`, per pipeline `T_total`, `T_iot_max`, `Tcomp_*`, `Tcomm_*`, `est_T` |
| Energy (Eq. 10–16), full cost boundary (R2.5, R3.7) | `energy_swarm_uav_J`, `energy_iot_J`, `energy_prop_J`, `energy_hover_cir_J`, `energy_comp_J`, `energy_comm_J`, `energy_excl_prop_J`, per pipeline `E_j`, `E_u`, `E_iot`, `E_parts` |
| Battery / lifetime / depletion (R1.6, R2.7) | `uav_alive`, `uav_depleted_now`, `uav_returned_now`, `uav_Eres_min/mean`, `relay_Eres_min`, `iot_Eres_*`, per pipeline `E_res_relay`, `bridge` |
| Constraint compliance (R3.8) | `latency_violations`, `energy_violations`, `n_deadline_miss`, `n_aborted`, per pipeline `viol_t`, `viol_e`, `status` |
| Stragglers / dropout per tier (R1.1) | `n_dropped`, `drop_rate`, per pipeline `drop_outage`, `drop_deadline`, `n_survive`, `link_failure`/`no_survivors` status |
| Association fairness (R3.C1) | `n_assoc`, `n_unassoc`, `assoc_load_jain`, `participation_jain` |
| Routing resilience, churn, repair (R1.6, R4.2) | `churn`, `forced_reselect`, `n_link_fail`, `mean_degree`, `cc_mean`, per pipeline `n_adj`, `n_cloud`, `n_feasible_relays` |
| Control overhead / onboard cost (R1.8, R3.10, R4.3) | `beacon_bytes`, `timing.{ddqn_decide_ms, linucb_decide_ms, agent_update_ms, train_ms}`; memory & FLOPs in `meta.json` |
| Learning quality (R1.5, R2.10, R3.15) | `train_loss`, `train_acc`, `grad_norm`, `test_acc`/`test_loss` (eval rounds) |
| Sparsification effects (R1.5, R3.C5) | `fwd_comp_err` and `bwd_comp_err` per tier (relative error), `ef_resets`, per pipeline `bits` |
| Communication volume (R1.5) | `bits_tx`, `bits_dense`, `compression_ratio`, `cum_bits_tx`, per pipeline `bits_tx` |
| Interference / PHY (R4.1, R3.C6) | `sinr_ul_db_mean/p10`, `sinr_dl_db_mean/p10`, per pipeline `rates` (UL/DL/aerial/backhaul, realised) |
| Energy outside the objective boundary (R2.5, R3.C7, R1.8) | `energy_cloud_comp_J`, `energy_agents_J`, `agent_flops` |
| CC vs link lifetime evidence (R3.C3, R1.9) | per pipeline `cc`, `relay_life_s`, `relay_qload`, `relay_snr_db`, `status` (link_failure) |
| Agent behaviour / theory check (R1.2, R3.11-13) | `inst_regret_sum`, `cum_regret` vs expected-cost oracle, per pipeline `r_ext`, `r_int`, `r_lin`, `arm`, `x_bar` (LinUCB context, for the linearity test), `regret_route`, `regret_split`, `ucb_width`, `alpha`, `ddqn_loss`, `ddqn_q`, `cost`, `oracle_cost` |

`eval.jsonl` adds per-class accuracy and device-level fairness: accuracy on each
device's own label distribution → mean, std, worst-10 %, Jain index.
`summary.json` adds rounds-to-target accuracy, completion / drop / violation
rates, bridge usage, depletion count and first-depletion round, churn per round,
average regret, total energy (with and without propulsion, plus cloud and agent energy),
time / energy / bits to target accuracy, and total communication volume.

## Code map

| File | Paper |
|---|---|
| `mtasfl/config.py` | all parameters (exported as the parameter table) |
| `mtasfl/env/network.py` | Sec. III: mobility, association Eq. (1), SINR Eq. (2), rates Eq. (3), CC Eq. (5), link lifetime, U_adj / U_cloud |
| `mtasfl/env/channel.py` | A2G / A2A / backhaul channel models |
| `mtasfl/cost/cost_model.py` | Sec. IV latency Eq. (8)-(9), energy Eq. (10)-(16); Eq. (17), (22), (28) |
| `mtasfl/cost/vectorized.py` | same model over all split tuples: P_feasible (Eq. 31), U_safe, regret oracle |
| `mtasfl/learning/split_trainer.py` | multi-tier forward/backward, Eq. (4) layer-wise aggregation |
| `mtasfl/learning/compression.py` | top-k Eq. (6)-(7), error feedback for gradients |
| `mtasfl/learning/models.py` | GroupNorm ResNet-18 (CIFAR), CNN (MNIST/FMNIST), 1-D CNN (HAR) |
| `mtasfl/agents/ddqn.py` | Sec. VI-A, Algorithm 2, Eq. (21)-(26) |
| `mtasfl/agents/linucb.py` | Sec. VI-B, Algorithm 3, Eq. (27)-(34), Lemma 2 alpha |
| `mtasfl/sim/runner.py` | Algorithm 1 (joint execution), metrics, checkpoints |
| `mtasfl/methods.py` | method registry (baselines / ablations plug in here) |

## Methods

| `--methods` | Paper | What is implemented |
|---|---|---|
| `mtasfl` | this paper | DDQN relay selection with pseudo-count exploration, route-specific LinUCB split selection, multi-tier split training, top-k (no EF forward, EF backward), topological bridge, Eq. (4) aggregation every round |
| `adaptsfl` | Lin et al., IEEE/ACM ToN 33(6), 2025 | per-device cut layers, server-side common sub-model every round, forged client-specific models aggregated every I rounds; Alg. 3 BCD over (I, μ) minimising Θ(I, μ) = R(I, μ)/I · T(I, μ) (Cor. 1, Eq. 26–28) |
| `sflsc3` | Hou et al., arXiv:2504.01443v2, 2026 (ICC 2025) | common cut L_c, client models aggregated every I rounds; BCD over (I, L_c, b, UAV position) minimising Ξ (Eq. 35–38, 40–46) |
| `shsfl` | Gao et al., Drones 10(1):37, 2026 | MAB-GT device selection (self-selection utility, K-means coalitions, Shapley contribution, UCB score with trust, Alg. 1), hybrid split/federated modes, cut after conv1, verifiable mechanism (challenge proofs + signatures charged as overhead), UE-side aggregation every round |

### One environment for every method

All methods run in the same simulator with the same seed-determined topology,
mobility, channel realisations, association, data partition, model, learning
rate, T_max deadline, straggler rule, energy model and logged metrics.
Two-tier baselines are mapped to the 4-tier network as follows:

* their server = the cloud; primary and relay UAVs only forward traffic;
* relay = the cloud-connected neighbour with the largest bottleneck rate min(R_ju, R_uS);
* every method uses the same top-k compression of activations / gradients
  (without it, dense CIFAR activations of 4–34 Mbit per device per round cannot meet
  the deadline on IoT uplinks and no dense baseline can train; the MTA-SFL
  no-top-k ablation isolates compression separately);
* every device takes one SGD step per round (same local computation budget);
* the baseline optimisers see the same straggler rule as MTA-SFL's masks: devices that
  cannot meet the local deadline are not counted in their latency max, and no decision
  may drop a device that could participate;
* model synchronisation traffic is charged to every method (see below).

### Model synchronisation (`sfl.model_sync`, default on)

Eq. (4) aggregates layer weights at the cloud every round and sends each node the
layers it runs next (Algorithm 1). This upload/download traffic is charged for
MTA-SFL every round, and for the baselines at their own aggregation schedule (every I
rounds; model download when a device joins). It runs after training, so it is not part
of the T_max deadline (Eq. 17b) but adds to the round duration, to T~ in the objective
and to the energy. `--set sfl.model_sync=false` reproduces the paper's Eq. (9) exactly.

### Offline agent pre-training (`agent_pretrain_rounds`, default 0 = off)

`--set agent_pretrain_rounds=6000` pre-trains MTA-SFL's DDQN and LinUCB agents in fast
mode on an independent topology (seed + 1000) before the run, then keeps learning
online. The baselines' optimisers use the analytic latency model from round 0, whereas
MTA-SFL learns from scratch; pre-training removes that cold-start gap. If you use it,
state it in the paper.

## Implementation choices where the paper is silent

Network
* Relay candidates are non-primary UAVs; a relay may serve several primaries,
  sharing its aerial bandwidth equally (this is Q_load).
* Aerial range ≈ 375 m (log-distance exponent 2.6, γ_th = 10 dB); backhaul rate
  falls with cloud distance (exponent 2.7).
* Primary UAVs hover over their IoT cluster (drift ≤ 80 m); relays follow
  Gauss-Markov mobility at ~10 m/s; mobility advances by T_round each round.
* A2G links use the mean LoS/NLoS excess loss (LoS state is position-dependent,
  not redrawn every round) with Rician small-scale fading (K ≥ 1).
* 20 MHz OFDMA channel per primary-UAV cluster for UL and DL; inter-cluster
  interference from clusters on the same frequency block (2 blocks by default).

Hardware
* UAV compute 100–200 GFLOP/s, IoT 1–2 GFLOP/s, cloud 1 TFLOP/s; κ calibrated to
  ~1.5e-10 J/FLOP (UAV) and ~2e-9 J/FLOP (IoT).
* Battery 300 kJ per UAV (~83 Wh). A depleted UAV goes offline for 300 s of
  simulated time (battery swap) and returns full; this is what makes 4000-round
  runs possible and produces the depletion / churn metrics.
* Every alive UAV pays hover + circuit energy each round in the battery update;
  the per-pipeline objective follows Eq. (10)-(15) exactly.

Deadlines and masks
* T_max per dataset: 12 s (CIFAR-10/100), 2 s (MNIST, Fashion-MNIST), 1.5 s (UCI HAR).
* Tier-1 straggler rule: a device is dropped if compute + upload + gradient
  return exceeds 50 % of T_max (the return path matters: a device with a bad
  downlink would otherwise stall the whole pipeline).
* Masks (U_safe, P_feasible, Eq. 31) use the expected model with a 15 % safety
  margin, and a split may not drop more devices than the best split must
  (otherwise latency could be "reduced" by discarding data).
* The cloud always computes at least the final block (p3 ≤ L−1), as Tier 4
  "computes the remaining layers to L and the loss".

Learning agents
* LinUCB exploration width: α_{t,δ} from Lemma 2 multiplied by
  `linucb.alpha_scale = 0.1`. With scale 1.0 the "+1" term of Lemma 2 keeps the
  confidence bonus as large as the reward gaps and the split regret does not
  decrease; 0.1 cut the split regret ~3x in a 2400-round sweep.
* Regret is measured against an expected-cost oracle (best relay × split under
  the mean model) and logged split into routing and split parts (Eq. 50).
* Optimiser is plain SGD (Eq. 4); momentum / weight decay are config options.
