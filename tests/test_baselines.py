import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mtasfl.config import Config
from mtasfl.cost import cost_model as cm
from mtasfl.cost.profile import synthetic_profile
from mtasfl.baselines.cost import forward_only_cost
from mtasfl.baselines.trainer import BaselineJob, PerDeviceSplitTrainer
from mtasfl.learning.data import DeviceSampler, load_dataset, partition
from mtasfl.learning.models import build_model


def _setup(ds="mnist", ratio=1.0):
    torch.manual_seed(0)
    cfg = Config().override({"sfl": {"topk_ratio": ratio, "batch_per_device": 4, "lr": 0.1,
                                     "momentum": 0.0, "weight_decay": 0.0, "lr_warmup_rounds": 0}})
    data = load_dataset(ds, smoke=True)
    model = build_model(ds, data.num_classes, data.in_shape)
    shards = partition(data.y_train.numpy(), 6, "iid", 0.5, np.random.default_rng(0))
    return cfg, data, model, shards


@pytest.mark.parametrize("Lf", [1, 3])
def test_per_device_trainer_equals_sgd_when_aggregating_every_round(Lf):
    """I = 1, equal batches, no compression: device models averaged every round
    == centralised SGD on the union of the mini-batches."""
    cfg, data, model, shards = _setup()
    ref = copy.deepcopy(model)
    tr = PerDeviceSplitTrainer(model, data, DeviceSampler(shards, np.random.default_rng(1)), cfg, "cpu",
                               front_depth=Lf)
    samp = DeviceSampler(shards, np.random.default_rng(1))
    opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    devs = [0, 1, 2, 3]
    for _ in range(3):
        tr.train_round([BaselineJob(0, 9, devs[:2], {}, {d: 1 for d in devs[:2]}),
                        BaselineJob(1, 8, devs[2:], {}, {d: Lf for d in devs[2:]})])
        assert tr.aggregate_from(set(devs)) == 4
        idx = np.concatenate([samp.next(d, 4) for d in devs])
        opt.zero_grad()
        F.cross_entropy(ref(data.prep(data.x_train[idx], True)), data.y_train[idx]).backward()
        opt.step()
    for a, b in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


def test_periodic_aggregation_keeps_device_models_apart():
    cfg, data, model, shards = _setup(ratio=0.2)
    tr = PerDeviceSplitTrainer(model, data, DeviceSampler(shards, np.random.default_rng(1)), cfg, "cpu",
                               front_depth=2)
    for _ in range(3):
        st = tr.train_round([BaselineJob(0, 9, [0, 1, 2], {}, {0: 1, 1: 2, 2: 2})])
    w0 = list(tr.copies[0].parameters())[0]
    w1 = list(tr.copies[1].parameters())[0]
    assert not torch.allclose(w0, w1)                 # diverged between aggregations
    assert st.fwd_comp_err and set(st.per_device_update) == {0, 1, 2}
    tr.aggregate_from({0, 1, 2})
    assert not tr.copies
    tr.set_front_depth(4)                             # allowed right after aggregation
    ev = tr.evaluate()
    assert 0 <= ev["test_acc"] <= 1


def test_forward_only_equals_mta_cost_with_uniform_cut():
    """Dense payloads, p1 = p2 = p3: the baseline cost model is MTA-SFL's."""
    prof = synthetic_profile(L=6)
    for sync in (False, True):          # sync only affects post-training traffic
        cfg = Config().override({"sfl.topk_ratio": 1.0, "sfl.model_sync": sync})
        b = np.array([16.0, 16.0, 8.0])
        f = np.array([1e9, 1e9, 5e8])
        rul, rdl = np.array([2e6, 1e6, 3e6]), np.array([2e6, 2e6, 2e6])
        for c in (1, 3, 5):
            ref = cm.pipeline_cost(prof, (c, c, c), b, f, rul, rdl, 8e9, 2e7, 2e7, 6e9, 5e7, 5e7, 1e12, cfg)
            pc = forward_only_cost(prof, [c] * 3, b, f, rul, rdl, 2e7, 2e7, 5e7, 5e7, 1e12, cfg)
            assert math.isclose(pc.t_total, ref.t_total, rel_tol=1e-12)
            assert np.allclose(pc.ecomp_i, ref.ecomp_i)
            if not sync:
                assert math.isclose(pc.ecomm_j, ref.ecomm_j, rel_tol=1e-12)
                assert math.isclose(pc.ecomm_u, ref.ecomm_u, rel_tol=1e-12)
                assert np.allclose(pc.ecomm_i, ref.ecomm_i)


@pytest.mark.parametrize("method", ["adaptsfl", "sflsc3", "shsfl"])
def test_baseline_runs_and_resumes(method, tmp_path):
    from mtasfl.methods import get_method, simulation_class
    cfg = Config().override({"dataset": "mnist", "sfl.eval_every": 5})
    spec = get_method(method)
    out = str(tmp_path / method)
    sim = simulation_class(spec)(cfg, spec, out, smoke=True, resume=False, print_every=10 ** 9, ckpt_every=6)
    sim.run(6)
    sim2 = simulation_class(spec)(cfg, spec, out, smoke=True, resume=True, print_every=10 ** 9, ckpt_every=6)
    assert sim2.start_round == 6
    s = sim2.run(10)
    assert s["rounds_done"] == 10 and s["final_test_acc"] is not None
    import json
    R = [json.loads(l) for l in open(out + "/rounds.jsonl")]
    assert [r["round"] for r in R] == list(range(10))
    assert all("baseline" in r for r in R)
