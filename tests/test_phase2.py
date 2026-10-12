import copy

import numpy as np
import torch
import torch.nn.functional as F

from mtasfl.config import Config
from mtasfl.learning.data import load_dataset, partition, DeviceSampler
from mtasfl.learning.models import build_model
from mtasfl.learning.profiler import profile_model
from mtasfl.learning.compression import topk_sparsify, ErrorFeedbackBank
from mtasfl.learning.split_trainer import SplitTrainer, PipelineJob


def _setup(ds="cifar10", ratio=1.0, seed=0):
    torch.manual_seed(seed)
    cfg = Config().override({"sfl": {"topk_ratio": ratio, "batch_per_device": 4, "lr": 0.1,
                                     "momentum": 0.0, "weight_decay": 0.0,
                                     "lr_warmup_rounds": 0}})
    data = load_dataset(ds, smoke=True)
    model = build_model(ds, data.num_classes, data.in_shape)
    shards = partition(data.y_train.numpy(), 6, "iid", 0.5, np.random.default_rng(0))
    return cfg, data, model, shards


def test_split_equals_centralised_without_compression():
    """Eq. (4) + heterogeneous cuts == plain minibatch SGD when k = D."""
    for ds in ["cifar10", "mnist", "ucihar"]:
        cfg, data, model, shards = _setup(ds)
        ref = copy.deepcopy(model)
        L = model.L
        tr = SplitTrainer(model, data, DeviceSampler(shards, np.random.default_rng(1)), cfg, "cpu")
        opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
        samp_ref = DeviceSampler(shards, np.random.default_rng(1))
        jobs = [PipelineJob(0, 10, (1, 2, L - 1), [0, 1]),
                PipelineJob(1, 11, (2, 2, 2), [2, 3]),          # primary + relay bridge
                PipelineJob(2, 12, (1, L - 1, L), [4, 5])]       # cloud computes nothing
        for _ in range(2):
            tr.train_round(jobs)
            # reference: same batches, centralised
            idx = np.concatenate([samp_ref.next(i, 4) for jb in jobs for i in jb.devices])
            x = data.prep(data.x_train[idx], True)
            y = data.y_train[idx]
            opt_ref.zero_grad()
            F.cross_entropy(ref(x), y).backward()
            opt_ref.step()
        for a, b in zip(model.parameters(), ref.parameters()):
            assert torch.allclose(a, b, atol=1e-5), ds


def test_profiles_reasonable():
    for ds, L in [("cifar10", 10), ("mnist", 6), ("ucihar", 6)]:
        cfg, data, model, _ = _setup(ds)
        prof = profile_model(model, data.in_shape)
        assert prof.L == L
        assert np.all(prof.fwd[1:] > 0) and np.all(prof.act > 0)
    # ResNet-18 CIFAR ~ 0.56 GMACs = 1.1 GFLOPs forward
    _, data, model, _ = _setup("cifar10")
    tot = profile_model(model, data.in_shape).fwd.sum()
    assert 0.9e9 < tot < 1.4e9


def test_topk_keeps_largest():
    x = torch.tensor([0.1, -5.0, 3.0, 0.2, -0.05, 4.0])
    y = topk_sparsify(x, 0.5)
    assert torch.equal(y, torch.tensor([0.0, -5.0, 3.0, 0.0, 0.0, 4.0]))


def test_error_feedback_no_information_lost():
    """Sum of transmitted sparse gradients + final residual == sum of true grads."""
    bank = ErrorFeedbackBank()
    torch.manual_seed(0)
    gs = [torch.randn(8, 5) for _ in range(20)]
    sent = torch.zeros(8, 5)
    for g in gs:
        sent += bank.compress_concat("S", g, [0, 1], 4, 3, 0.1)
    resid = torch.cat([bank.store[("S", 0)][1], bank.store[("S", 1)][1]])
    assert bank.end_round() == 0
    assert torch.allclose(sent + resid, sum(gs), atol=1e-5)


def test_training_with_compression_reduces_loss():
    cfg, data, model, shards = _setup("mnist", ratio=0.2)
    tr = SplitTrainer(model, data, DeviceSampler(shards, np.random.default_rng(1)), cfg, "cpu")
    jobs = [PipelineJob(0, 6, (1, 3, 5), [0, 1, 2]), PipelineJob(1, 7, (2, 4, 4), [3, 4, 5])]
    first = np.mean([tr.train_round(jobs).loss for _ in range(3)])
    for _ in range(30):
        st = tr.train_round(jobs)
    last = np.mean([tr.train_round(jobs).loss for _ in range(3)])
    assert last < first
    assert set(st.fwd_comp_err) == {1, 2, 3}
    ev = tr.evaluate()
    assert 0 <= ev["test_acc"] <= 1 and len(ev["per_class_acc"]) == 10


def test_error_feedback_eviction():
    bank = ErrorFeedbackBank(ttl=3)
    bank.compress_concat(("R", 7, 0), torch.randn(8, 5), [0, 1], 4, 3, 0.1)
    for _ in range(3):
        assert ("R", 7, 0) in [k[0] for k in bank.store]
        bank.end_round()
    assert bank.end_round() == 2 and len(bank.store) == 0   # unused for ttl rounds
