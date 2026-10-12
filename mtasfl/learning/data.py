"""
Datasets for the five benchmarks of the paper and their partition over the
IoT devices (IID / Dirichlet non-IID).

* MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100: torchvision (auto-download)
* UCI HAR: raw inertial signals (9 channels x 128 samples), downloaded and
  parsed from the UCI repository (auto-download)
* synthetic: tiny learnable dataset with identical tensor shapes, used only
  for smoke tests (`--smoke`)

All data is held in memory as tensors (uint8 for images) and moved to the
compute device once; batches are sampled per device per round.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import torch

HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

DATASET_INFO = {
    #            in_shape        classes  target-acc (rounds-to-target)
    "mnist":    ((1, 28, 28),    10,      0.98),
    "fmnist":   ((1, 28, 28),    10,      0.88),
    "cifar10":  ((3, 32, 32),    10,      0.75),
    "cifar100": ((3, 32, 32),    100,     0.45),
    "ucihar":   ((9, 128),       6,       0.88),
}

_MEAN_STD = {
    "mnist": ((0.1307,), (0.3081,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


@dataclass
class DataBundle:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int
    in_shape: tuple
    is_image_uint8: bool
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    augment: bool = False

    def to(self, device):
        for k in ("x_train", "y_train", "x_test", "y_test", "mean", "std"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))
        return self

    # ------------------------------------------------------------------ #
    def prep(self, x: torch.Tensor, train: bool) -> torch.Tensor:
        """uint8 -> normalised float (+ light augmentation for CIFAR)."""
        if self.is_image_uint8:
            x = x.float().div_(255.0)
            if train and self.augment:
                x = _augment_crop_flip(x)
            x = (x - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        return x


def _augment_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    n, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i.view(n, 1) + torch.arange(h, device=x.device).view(1, h))          # n,h
    cols = (j.view(n, 1) + torch.arange(w, device=x.device).view(1, w))          # n,w
    idx_n = torch.arange(n, device=x.device).view(n, 1, 1)
    out = xp.permute(0, 2, 3, 1)[idx_n, rows.view(n, h, 1), cols.view(n, 1, w)]  # n,h,w,c
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out.contiguous()


# --------------------------------------------------------------------------- #
def load_dataset(name: str, root: str = "./data", smoke: bool = False) -> DataBundle:
    name = name.lower()
    if name not in DATASET_INFO:
        raise ValueError(f"unknown dataset {name}")
    in_shape, ncls, _ = DATASET_INFO[name]
    if smoke:
        return _synthetic(name, in_shape, ncls)
    if name == "ucihar":
        return _load_har(root)
    import torchvision.datasets as tvd
    cls = {"mnist": tvd.MNIST, "fmnist": tvd.FashionMNIST,
           "cifar10": tvd.CIFAR10, "cifar100": tvd.CIFAR100}[name]
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)

    def to_tensor(ds):
        x = ds.data
        x = torch.as_tensor(np.asarray(x))
        if x.ndim == 3:                      # MNIST-type (N,28,28)
            x = x.unsqueeze(1)
        else:                                # CIFAR (N,32,32,3)
            x = x.permute(0, 3, 1, 2)
        y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
        return x.contiguous(), y

    xtr, ytr = to_tensor(tr)
    xte, yte = to_tensor(te)
    m, s = _MEAN_STD[name]
    return DataBundle(name, xtr, ytr, xte, yte, ncls, in_shape, True,
                      torch.tensor(m), torch.tensor(s), augment=name.startswith("cifar"))


def _load_har(root: str) -> DataBundle:
    """UCI HAR raw inertial signals, standardised per channel."""
    cache = os.path.join(root, "ucihar.npz")
    if not os.path.exists(cache):
        os.makedirs(root, exist_ok=True)
        print(f"[data] downloading UCI HAR from {HAR_URL}")
        raw = urllib.request.urlopen(HAR_URL, timeout=120).read()
        outer = zipfile.ZipFile(io.BytesIO(raw))
        inner_name = [n for n in outer.namelist() if n.endswith(".zip")]
        z = zipfile.ZipFile(io.BytesIO(outer.read(inner_name[0]))) if inner_name else outer
        sig = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y",
               "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]

        def find(suffix):
            m = [n for n in z.namelist() if n.endswith(suffix)]
            if not m:
                raise FileNotFoundError(suffix)
            return z.read(m[0]).decode()

        def split(part):
            xs = [np.loadtxt(io.StringIO(find(f"{part}/Inertial Signals/{s}_{part}.txt"))) for s in sig]
            x = np.stack(xs, 1).astype(np.float32)                       # N,9,128
            y = np.loadtxt(io.StringIO(find(f"{part}/y_{part}.txt"))).astype(np.int64) - 1
            return x, y

        xtr, ytr = split("train")
        xte, yte = split("test")
        np.savez_compressed(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    d = np.load(cache)
    xtr, xte = d["xtr"], d["xte"]
    mu = xtr.mean((0, 2), keepdims=True)
    sd = xtr.std((0, 2), keepdims=True) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    return DataBundle("ucihar", torch.tensor(xtr), torch.tensor(d["ytr"]), torch.tensor(xte),
                      torch.tensor(d["yte"]), 6, (9, 128), False)


def _synthetic(name, in_shape, ncls, n_train=4000, n_test=1000, seed=0) -> DataBundle:
    g = torch.Generator().manual_seed(seed)
    if len(in_shape) == 2:      # time series (HAR): class-specific sinusoids per channel
        c, T = in_shape
        freqs = torch.rand((ncls, c), generator=g) * 8 + 1
        phase = torch.rand((ncls, c), generator=g) * 6.28
        tt = torch.linspace(0, 1, T)

        def make(n):
            y = torch.randint(0, ncls, (n,), generator=g)
            x = torch.sin(2 * np.pi * freqs[y][:, :, None] * tt + phase[y][:, :, None])
            return (x + 0.8 * torch.randn((n, c, T), generator=g)).float(), y
    else:                       # images: spatially smooth class patterns (upsampled 4x4)
        c, h, w = in_shape
        low = torch.randn((ncls, c, 4, 4), generator=g)
        protos = torch.nn.functional.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        protos = protos / protos.std()

        def make(n):
            y = torch.randint(0, ncls, (n,), generator=g)
            x = protos[y] + 1.0 * torch.randn((n, c, h, w), generator=g)
            return x.float(), y
    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return DataBundle(name + "-synthetic", xtr, ytr, xte, yte, ncls, in_shape, False)


# --------------------------------------------------------------------------- #
def partition(y: np.ndarray, n_devices: int, scheme: str, alpha: float,
              rng: np.random.Generator) -> list[np.ndarray]:
    """Index sets per IoT device.  'iid' or 'dirichlet' (label skew, alpha)."""
    y = np.asarray(y)
    n = len(y)
    if scheme == "iid":
        perm = rng.permutation(n)
        return [np.sort(s) for s in np.array_split(perm, n_devices)]
    if scheme != "dirichlet":
        raise ValueError(scheme)
    classes = np.unique(y)
    parts = [[] for _ in range(n_devices)]
    for c in classes:
        idx = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet(alpha * np.ones(n_devices))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for d, s in enumerate(np.split(idx, cuts)):
            parts[d].extend(s.tolist())
    # guarantee a minimum shard so every device can form a batch
    min_size = 8
    for d in range(n_devices):
        if len(parts[d]) < min_size:
            parts[d].extend(rng.choice(n, min_size - len(parts[d]), replace=False).tolist())
    return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


class DeviceSampler:
    """Per-device epoch-style batch sampler over its local shard."""

    def __init__(self, shards: list[np.ndarray], rng: np.random.Generator):
        self.shards = shards
        self.rng = rng
        self.ptr = [0] * len(shards)
        self.order = [rng.permutation(s) for s in shards]

    def next(self, i: int, b: int) -> np.ndarray:
        out = []
        while len(out) < b:
            if self.ptr[i] >= len(self.order[i]):
                self.order[i] = self.rng.permutation(self.shards[i])
                self.ptr[i] = 0
            take = min(b - len(out), len(self.order[i]) - self.ptr[i])
            out.extend(self.order[i][self.ptr[i]: self.ptr[i] + take])
            self.ptr[i] += take
        return np.asarray(out, dtype=np.int64)
