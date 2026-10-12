"""
Block-structured DNNs.  Each model is an ordered list of L blocks; a split
tuple p = (p1, p2, p3) with 1 <= p1 <= p2 <= p3 <= L assigns blocks
  1..p1        -> IoT device      (tier 1)
  p1+1..p2     -> primary UAV     (tier 2)
  p2+1..p3     -> relay UAV       (tier 3)
  p3+1..L      -> cloud           (tier 4)
GroupNorm is used throughout (Sec. III-C relies on it to recalibrate the
activation shift caused by zero-padded top-k features).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4) if c >= 8 else 1, c)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = gn(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), gn(cout))

    def forward(self, x):
        o = F.relu(self.n1(self.c1(x)))
        o = self.n2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


class Head(nn.Module):
    def __init__(self, cin, ncls):
        super().__init__()
        self.fc = nn.Linear(cin, ncls)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class BlockNet(nn.Module):
    def __init__(self, blocks: list[nn.Module], name: str):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.name = name

    @property
    def L(self) -> int:
        return len(self.blocks)

    def run(self, x, a: int, b: int):
        """Blocks a..b inclusive (1-indexed); identity if a > b."""
        for l in range(a, b + 1):
            x = self.blocks[l - 1](x)
        return x

    def forward(self, x):
        return self.run(x, 1, self.L)

    def block_params(self, l: int):
        return list(self.blocks[l - 1].parameters())


def resnet18_gn(ncls: int, in_ch: int = 3) -> BlockNet:
    """ResNet-18 (CIFAR variant) split into L = 10 blocks."""
    stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False), gn(64), nn.ReLU(inplace=True))
    cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1),
           (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
    blocks = [stem] + [BasicBlock(*c) for c in cfg] + [Head(512, ncls)]
    return BlockNet(blocks, "resnet18-gn")


def cnn_mnist(ncls: int = 10) -> BlockNet:
    """4-conv + 2-FC CNN for (Fashion-)MNIST, L = 6 blocks."""
    def cb(ci, co, pool):
        layers = [nn.Conv2d(ci, co, 3, 1, 1, bias=False), gn(co), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    blocks = [cb(1, 32, False), cb(32, 32, True), cb(32, 64, False), cb(64, 64, True),
              nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True)),
              nn.Linear(256, ncls)]
    return BlockNet(blocks, "cnn-gn")


def cnn1d_har(ncls: int = 6, in_ch: int = 9) -> BlockNet:
    """1-D CNN for UCI HAR raw inertial signals (9 x 128), L = 6 blocks."""
    def cb(ci, co, pool):
        layers = [nn.Conv1d(ci, co, 5, 1, 2, bias=False), gn(co), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool1d(2))
        return nn.Sequential(*layers)
    blocks = [cb(in_ch, 64, False), cb(64, 64, True), cb(64, 128, False), cb(128, 128, True),
              nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 64), nn.ReLU(inplace=True)),
              nn.Linear(64, ncls)]
    return BlockNet(blocks, "cnn1d-gn")


def build_model(dataset: str, ncls: int, in_shape: tuple) -> BlockNet:
    d = dataset.split("-")[0]
    if d in ("cifar10", "cifar100"):
        return resnet18_gn(ncls, in_shape[0])
    if d in ("mnist", "fmnist"):
        return cnn_mnist(ncls)
    if d == "ucihar":
        return cnn1d_har(ncls, in_shape[0])
    raise ValueError(dataset)
