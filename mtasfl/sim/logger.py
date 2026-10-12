"""
Result logging.

<out>/
  config.json        full configuration (+ parameter table in meta.json)
  meta.json          model profile, agent memory/compute footprint, environment
  rounds.jsonl       ONE line per round, written + flushed as soon as the round ends
  eval.jsonl         test metrics every `eval_every` rounds
  summary.json       running summary, rewritten every `print_every` rounds and at the end
  checkpoint.pt      resumable state, every `ckpt_every` rounds
"""
from __future__ import annotations

import json
import math
import os

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class RunLogger:
    def __init__(self, out_dir: str, resume_round: int | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self.rounds_path = os.path.join(out_dir, "rounds.jsonl")
        self.eval_path = os.path.join(out_dir, "eval.jsonl")
        if resume_round is not None:
            self._truncate(self.rounds_path, resume_round)
            self._truncate(self.eval_path, resume_round)
            mode = "a"
        else:
            mode = "w"
        self.f_rounds = open(self.rounds_path, mode, buffering=1)
        self.f_eval = open(self.eval_path, mode, buffering=1)

    @staticmethod
    def _truncate(path, upto_round):
        """Keep only lines with round < upto_round (resume consistency)."""
        if not os.path.exists(path):
            return
        keep = []
        with open(path) as f:
            for line in f:
                try:
                    if json.loads(line)["round"] < upto_round:
                        keep.append(line)
                except Exception:
                    pass
        with open(path, "w") as f:
            f.writelines(keep)

    def write_json(self, name, obj):
        tmp = os.path.join(self.dir, name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(_clean(obj), f, indent=2)
        os.replace(tmp, os.path.join(self.dir, name))

    def log_round(self, rec: dict):
        self.f_rounds.write(json.dumps(_clean(rec)) + "\n")
        self.f_rounds.flush()
        os.fsync(self.f_rounds.fileno())

    def log_eval(self, rec: dict):
        self.f_eval.write(json.dumps(_clean(rec)) + "\n")
        self.f_eval.flush()
        os.fsync(self.f_eval.fileno())

    def close(self):
        self.f_rounds.close()
        self.f_eval.close()


def jain(x) -> float:
    x = np.asarray(x, float)
    if x.size == 0 or np.all(x == 0):
        return float("nan")
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))
