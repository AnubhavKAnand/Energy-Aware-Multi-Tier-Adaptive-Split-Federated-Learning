#!/usr/bin/env python
"""
MTA-SFL experiment driver.

    python main.py                  # EVERYTHING: MTA-SFL + AdaptSFL + SFLSC3 + S-HSFL,
                                    # 5 datasets x {IID, non-IID} x seeds {0,1,2}, 4000 rounds
    python main.py --pilot          # ~200-round check of all methods on MNIST + CIFAR-10 (timing + sanity)
    python main.py --workers 2      # run two jobs at once on the same GPU
    python main.py --methods mtasfl --datasets cifar10 --seeds 0     # any subset
    python main.py --smoke          # 60 rounds on synthetic data, no downloads

Order is seed-major: every method/dataset/partition for seed 0 first, then
seed 1, then seed 2 - a complete comparison exists after the first pass.

Results: results/<iid|dirichlet>/<dataset>/<method>/seed<k>/
    rounds.jsonl  (every round, written the moment it ends)   eval.jsonl
    summary.json  config.json  meta.json  checkpoint.pt
results/progress.json   live status of all jobs + ETA
results/index.json      final summaries of finished jobs
Re-running the same command skips finished jobs and resumes unfinished ones.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

ALL_DATASETS = ["mnist", "fmnist", "cifar10", "cifar100", "ucihar"]


def parse():
    from mtasfl.methods import ALL_METHODS, METHODS
    ap = argparse.ArgumentParser(description="MTA-SFL experiment driver")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    help=f"subset of {list(METHODS)} (space or comma separated)")
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--partitions", nargs="+", default=["iid", "dirichlet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--out", default=None, help="results folder (default: results, results_pilot, results_smoke)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    ap.add_argument("--workers", type=int, default=1, help="jobs run in parallel")
    ap.add_argument("--config", default=None, help="YAML file with config overrides")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="dotted overrides, e.g. sfl.topk_ratio=0.05 obj.lam1=0.5")
    ap.add_argument("--mode", default="full", choices=["full", "fast"],
                    help="full = real DNN training; fast = network/agents only (no accuracy)")
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="200 rounds, all methods, MNIST + CIFAR-10, non-IID, seed 0")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 60 rounds")
    a = ap.parse_args()

    def flat(xs):
        return [y for x in xs for y in str(x).split(",") if y]
    a.methods, a.datasets, a.partitions = flat(a.methods), flat(a.datasets), flat(a.partitions)
    for m in a.methods:
        if m not in METHODS:
            ap.error(f"unknown method {m}; available: {list(METHODS)}")
    for d in a.datasets:
        if d not in ALL_DATASETS:
            ap.error(f"unknown dataset {d}; available: {ALL_DATASETS}")
    if a.pilot:
        a.datasets = [d for d in a.datasets if d in ("mnist", "cifar10")] or ["mnist", "cifar10"]
        a.partitions, a.seeds = ["dirichlet"], [0]
        if a.rounds == 4000:
            a.rounds = 200
    if a.smoke and a.rounds == 4000:
        a.rounds = 60
    if a.out is None:
        a.out = "results_pilot" if a.pilot else ("results_smoke" if a.smoke else "results")
    return a


def _parse_val(v: str):
    try:
        return json.loads(v)
    except Exception:
        return v


def build_jobs(a):
    jobs = []
    for seed in a.seeds:                         # seed-major
        for ds in a.datasets:
            for part in a.partitions:
                for m in a.methods:
                    jobs.append(dict(seed=seed, dataset=ds, partition=part, method=m,
                                     out=os.path.join(a.out, part, ds, m, f"seed{seed}"),
                                     tag=f"{part}/{ds}/{m}/seed{seed}"))
    return jobs


def finished(job, rounds) -> dict | None:
    p = os.path.join(job["out"], "summary.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            s = json.load(f)
    except Exception:
        return None
    return s if s.get("finished") and s.get("rounds_done", 0) >= rounds else None


# --------------------------------------------------------------------------- #
def run_job(payload):
    """Executed in the main process (workers=1) or in a worker process."""
    job, a, device = payload
    import torch
    from mtasfl.config import Config
    from mtasfl.methods import get_method, simulation_class
    t0 = time.time()
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // max(a.workers, 1)))
        base = Config.from_yaml(a.config) if a.config else Config()
        if a.set:
            base = base.override({k: _parse_val(v) for k, v in (s.split("=", 1) for s in a.set)})
        if a.smoke or a.pilot:
            base = base.override({"sfl.eval_every": 20})
        cfg = base.override({"dataset": job["dataset"], "seed": job["seed"], "mode": a.mode,
                             "sfl.partition": job["partition"]})
        spec = get_method(job["method"])
        sim = simulation_class(spec)(cfg, spec, job["out"], smoke=a.smoke, data_root=a.data_root,
                                     device=device, print_every=a.print_every if not a.smoke else 20,
                                     ckpt_every=a.ckpt_every, resume=not a.no_resume, tag=job["tag"],
                                     tagged=a.workers > 1)
        s = sim.run(a.rounds)
        return dict(job=job, ok=True, summary=s, elapsed=time.time() - t0)
    except Exception:
        return dict(job=job, ok=False, error=traceback.format_exc(), elapsed=time.time() - t0)


class Progress:
    def __init__(self, path, jobs, rounds):
        self.path, self.jobs, self.rounds = path, jobs, rounds
        self.state = {j["tag"]: dict(status="pending", **{k: j[k] for k in ("method", "dataset", "partition", "seed")})
                      for j in jobs}
        self.t0 = time.time()

    def set(self, tag, **kw):
        self.state[tag].update(kw)
        self.write()

    def write(self):
        done = [v for v in self.state.values() if v["status"] == "done" and v.get("elapsed_s")]
        pend = [v for v in self.state.values() if v["status"] in ("pending", "running")]
        eta = None
        if done:
            per_ds = {}
            for v in done:
                per_ds.setdefault(v["dataset"], []).append(v["elapsed_s"])
            mean_all = sum(x for xs in per_ds.values() for x in xs) / sum(len(xs) for xs in per_ds.values())
            eta = sum((sum(per_ds[v["dataset"]]) / len(per_ds[v["dataset"]])) if v["dataset"] in per_ds else mean_all
                      for v in pend)
        out = dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), rounds_per_job=self.rounds,
                   total=len(self.state),
                   done=sum(v["status"] == "done" for v in self.state.values()),
                   skipped_already_done=sum(v.get("skipped", False) for v in self.state.values()),
                   failed=sum(v["status"] == "failed" for v in self.state.values()),
                   running=[k for k, v in self.state.items() if v["status"] == "running"],
                   eta_hours_sequential=None if eta is None else round(eta / 3600, 2),
                   wall_hours=round((time.time() - self.t0) / 3600, 2), jobs=self.state)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, self.path)


def main():
    a = parse()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    os.makedirs(a.out, exist_ok=True)
    jobs = build_jobs(a)
    print(f"[MTA-SFL] {len(jobs)} jobs | methods={a.methods} datasets={a.datasets} partitions={a.partitions} "
          f"seeds={a.seeds} rounds={a.rounds} mode={a.mode} device={device} workers={a.workers} out={a.out}")

    # download datasets once, before any worker starts
    if a.mode == "full" and not a.smoke:
        from mtasfl.learning.data import load_dataset
        for ds in a.datasets:
            print(f"[data] preparing {ds} ...", flush=True)
            load_dataset(ds, a.data_root)

    prog = Progress(os.path.join(a.out, "progress.json"), jobs, a.rounds)
    todo, index = [], []
    for j in jobs:
        s = None if a.no_resume else finished(j, a.rounds)
        if s is not None:
            prog.state[j["tag"]].update(status="done", skipped=True)
            index.append(s)
        else:
            todo.append(j)
    prog.write()
    print(f"[MTA-SFL] {len(jobs) - len(todo)} already finished, {len(todo)} to run. "
          f"Live status: {prog.path}")

    def record(res, k):
        j = res["job"]
        if res["ok"]:
            prog.set(j["tag"], status="done", elapsed_s=round(res["elapsed"], 1),
                     final_test_acc=res["summary"].get("final_test_acc"))
            index.append(res["summary"])
            with open(os.path.join(a.out, "index.json"), "w") as f:
                json.dump(index, f, indent=1, default=str)
        else:
            prog.set(j["tag"], status="failed", error=res["error"][-2000:])
            print(f"\n[FAILED] {j['tag']}\n{res['error']}", file=sys.stderr, flush=True)
        n_done = sum(v["status"] in ("done", "failed") for v in prog.state.values())
        eta = json.load(open(prog.path)).get("eta_hours_sequential")
        print(f"[progress] {n_done}/{len(jobs)} jobs finished | last: {j['tag']} "
              f"({res['elapsed'] / 60:.1f} min) | ETA (sequential) ~{eta} h", flush=True)

    t0 = time.time()
    if a.workers <= 1:
        for k, j in enumerate(todo, 1):
            prog.set(j["tag"], status="running")
            print(f"\n[run {len(jobs) - len(todo) + k}/{len(jobs)}] {j['tag']}", flush=True)
            record(run_job((j, a, device)), k)
    else:
        ctx = mp.get_context("spawn")
        for j in todo:
            prog.state[j["tag"]]["status"] = "queued"
        prog.write()
        with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
            for k, res in enumerate(pool.imap_unordered(run_job, [(j, a, device) for j in todo]), 1):
                record(res, k)
    n_fail = sum(v["status"] == "failed" for v in prog.state.values())
    print(f"\n[MTA-SFL] finished in {(time.time() - t0) / 3600:.2f} h | failed jobs: {n_fail} | "
          f"summaries: {os.path.join(a.out, 'index.json')} | next: python summarize.py --results {a.out}")
    if a.pilot:
        pilot_report(a.out, a)


def pilot_report(out, a):
    """Timing / sanity report after --pilot, with a full-run estimate."""
    rows = []
    for ds in a.datasets:
        for m in a.methods:
            d = os.path.join(out, "dirichlet", ds, m, "seed0")
            try:
                s = json.load(open(os.path.join(d, "summary.json")))
                R = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
            except Exception:
                rows.append((ds, m, None))
                continue
            wall = sum(r["timing"]["round_ms"] for r in R) / 1000 / max(len(R), 1)
            rows.append((ds, m, dict(acc=s.get("final_test_acc"), loss=s.get("final_test_loss"), wall=wall,
                                     lat=s["mean_round_latency_s"], drop=s["drop_rate"])))
    print("\n================ PILOT REPORT ================")
    print(f"{'dataset':>8} {'method':>9} {'acc':>6} {'loss':>7} {'sim s/rnd':>9} {'drop':>5} {'wall s/rnd':>10} {'4000 rnds':>9}")
    for ds, m, r in rows:
        if r is None:
            print(f"{ds:>8} {m:>9}   (missing / failed)")
            continue
        print(f"{ds:>8} {m:>9} {r['acc'] or float('nan'):>6.3f} {r['loss'] or float('nan'):>7.3f} {r['lat']:>9.2f} "
              f"{r['drop']:>5.2f} {r['wall']:>10.3f} {r['wall'] * 4000 / 3600:>8.2f}h")
    est = {}
    for ds, m, r in rows:
        if r:
            est[ds] = est.get(ds, 0.0) + r["wall"] * 4000
    if est:
        small = est.get("mnist", 0.0)
        big = est.get("cifar10", 0.0)
        total = (2 * small + small * 0.6 + 2 * big) * 2 * 3        # mnist, fmnist, har(~0.6x), cifar10/100; x2 partitions x3 seeds
        print(f"\nEstimated full run (all methods, 5 datasets x 2 partitions x 3 seeds, 1 worker): ~{total / 3600:.1f} h"
              f"  (CIFAR-100 assumed = CIFAR-10, UCI HAR ~0.6x MNIST)")
    print("Check that accuracy is rising for every method before launching the full run.")


if __name__ == "__main__":
    main()
