#!/usr/bin/env python
"""
Aggregate finished runs into tables (mean and 95 % confidence interval over seeds).

    python summarize.py                      # reads results/, writes results/summary_table.csv
    python summarize.py --results results_pilot

Output
  <results>/summary_table.csv   one row per (partition, dataset, method):
                                n_seeds, <metric>_mean, <metric>_ci95 for every metric below
  terminal                      compact accuracy / latency / energy table per dataset
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

METRICS = [
    ("final_test_acc", "final accuracy"), ("best_test_acc", "best accuracy"), ("final_test_loss", "final loss"),
    ("rounds_to_target", "rounds to target"), ("sim_time_to_target_s", "time to target [s]"),
    ("energy_to_target_J", "energy to target [J]"), ("bits_to_target", "bits to target"),
    ("mean_round_latency_s", "round latency [s]"), ("energy_per_round_J", "energy / round [J]"),
    ("total_energy_J", "total energy [J]"), ("total_energy_excl_prop_J", "energy excl. propulsion [J]"),
    ("total_energy_prop_J", "propulsion energy [J]"), ("energy_comp_J", "computation energy [J]"),
    ("energy_comm_J", "communication energy [J]"), ("total_energy_cloud_comp_J", "cloud energy [J]"),
    ("total_bits", "total bits"), ("completion_rate", "completion rate"), ("drop_rate", "drop rate"),
    ("latency_violation_rate", "latency violation rate"), ("energy_violation_rate", "energy violation rate"),
    ("deadline_miss", "deadline misses"), ("link_failures", "link failures"), ("aborted", "aborted"),
    ("bridge_rate", "bridge rate"), ("uav_depletions", "UAV depletions"),
    ("first_depletion_round", "first depletion round"), ("churn_per_round", "relay churn / round"),
    ("participation_jain", "participation Jain"), ("device_acc_jain", "device accuracy Jain"),
    ("device_acc_worst10", "worst-10% device accuracy"), ("avg_regret", "average regret"),
]

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def mean_ci(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n - 1, 1.96) * sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--include-unfinished", action="store_true")
    a = ap.parse_args()
    groups = defaultdict(list)
    unfinished = []
    for p in glob.glob(os.path.join(a.results, "*", "*", "*", "seed*", "summary.json")):
        part, ds, m, seed = p.split(os.sep)[-5:-1]
        try:
            s = json.load(open(p))
        except Exception:
            continue
        if not s.get("finished") and not a.include_unfinished:
            unfinished.append(f"{part}/{ds}/{m}/{seed} (round {s.get('rounds_done')})")
            continue
        groups[(part, ds, m)].append(s)
    if not groups:
        print(f"no finished runs under {a.results}")
        return
    rows = []
    for (part, ds, m), ss in sorted(groups.items()):
        row = dict(partition=part, dataset=ds, method=m, n_seeds=len(ss))
        for k, _ in METRICS:
            mu, ci, n = mean_ci([s.get(k) for s in ss])
            row[f"{k}_mean"], row[f"{k}_ci95"], row[f"{k}_n"] = mu, ci, n
        rows.append(row)
    out = os.path.join(a.results, "summary_table.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(r, k, scale=1.0, d=3):
        mu, ci = r[f"{k}_mean"], r[f"{k}_ci95"]
        if mu is None:
            return "-"
        return f"{mu * scale:.{d}f}" + (f"±{ci * scale:.{d}f}" if ci is not None else "")

    print(f"{'part':>9} {'dataset':>8} {'method':>9} {'n':>2} {'accuracy':>15} {'rnds→tgt':>11} "
          f"{'latency s':>13} {'kJ/round':>13} {'drop':>11}")
    for r in rows:
        print(f"{r['partition']:>9} {r['dataset']:>8} {r['method']:>9} {r['n_seeds']:>2} "
              f"{fmt(r, 'final_test_acc'):>15} {fmt(r, 'rounds_to_target', 1, 0):>11} "
              f"{fmt(r, 'mean_round_latency_s', 1, 2):>13} {fmt(r, 'energy_per_round_J', 1e-3, 2):>13} "
              f"{fmt(r, 'drop_rate'):>11}")
    print(f"\nwrote {out} ({len(rows)} groups, {len(METRICS)} metrics each)")
    if unfinished:
        print(f"{len(unfinished)} unfinished runs not included (use --include-unfinished): "
              + ", ".join(unfinished[:5]) + (" ..." if len(unfinished) > 5 else ""))


if __name__ == "__main__":
    main()
