"""Compare two rounds.jsonl files field by field, ignoring wall-clock timing."""
import json, sys, math

IGNORE = {"timing"}
# fields added after the reference run (pure additions, values not compared)
NEW_KEYS = {"bits_sync", "bits_total", "cum_bits_total", "t_sync_max"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in IGNORE and k not in NEW_KEYS}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                return f"{path}/{k} missing on one side"
            d = diff(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) and isinstance(b, float):
        return None if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) else f"{path}: {a} vs {b}"
    return None if a == b else f"{path}: {a} vs {b}"


A = [strip(json.loads(l)) for l in open(sys.argv[1])]
B = [strip(json.loads(l)) for l in open(sys.argv[2])]
assert len(A) == len(B), (len(A), len(B))
for i, (x, y) in enumerate(zip(A, B)):
    d = diff(x, y)
    if d:
        print(f"MISMATCH round {i}: {d}")
        sys.exit(1)
print(f"IDENTICAL: {len(A)} rounds")
