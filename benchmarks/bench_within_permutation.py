"""Scale verification for synchronized within-subject permutation inference.

Runs 1,000 and 10,000 random resampling rounds with a fixed batch=32 on
n = 100 subjects x k = 8 conditions (m = 28 pairs), and reports:

- the wall-clock time of each call,
- the peak RSS (ru_maxrss delta) of each call,
- the maximum shapes of every intermediate array of the engine (instrumented run).

A correct implementation processes the rounds in batches and never stores anything
proportional to n_resamples * n or n_resamples * m, so the peak memory and the
intermediate shapes must be identical for 1,000 and 10,000 rounds.

Usage: python benchmarks/bench_within_permutation.py
"""

import resource
import time

import numpy as np

import pingouin as pg
from pingouin.resampling import _iter_assignments, _paired_t_stats

N, K, BATCH = 100, 8, 32
M = K * (K - 1) // 2


def make_data():
    rng = np.random.default_rng(42)
    vals = rng.normal(size=(N, K))
    rows = [{"subj": i, "cond": j, "dv": vals[i, j]} for i in range(N) for j in range(K)]
    import pandas as pd

    return pd.DataFrame(rows), vals


def peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def instrumented_shapes(values, n_resamples):
    """Re-run the engine's accumulation loop, recording the max shape of every
    intermediate array (mirrors pingouin.resampling._within_perm_stats)."""
    cfg = pg.WithinPermutation(n_resamples, seed=0, batch=BATCH)
    i1, i2 = np.triu_indices(K, k=1)
    shapes = {}
    rows = np.arange(N)
    counts = np.zeros(M, dtype=np.int64)
    for perm in _iter_assignments(N, K, cfg):
        shapes["perm (assignments)"] = np.maximum(
            shapes.get("perm (assignments)", 0), np.array(perm.shape)
        )
        vals = values[rows[None, :, None], perm]
        shapes["permuted table"] = np.maximum(shapes.get("permuted table", 0), np.array(vals.shape))
        t_perm = _paired_t_stats(vals, i1, i2)
        shapes["t_perm (batch stats)"] = np.maximum(
            shapes.get("t_perm (batch stats)", 0), np.array(t_perm.shape)
        )
        counts += (np.abs(t_perm) >= 0).sum(axis=0)
    shapes["counts (accumulator)"] = np.array(counts.shape)
    return shapes


def main():
    df, values = make_data()
    print(f"design: n={N} subjects, k={K} conditions, m={M} pairs, batch={BATCH}")
    print(f"{'n_resamples':>12} {'time (s)':>10} {'peak RSS delta (MB)':>20}")
    for n_res in [1000, 10000]:
        cfg = pg.WithinPermutation(n_res, seed=0, batch=BATCH)
        before = peak_rss_kb()
        t0 = time.perf_counter()
        pg.pairwise_tests(data=df, dv="dv", within="cond", subject="subj", resampling=cfg)
        dt = time.perf_counter() - t0
        delta_mb = (peak_rss_kb() - before) / 1024
        print(f"{n_res:>12} {dt:>10.3f} {delta_mb:>20.1f}")
    for n_res in [1000, 10000]:
        shapes = instrumented_shapes(values, n_res)
        print(f"intermediate max shapes (n_resamples={n_res}):")
        for name, shape in shapes.items():
            print(f"  {name:<24} {tuple(shape)}")


if __name__ == "__main__":
    main()
