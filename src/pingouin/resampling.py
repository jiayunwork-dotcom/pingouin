"""Synchronized within-subject permutation (randomization) inference.

This module provides the public :py:class:`WithinPermutation` configuration object and the
shared engine behind the ``resampling`` argument of :py:func:`pingouin.ttest` and
:py:func:`pingouin.pairwise_tests`.

Design notes
------------
A single resampling round permutes, for every subject independently, the values of *all*
conditions, and every pairwise comparison is computed on that same permuted table. This keeps
the comparisons dependent on each other, which is what the family-wise ``maxT`` correction
needs: the maximum absolute T over the whole family of pairs is taken within each round.

Resamples are processed in batches of ``batch`` rounds. Batching only bounds the temporary
workspace (roughly ``batch * (n * k + m)`` values for n subjects, k conditions and
m = k * (k - 1) / 2 pairs); it never changes the results, because the random stream is
consumed in resample-major order: resample ``r`` always uses the draws
``[r * n * k, (r + 1) * n * k)`` of a single :py:class:`numpy.random.Generator`, whatever the
batch size. For the same reason, increasing ``n_resamples`` only extends the stream: the first
resamples are identical.
"""

import math
from itertools import permutations

import numpy as np

__all__ = ["WithinPermutation"]

# Maximum size of the exact permutation space, i.e. max number of distinct
# within-subject condition assignments (k!) ** n that are enumerated exhaustively.
_MAX_EXACT_RESAMPLES = 100_000


class WithinPermutation:
    """Configuration for within-subject permutation (randomization) inference.

    .. versionadded:: 0.7.1

    Parameters
    ----------
    n_resamples : int or 'exact'
        Number of random resampling rounds (default 9999), or ``'exact'`` to enumerate the
        complete space of :math:`(k!)^n` within-subject condition assignments, where :math:`n`
        is the number of complete subjects and :math:`k` the number of conditions. The exact
        space is capped at 100,000 assignments; larger spaces raise an :py:class:`OverflowError`
        before any resampling starts.
    seed : int
        Seed of the random number generator, an integer in :math:`[0, 2^{64})`. Only used in
        random mode. The same configuration object can be reused across calls: every call
        starts a fresh generator from ``seed``, so repeated calls give identical results and
        never consume the initial state.
    batch : int
        Number of resampling rounds processed at once (default 128). This only bounds the
        temporary workspace; changing it never changes the results for a given ``seed``.

    See Also
    --------
    ttest, pairwise_tests

    Notes
    -----
    The procedure is a synchronized randomization test for repeated measures. Under the joint
    null hypothesis, the :math:`k` condition labels of a complete subject are exchangeable, so
    each resampling round reassigns the observed values of every subject to the condition
    labels independently and uniformly (a random permutation of the :math:`k` conditions per
    subject). All pairwise T-statistics — and, for ``padjust='maxT'``, the family maximum
    absolute T — are recomputed on the same permuted table, which preserves the dependence
    between the comparisons.

    In random mode, exactly ``n_resamples`` rounds are drawn independently with replacement
    (the mode never switches to exact automatically), and p-values use the plus-one correction
    that includes the observed statistic: ``(1 + count) / (1 + n_resamples)``. In exact mode,
    p-values are the true proportions over the complete space, with no plus-one correction.
    Assignments that are numerically identical (e.g. because of tied values) are kept and
    counted separately. Ties with the observed statistic are always counted towards the tail.

    Examples
    --------
    Exact paired permutation test (2 conditions, 5 pairs -> 2**5 = 32 assignments):

    >>> import pandas as pd
    >>> import pingouin as pg
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> pd.set_option("display.max_columns", 20)
    >>> pre = [5.5, 2.4, 6.8, 9.6, 4.2]
    >>> post = [6.4, 3.4, 6.4, 11.0, 4.8]
    >>> cfg = pg.WithinPermutation(n_resamples="exact")
    >>> pg.ttest(pre, post, paired=True, resampling=cfg).round(4)
               T  dof alternative  p_val             method  resamples  exact  cohen_d
    T_test -2.3078    4   two-sided  0.125  within-permutation         32   True   0.2508

    Randomized joint test of all pairs of a repeated-measures factor, with the family-wise
    maxT correction:

    >>> df = pg.read_dataset("mixed_anova.csv")
    >>> cfg = pg.WithinPermutation(n_resamples=999, seed=42)
    >>> pg.pairwise_tests(
    ...     data=df, dv="Scores", within="Time", subject="Subject", resampling=cfg, padjust="maxT"
    ... ).round(3)
      Contrast       A       B  Paired  Parametric      T   dof alternative  p_unc  p_corr p_adjust  resamples  exact  hedges
    0     Time  August  January    True        True -1.740  59.0   two-sided  0.108  0.223     maxT        999  False  -0.328
    1     Time  August     June    True        True -2.743  59.0   two-sided  0.010  0.026     maxT        999  False  -0.483
    2     Time January     June    True        True -1.024  59.0   two-sided  0.296  0.564     maxT        999  False  -0.170
    """

    __slots__ = ("n_resamples", "seed", "batch")

    def __init__(self, n_resamples=9999, seed=0, batch=128):
        # n_resamples: positive non-bool integer, or the string 'exact'
        if isinstance(n_resamples, str):
            if n_resamples != "exact":
                raise ValueError(
                    f"n_resamples must be a positive integer or 'exact', got {n_resamples!r}."
                )
        elif isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
            n_resamples, (int, np.integer)
        ):
            raise ValueError(
                f"n_resamples must be a positive integer or 'exact', got {n_resamples!r}."
            )
        elif n_resamples < 1:
            raise ValueError(f"n_resamples must be a positive integer, got {n_resamples}.")
        else:
            n_resamples = int(n_resamples)
        # seed: non-bool integer in [0, 2**64)
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an integer in [0, 2**64), got {seed!r}.")
        if not 0 <= seed < 2**64:
            raise ValueError(f"seed must be in [0, 2**64), got {seed}.")
        # batch: positive non-bool integer
        if isinstance(batch, (bool, np.bool_)) or not isinstance(batch, (int, np.integer)):
            raise ValueError(f"batch must be a positive integer, got {batch!r}.")
        if batch < 1:
            raise ValueError(f"batch must be a positive integer, got {batch}.")
        self.n_resamples = n_resamples
        self.seed = int(seed)
        self.batch = int(batch)

    def __repr__(self):
        return (
            f"WithinPermutation(n_resamples={self.n_resamples!r}, seed={self.seed}, "
            f"batch={self.batch})"
        )

    def __eq__(self, other):
        if not isinstance(other, WithinPermutation):
            return NotImplemented
        return (
            self.n_resamples == other.n_resamples
            and self.seed == other.seed
            and self.batch == other.batch
        )

    def __hash__(self):
        return hash((self.n_resamples, self.seed, self.batch))


def _check_resampling(resampling):
    """Check that ``resampling`` is a :py:class:`WithinPermutation` object, and return it."""
    if not isinstance(resampling, WithinPermutation):
        raise ValueError(
            "resampling must be a pingouin.WithinPermutation object, got "
            f"{type(resampling).__name__}."
        )
    return resampling


def _exact_space_size(n, k):
    """Size (k!) ** n of the exact within-subject permutation space.

    Raises an :py:class:`OverflowError` if the space exceeds ``_MAX_EXACT_RESAMPLES``. This is
    checked before any resampling starts, so an oversized space never triggers the exponential
    enumeration.
    """
    total = math.factorial(k) ** n
    if total > _MAX_EXACT_RESAMPLES:
        raise OverflowError(
            f"The exact permutation space has ({k}!)^n = {total} assignments, which exceeds "
            f"the maximum of {_MAX_EXACT_RESAMPLES}. Use random resampling "
            "(n_resamples=<int>) instead."
        )
    return total


def _iter_assignments(n, k, resampling):
    """Yield the within-subject permutations, as (batch, n, k) arrays of condition indices.

    ``perm[b, i, j]`` is the index of the original condition assigned to condition ``j`` of
    subject ``i`` in resampling round ``b``. In exact mode, all (k!)^n assignments are
    enumerated in a fixed order; in random mode, exactly ``n_resamples`` independent uniform
    assignments are drawn (with replacement). The batch size only splits the stream into
    chunks: the sequence of assignments is identical for any batch size, and increasing
    ``n_resamples`` only extends it.
    """
    batch = resampling.batch
    if resampling.n_resamples == "exact":
        total = _exact_space_size(n, k)
        # (k!, k) table of the distinct permutations of the k conditions
        table = np.array(list(permutations(range(k))))
        for start in range(0, total, batch):
            idx = np.arange(start, min(start + batch, total))
            # Mixed-radix (base k!) decomposition: one permutation index per subject
            subj_perm = np.stack(np.unravel_index(idx, (table.shape[0],) * n), axis=1)
            yield table[subj_perm]
    else:
        rng = np.random.default_rng(resampling.seed)
        for start in range(0, resampling.n_resamples, batch):
            b = min(batch, resampling.n_resamples - start)
            # Uniform permutation of the k conditions per round and subject. The (b, n, k)
            # array is filled in resample-major order, so round r always consumes the draws
            # [r * n * k, (r + 1) * n * k) of the generator, whatever the batch size. The
            # stable argsort keeps the (measure-zero) ties deterministic.
            yield rng.random((b, n, k)).argsort(axis=2, kind="stable")


def _paired_t_stats(vals, i1, i2):
    """Paired T-statistics of all pairs of columns of ``vals``, (..., n, k) -> (..., m).

    ``i1`` and ``i2`` are the upper-triangle indices of the condition pairs. Degenerate
    (bitwise constant) differences give T = 0 when the mean difference is zero, and +-inf
    otherwise, so that no resampling round is ever dropped.
    """
    n = vals.shape[-2]
    d = vals[..., i1] - vals[..., i2]  # (..., n, m) pairwise differences
    mean = d.mean(axis=-2)
    sd = d.std(axis=-2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / math.sqrt(n))
    # Bitwise constant differences have exactly zero variance
    const = (d == d[..., :1, :]).all(axis=-2)
    return np.where(const, np.where(mean == 0, 0.0, np.copysign(np.inf, mean)), t)


def _within_perm_stats(values, alternative="two-sided", resampling=None, maxt=False):
    """Synchronized within-subject permutation T-statistics and p-values.

    Parameters
    ----------
    values : (n, k) array_like
        Complete-case matrix with one row per subject and one column per condition. Must be
        finite, with n >= 2 and k >= 2 (checked by the callers).
    alternative : {'two-sided', 'greater', 'less'}
        Tail of the test. 'greater' ('less') counts permuted T at least as large (small) as
        observed; 'two-sided' compares absolute T values. Ties are counted towards the tail.
    resampling : WithinPermutation
        Resampling configuration.
    maxt : bool
        If True, also accumulate the family-wise max-|T| statistics (two-sided only).

    Returns
    -------
    out : dict
        't_obs': (m,) observed T-values, 'p_unc': (m,) permutation p-values of each pair,
        'p_maxt': (m,) maxT-corrected p-values or None, 'n_resamples': number of resampling
        rounds actually performed, 'exact': whether the complete space was enumerated.
    """
    assert alternative in ["two-sided", "greater", "less"]
    assert not maxt or alternative == "two-sided", "maxT requires alternative='two-sided'."
    resampling = _check_resampling(resampling)
    values = np.asarray(values, dtype=np.float64)
    n, k = values.shape
    i1, i2 = np.triu_indices(k, k=1)
    m = i1.size
    exact = resampling.n_resamples == "exact"
    if exact:
        # Raises OverflowError before any resampling if the space is too large
        total = _exact_space_size(n, k)
    else:
        total = resampling.n_resamples

    t_obs = _paired_t_stats(values[None], i1, i2)[0]

    # Tail thresholds. Ties are counted towards the tail, with the same relative tolerance as
    # pingouin._perm_pval, so that values that are mathematically equal to the observed
    # statistic but differ by floating-point error are still counted.
    gamma = np.abs(t_obs) * np.finfo(np.float64).eps * 100
    if alternative == "two-sided":
        obs = np.abs(t_obs)
    elif alternative == "greater":
        obs = t_obs
    else:  # 'less': count T_perm <= T_obs, rewritten as -T_perm >= -T_obs
        obs = -t_obs
    # An infinite observed statistic keeps its infinite threshold (inf - inf would be NaN)
    with np.errstate(invalid="ignore"):
        thr = np.where(np.isfinite(obs - gamma), obs - gamma, obs)

    counts = np.zeros(m, dtype=np.int64)
    maxt_counts = np.zeros(m, dtype=np.int64) if maxt else None
    rows = np.arange(n)
    for perm in _iter_assignments(n, k, resampling):
        vals = values[rows[None, :, None], perm]  # (b, n, k) permuted table
        t_perm = _paired_t_stats(vals, i1, i2)  # (b, m) all pairs, same rounds
        if alternative == "two-sided":
            counts += (np.abs(t_perm) >= thr).sum(axis=0)
        elif alternative == "greater":
            counts += (t_perm >= thr).sum(axis=0)
        else:
            counts += (-t_perm >= thr).sum(axis=0)
        if maxt:
            # Family-wise statistic: max absolute T over all pairs of the same round
            mx = np.abs(t_perm).max(axis=1)  # (b,)
            maxt_counts += (mx[:, None] >= thr[None, :]).sum(axis=0)

    if exact:
        # True proportion over the complete space, no plus-one correction
        p_unc = counts / total
        p_maxt = maxt_counts / total if maxt else None
    else:
        # Plus-one correction including the observed statistic
        p_unc = (1 + counts) / (1 + total)
        p_maxt = (1 + maxt_counts) / (1 + total) if maxt else None
    return {
        "t_obs": t_obs,
        "p_unc": p_unc,
        "p_maxt": p_maxt,
        "n_resamples": total,
        "exact": exact,
    }
