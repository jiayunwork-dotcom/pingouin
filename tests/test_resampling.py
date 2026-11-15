"""Tests for synchronized within-subject permutation (randomization) inference.

Covers pg.WithinPermutation, the ``resampling`` argument of pg.ttest and
pg.pairwise_tests, and the family-wise padjust='maxT' correction.

The reference expectations are computed by independent brute-force enumerations of the
complete space of condition-label assignments (see ``_ref_exact``), which never calls the
new engine (pingouin.resampling).
"""

import math
import subprocess
import sys
import textwrap
from itertools import combinations, permutations, product
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import pingouin as pg
from pingouin import WithinPermutation, multicomp, pairwise_tests, ttest
from pingouin.resampling import _iter_assignments

###############################################################################
# Independent reference implementation (never calls the new engine)
###############################################################################


def _ref_tstat(d):
    """Paired T-statistic of the differences d, in pure Python."""
    d = [float(v) for v in d]
    n = len(d)
    mean = sum(d) / n
    if all(v == d[0] for v in d):  # zero variance
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    return mean / (math.sqrt(var) / math.sqrt(n))


def _ref_exact(values):
    """Enumerate the complete (k!)^n space of within-subject condition assignments.

    Returns the (m,) observed T-values and the (total, m) permuted T-values, with
    m = k * (k - 1) / 2 pairs in combinations() order. Numerically identical
    assignments (tied values) are kept and counted separately.
    """
    values = [list(map(float, row)) for row in values]
    n, k = len(values), len(values[0])
    pairs = list(combinations(range(k), 2))
    perms = list(permutations(range(k)))
    t_obs = np.array([_ref_tstat([r[a] - r[b] for r in values]) for a, b in pairs])
    stats = np.array(
        [
            [
                _ref_tstat([values[i][asg[i][a]] - values[i][asg[i][b]] for i in range(n)])
                for a, b in pairs
            ]
            for asg in product(perms, repeat=n)
        ]
    )
    return t_obs, stats


def _ref_tail(stats, t_obs, alternative="two-sided"):
    """Reference tail proportion; ties are counted, with a small tolerance."""
    if alternative == "two-sided":
        a, b = np.abs(stats), abs(t_obs)
    elif alternative == "greater":
        a, b = np.asarray(stats, dtype=float), t_obs
    else:  # 'less': P(T <= t_obs) = P(-T >= -t_obs)
        a, b = -np.asarray(stats, dtype=float), -t_obs
    if np.isinf(b):
        return float(np.mean(a >= b))
    tol = 1e-9 * max(1.0, abs(b))
    return float(np.mean(a >= b - tol))


def _ref_maxt_p(stats, t_obs):
    """Reference maxT-corrected p-values: proportion of rounds whose family maximum
    absolute T reaches the observed absolute T of each pair."""
    mx = np.abs(stats).max(axis=1)
    return np.array([_ref_tail(mx, abs(t), "greater") for t in t_obs])


###############################################################################
# Datasets (small integer tables, hand-checkable)
###############################################################################

# 2 conditions, 4 subjects, differences [2, -1, 3, 1]
DATA2 = [[3, 1], [1, 2], [4, 1], [2, 1]]
# 2 conditions, tied differences [1, 1, 4]
DATA2_TIES = [[2, 1], [2, 1], [5, 1]]
# 2 conditions, all-zero differences
DATA2_ZERO = [[1, 1], [2, 2], [3, 3]]
# 2 conditions, constant non-zero differences [1, 1, 1]
DATA2_CONST = [[2, 1], [3, 2], [4, 3]]
# 3 conditions, 5 subjects, with within-subject tied values
DATA3 = [[3, 1, 2], [1, 2, 1], [4, 2, 3], [2, 1, 1], [5, 3, 4]]
# 3 conditions, strongly unequal variances across conditions
DATA3_HET = [[1, 0, 12], [2, 3, 31], [3, 2, 25], [1, 5, 18], [4, 1, 22]]
# 3 conditions, 4 subjects, all values distinct within each subject
DATA3_COUNTER = [[2, 4, 1], [1, 7, 6], [8, 2, 3], [4, 2, 5]]


def _long_df(values, conds=None, dv="dv"):
    """Long-format dataframe from a subject x condition matrix (subject s0, s1, ...)."""
    k = len(values[0])
    conds = conds if conds is not None else [chr(ord("A") + j) for j in range(k)]
    rows = []
    for i, row in enumerate(values):
        for c, v in zip(conds, row):
            rows.append({"subj": f"s{i}", "cond": c, dv: v})
    return pd.DataFrame(rows)


def _pivot(values, conds=None):
    """Complete-case subject x condition matrix, in the same order as pairwise_tests."""
    df = _long_df(values, conds)
    return df.pivot_table(index="subj", columns="cond", values="dv", observed=True).dropna()


class TestWithinPermutation(TestCase):
    """Test the WithinPermutation configuration object."""

    def test_defaults_and_repr(self):
        cfg = WithinPermutation()
        assert cfg.n_resamples == 9999
        assert cfg.seed == 0
        assert cfg.batch == 128
        assert repr(cfg) == "WithinPermutation(n_resamples=9999, seed=0, batch=128)"
        cfg = WithinPermutation(n_resamples="exact", seed=3, batch=8)
        assert cfg.n_resamples == "exact"
        assert repr(cfg) == "WithinPermutation(n_resamples='exact', seed=3, batch=8)"
        # numpy integers are accepted and normalized to Python int
        cfg = WithinPermutation(n_resamples=np.int64(10), seed=np.int32(2), batch=np.int16(4))
        assert (cfg.n_resamples, cfg.seed, cfg.batch) == (10, 2, 4)
        assert type(cfg.n_resamples) is int and type(cfg.seed) is int
        # equality / hash
        assert WithinPermutation(10, 2, 4) == cfg
        assert hash(WithinPermutation(10, 2, 4)) == hash(cfg)
        assert WithinPermutation(10, 2, 4) != WithinPermutation(10, 2, 5)
        assert cfg != "not a config"

    def test_invalid_n_resamples(self):
        for bad in [
            0,
            -1,
            -100,
            1.5,
            10.0,
            True,
            False,
            np.True_,
            None,
            "EXACT",
            "random",
            "Exact",
            [10],
        ]:
            with pytest.raises(ValueError):
                WithinPermutation(n_resamples=bad)

    def test_invalid_seed(self):
        for bad in [-1, -100, 2**64, 2**64 + 1, 1.5, 0.0, True, False, np.True_, None, "0"]:
            with pytest.raises(ValueError):
                WithinPermutation(seed=bad)
        # boundary values are valid
        WithinPermutation(seed=0)
        WithinPermutation(seed=2**64 - 1)

    def test_invalid_batch(self):
        for bad in [0, -1, 1.5, 128.0, True, False, np.True_, None, "128"]:
            with pytest.raises(ValueError):
                WithinPermutation(batch=bad)

    def test_config_reuse_no_state_consumption(self):
        """A config object can be reused: calls never consume its initial random state."""
        cfg = WithinPermutation(n_resamples=500, seed=9, batch=32)
        x, y = [3, 1, 4, 2, 5, 2], [1, 2, 1, 1, 3, 1]
        r1 = ttest(x, y, paired=True, resampling=cfg)
        r2 = ttest(x, y, paired=True, resampling=cfg)
        pd.testing.assert_frame_equal(r1, r2)
        # The object itself is unchanged
        assert (cfg.n_resamples, cfg.seed, cfg.batch) == (500, 9, 32)


class TestTtestResampling(TestCase):
    """Test ttest(..., resampling=WithinPermutation)."""

    def test_exact_two_conditions(self):
        """Exact enumeration of the 2**n swaps, checked against the reference."""
        x = [v[0] for v in DATA2]
        y = [v[1] for v in DATA2]
        t_ref, stats = _ref_exact(DATA2)
        for alt in ["two-sided", "greater", "less"]:
            tt = ttest(x, y, paired=True, alternative=alt, resampling=WithinPermutation("exact"))
            assert tt.loc["T_test", "T"] == t_ref[0]
            assert tt.loc["T_test", "dof"] == 3
            assert tt.loc["T_test", "alternative"] == alt
            assert tt.loc["T_test", "p_val"] == _ref_tail(stats[:, 0], t_ref[0], alt)
            assert tt.loc["T_test", "method"] == "within-permutation"
            assert tt.loc["T_test", "resamples"] == 16
            assert tt.loc["T_test", "exact"] == True  # noqa: E712
            # No parametric-distribution quantities in this branch
            for col in ["CI95", "BF10", "power"]:
                assert col not in tt.columns
        # Hand-checked: |T| = 1.5 is reached by 6 of the 16 swaps (two-sided p = 0.375)
        tt = ttest(x, y, paired=True, resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "p_val"] == 0.375

    def test_exact_tied_differences(self):
        """Tied values: numerically identical swaps are kept and counted separately."""
        x = [v[0] for v in DATA2_TIES]
        y = [v[1] for v in DATA2_TIES]
        t_ref, stats = _ref_exact(DATA2_TIES)
        tt = ttest(x, y, paired=True, resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "T"] == t_ref[0]
        assert tt.loc["T_test", "p_val"] == _ref_tail(stats[:, 0], t_ref[0])
        assert tt.loc["T_test", "resamples"] == 8

    def test_all_zero_differences(self):
        """All-zero differences: T = 0 (not NaN) and p = 1 for every tail."""
        x = [v[0] for v in DATA2_ZERO]
        y = [v[1] for v in DATA2_ZERO]
        for alt in ["two-sided", "greater", "less"]:
            tt = ttest(x, y, paired=True, alternative=alt, resampling=WithinPermutation("exact"))
            assert tt.loc["T_test", "T"] == 0.0
            assert tt.loc["T_test", "p_val"] == 1.0
            assert tt.loc["T_test", "cohen_d"] == 0.0

    def test_constant_nonzero_differences(self):
        """Constant non-zero differences: T = +-inf, counted towards the tails."""
        x = [v[0] for v in DATA2_CONST]
        y = [v[1] for v in DATA2_CONST]
        # d = [1, 1, 1]: T = +inf. Only the all-plus (all-minus) swap gives +inf (-inf).
        tt = ttest(x, y, paired=True, resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "T"] == np.inf
        assert tt.loc["T_test", "p_val"] == 0.25  # 2 of 8 swaps give |T| = inf
        tt = ttest(x, y, paired=True, alternative="greater", resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "p_val"] == 0.125  # 1 of 8 swaps gives T = +inf
        tt = ttest(x, y, paired=True, alternative="less", resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "p_val"] == 1.0  # every swap gives T <= +inf
        # Reversed sign: T = -inf
        tt = ttest(y, x, paired=True, resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "T"] == -np.inf
        assert tt.loc["T_test", "p_val"] == 0.25
        tt = ttest(y, x, paired=True, alternative="less", resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "p_val"] == 0.125

    def test_nan_pairwise_deletion(self):
        """Missing values are removed pairwise before resampling."""
        x = [1.0, 2.0, np.nan, 5.0]
        y = [2.0, 1.0, 1.0, 1.0]
        t_ref, stats = _ref_exact([[1.0, 2.0], [2.0, 1.0], [5.0, 1.0]])
        tt = ttest(x, y, paired=True, resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "T"] == t_ref[0]
        assert tt.loc["T_test", "dof"] == 2
        assert tt.loc["T_test", "p_val"] == _ref_tail(stats[:, 0], t_ref[0])
        assert tt.loc["T_test", "resamples"] == 8

    def test_invalid_inputs(self):
        cfg = WithinPermutation(100, seed=1)
        x, y = [3, 1, 4, 2], [1, 2, 1, 1]
        # Infinite values are rejected (after pairwise NaN deletion)
        with pytest.raises(ValueError, match="infinite"):
            ttest([1, 2, np.inf], [1, 2, 3], paired=True, resampling=cfg)
        with pytest.raises(ValueError, match="infinite"):
            ttest([1, 2, -np.inf], [1, 2, 3], paired=True, resampling=cfg)
        # Unequal lengths: no automatic switch to independent samples
        with pytest.raises(ValueError, match="equal length"):
            ttest([1, 2, 3], [1, 2], paired=True, resampling=cfg)
        # Unpaired design
        with pytest.raises(ValueError, match="paired"):
            ttest(x, y, paired=False, resampling=cfg)
        with pytest.raises(ValueError, match="paired"):
            ttest(x, y, resampling=cfg)
        # One-sample test (scalar y)
        with pytest.raises(ValueError):
            ttest(x, 4, paired=True, resampling=cfg)
        # Multi-dimensional samples
        with pytest.raises(ValueError, match="one-dimensional"):
            ttest([[1, 2], [3, 4]], [1, 2, 3, 4], paired=True, resampling=cfg)
        # Non-numeric samples
        with pytest.raises(ValueError, match="numeric"):
            ttest(["a", "b"], ["c", "d"], paired=True, resampling=cfg)
        # Fewer than two complete pairs
        with pytest.raises(ValueError, match="two complete pairs"):
            ttest([1.0], [2.0], paired=True, resampling=cfg)
        with pytest.raises(ValueError, match="two complete pairs"):
            ttest([1.0, np.nan], [2.0, 1.0], paired=True, resampling=cfg)
        # resampling must be a WithinPermutation object
        for bad in [True, 100, "exact", {"n_resamples": 100}]:
            with pytest.raises(ValueError, match="WithinPermutation"):
                ttest(x, y, paired=True, resampling=bad)

    def test_random_reproducibility(self):
        """Same seed -> same result; the config object is reusable."""
        x, y = [3, 1, 4, 2, 6, 3], [1, 2, 1, 1, 2, 2]
        cfg = WithinPermutation(999, seed=42)
        r1 = ttest(x, y, paired=True, resampling=cfg)
        r2 = ttest(x, y, paired=True, resampling=cfg)
        pd.testing.assert_frame_equal(r1, r2)
        assert r1.loc["T_test", "resamples"] == 999
        assert r1.loc["T_test", "exact"] == False  # noqa: E712
        # A different seed gives a different stream (checked on the assignments)
        s1 = np.concatenate(list(_iter_assignments(4, 2, WithinPermutation(50, seed=1))))
        s2 = np.concatenate(list(_iter_assignments(4, 2, WithinPermutation(50, seed=2))))
        assert not np.array_equal(s1, s2)

    def test_random_never_switches_to_exact(self):
        """Random mode draws exactly n_resamples rounds, even if the space is smaller."""
        x, y = [2, 1, 5], [1, 1, 1]  # space has only 2**3 = 8 assignments
        tt = ttest(x, y, paired=True, resampling=WithinPermutation(9999, seed=0))
        assert tt.loc["T_test", "resamples"] == 9999
        assert tt.loc["T_test", "exact"] == False  # noqa: E712
        # Plus-one correction: p = (1 + count) / (1 + n_resamples)
        p = tt.loc["T_test", "p_val"]
        count = p * (9999 + 1) - 1
        assert abs(count - round(count)) < 1e-9 and 0 <= round(count) <= 9999

    def test_exact_true_proportion(self):
        """Exact mode uses the true proportion of the complete space, no plus-one."""
        x, y = [v[0] for v in DATA2], [v[1] for v in DATA2]
        tt = ttest(x, y, paired=True, resampling=WithinPermutation("exact"))
        p = tt.loc["T_test", "p_val"]
        count = p * 16
        assert abs(count - round(count)) < 1e-9  # integer count over the 16 assignments

    def test_batch_invariance(self):
        """The batch size only bounds the workspace: identical results for any batch."""
        x, y = [3, 1, 4, 2, 6, 3, 5, 2], [1, 2, 1, 1, 2, 2, 3, 1]
        ref = ttest(x, y, paired=True, resampling=WithinPermutation(500, seed=7, batch=1))
        for batch in [2, 3, 7, 128, 1000]:
            out = ttest(x, y, paired=True, resampling=WithinPermutation(500, seed=7, batch=batch))
            pd.testing.assert_frame_equal(ref, out)
        # Exact mode too
        ref = ttest(x, y, paired=True, resampling=WithinPermutation("exact", batch=1))
        out = ttest(x, y, paired=True, resampling=WithinPermutation("exact", batch=1000))
        pd.testing.assert_frame_equal(ref, out)

    def test_stream_prefix_property(self):
        """Increasing n_resamples only extends the stream: the prefix is unchanged."""
        n, k = 5, 3
        s50 = np.concatenate(list(_iter_assignments(n, k, WithinPermutation(50, seed=3))))
        s200 = np.concatenate(list(_iter_assignments(n, k, WithinPermutation(200, seed=3))))
        assert s50.shape == (50, n, k) and s200.shape == (200, n, k)
        assert_array_equal(s200[:50], s50)
        # ... for any batch size
        s_b1 = np.concatenate(
            list(_iter_assignments(n, k, WithinPermutation(200, seed=3, batch=1)))
        )
        s_b7 = np.concatenate(
            list(_iter_assignments(n, k, WithinPermutation(200, seed=3, batch=7)))
        )
        assert_array_equal(s_b1, s200)
        assert_array_equal(s_b7, s200)

    def test_exact_enumeration_complete(self):
        """Exact mode enumerates every assignment exactly once, in a fixed order."""
        e1 = np.concatenate(list(_iter_assignments(4, 2, WithinPermutation("exact", batch=1))))
        e9 = np.concatenate(list(_iter_assignments(4, 2, WithinPermutation("exact", batch=1000))))
        assert e1.shape == (16, 4, 2)
        assert_array_equal(e1, e9)  # batch-invariant
        # All 2**4 assignments are distinct: complete coverage of the space
        assert len(np.unique(e1.reshape(16, -1), axis=0)) == 16
        # 3 conditions: (3!)**3 = 216 distinct assignments
        e3 = np.concatenate(list(_iter_assignments(3, 3, WithinPermutation("exact"))))
        assert e3.shape == (216, 3, 3)
        assert len(np.unique(e3.reshape(216, -1), axis=0)) == 216

    def test_exact_overflow(self):
        """Exact spaces > 100,000 assignments raise OverflowError before resampling."""
        # 2**17 = 131,072 > 100,000
        x = np.arange(17.0)
        with pytest.raises(OverflowError, match="131072"):
            ttest(x, x + 1, paired=True, resampling=WithinPermutation("exact"))
        # 2**1000: the check is arithmetic, before any exponential enumeration
        x = np.arange(1000.0)
        with pytest.raises(OverflowError, match="100000"):
            ttest(x, x + 1, paired=True, resampling=WithinPermutation("exact"))
        # 2**16 = 65,536 is still allowed
        x = np.arange(16.0)
        tt = ttest(x, x + 1, paired=True, resampling=WithinPermutation("exact"))
        assert tt.loc["T_test", "resamples"] == 65536

    def test_random_statistical_bias(self):
        """Random mode converges to the exact tail probabilities.

        n_resamples = 20,000; the binomial standard error at p = 0.5 is
        sqrt(0.25 / 20000) = 0.0035, so the tolerance 0.02 is about 5.7 SE.
        """
        x = [v[0] for v in DATA2]
        y = [v[1] for v in DATA2]
        t_ref, stats = _ref_exact(DATA2)
        for alt in ["two-sided", "greater", "less"]:
            p_exact = _ref_tail(stats[:, 0], t_ref[0], alt)
            for seed in [0, 1, 2, 3, 4]:
                tt = ttest(
                    x,
                    y,
                    paired=True,
                    alternative=alt,
                    resampling=WithinPermutation(20000, seed=seed),
                )
                assert abs(tt.loc["T_test", "p_val"] - p_exact) < 0.02

    def test_no_global_side_effects(self):
        """No modification of the NumPy global random state, the inputs or pg.options."""
        x = np.array([3.0, 1, 4, 2, 6, 3])
        y = np.array([1.0, 2, 1, 1, 2, 2])
        x0, y0 = x.copy(), y.copy()
        opts0 = dict(pg.options)
        cfg = WithinPermutation(500, seed=11)
        np.random.seed(2024)
        ttest(x, y, paired=True, resampling=cfg)
        draws = np.random.rand(4)
        np.random.seed(2024)
        assert_array_equal(draws, np.random.rand(4))  # global RNG untouched
        assert_array_equal(x, x0)
        assert_array_equal(y, y0)
        assert dict(pg.options) == opts0
        assert (cfg.n_resamples, cfg.seed, cfg.batch) == (500, 11, 128)

    def test_failure_isolation(self):
        """A failed call does not affect subsequent calls."""
        x, y = [3, 1, 4, 2], [1, 2, 1, 1]
        with pytest.raises(OverflowError):
            ttest(
                np.arange(20.0),
                np.arange(20.0) + 1,
                paired=True,
                resampling=WithinPermutation("exact"),
            )
        with pytest.raises(ValueError):
            ttest(x, y, paired=False, resampling=WithinPermutation(100))
        cfg = WithinPermutation(999, seed=5)
        r1 = ttest(x, y, paired=True, resampling=cfg)
        r2 = ttest(x, y, paired=True, resampling=cfg)
        pd.testing.assert_frame_equal(r1, r2)

    def test_resampling_none_unchanged(self):
        """resampling=None keeps the standard parametric output."""
        x, y = [3, 1, 4, 2], [1, 2, 1, 1]
        tt = ttest(x, y, paired=True)
        for col in ["T", "dof", "alternative", "p_val", "CI95", "cohen_d", "power", "BF10"]:
            assert col in tt.columns
        for col in ["method", "resamples", "exact"]:
            assert col not in tt.columns


class TestPairwiseResampling(TestCase):
    """Test pairwise_tests(..., resampling=WithinPermutation)."""

    def test_exact_two_conditions(self):
        """Exact 2-condition family, checked against the reference enumeration."""
        df = _long_df(DATA2)
        t_ref, stats = _ref_exact(DATA2)
        for alt in ["two-sided", "greater", "less"]:
            pt = pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                alternative=alt,
                resampling=WithinPermutation("exact"),
            )
            assert pt.shape[0] == 1
            row = pt.iloc[0]
            assert row["Contrast"] == "cond" and row["A"] == "A" and row["B"] == "B"
            assert row["Paired"] == True and row["Parametric"] == True  # noqa: E712
            assert row["T"] == t_ref[0]
            assert row["dof"] == 3.0
            assert row["p_unc"] == _ref_tail(stats[:, 0], t_ref[0], alt)
            assert row["resamples"] == 16
            assert row["exact"] == True  # noqa: E712
            assert "BF10" not in pt.columns
            assert "p_corr" not in pt.columns  # padjust='none' by default

    def test_exact_two_conditions_matches_ttest(self):
        """With 2 conditions, pairwise T, p_unc and maxT p_corr match the single ttest
        with the same configuration and the same effective subject order."""
        df = _long_df(DATA2)
        cfg = WithinPermutation(999, seed=11)
        pt = pairwise_tests(
            data=df, dv="dv", within="cond", subject="subj", resampling=cfg, padjust="maxT"
        )
        piv = _pivot(DATA2)  # same complete subjects, same order
        tt = ttest(piv["A"].to_numpy(), piv["B"].to_numpy(), paired=True, resampling=cfg)
        assert pt.loc[0, "T"] == tt.loc["T_test", "T"]
        assert pt.loc[0, "p_unc"] == tt.loc["T_test", "p_val"]
        # With a single pair, the family max |T| is |T| itself
        assert pt.loc[0, "p_corr"] == pt.loc[0, "p_unc"]
        # Same with exact enumeration
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        tt = ttest(
            piv["A"].to_numpy(),
            piv["B"].to_numpy(),
            paired=True,
            resampling=WithinPermutation("exact"),
        )
        assert pt.loc[0, "T"] == tt.loc["T_test", "T"]
        assert pt.loc[0, "p_unc"] == tt.loc["T_test", "p_val"]
        assert pt.loc[0, "p_corr"] == tt.loc["T_test", "p_val"]

    def test_exact_three_conditions(self):
        """Exact 3-condition family: T, p_unc and maxT p_corr of every pair."""
        df = _long_df(DATA3)
        t_ref, stats = _ref_exact(DATA3)
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        assert pt.shape[0] == 3
        assert pt["A"].tolist() == ["A", "A", "B"]
        assert pt["B"].tolist() == ["B", "C", "C"]
        assert pt["resamples"].tolist() == [7776] * 3
        assert pt["exact"].all()
        assert (pt["p_adjust"] == "maxT").all()
        p_maxt_ref = _ref_maxt_p(stats, t_ref)
        for j in range(3):
            assert pt.loc[j, "T"] == t_ref[j]
            assert pt.loc[j, "p_unc"] == _ref_tail(stats[:, j], t_ref[j])
            assert pt.loc[j, "p_corr"] == p_maxt_ref[j]
            assert pt.loc[j, "dof"] == 4.0
        # maxT is a family-wise correction: p_corr >= p_unc
        assert (pt["p_corr"] >= pt["p_unc"]).all()

    def test_exact_heteroscedastic(self):
        """Unequal variances across conditions are handled (reference-checked)."""
        df = _long_df(DATA3_HET)
        t_ref, stats = _ref_exact(DATA3_HET)
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        p_maxt_ref = _ref_maxt_p(stats, t_ref)
        for j in range(3):
            assert pt.loc[j, "T"] == t_ref[j]
            assert pt.loc[j, "p_unc"] == _ref_tail(stats[:, j], t_ref[j])
            assert pt.loc[j, "p_corr"] == p_maxt_ref[j]

    def test_exact_ties_and_degenerate(self):
        """Tied values, all-zero and constant non-zero differences in the family."""
        # All-zero differences in every pair: T = 0, p = 1
        df = _long_df([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        assert (pt["T"] == 0).all()
        assert (pt["p_unc"] == 1.0).all()
        assert (pt["p_corr"] == 1.0).all()
        # Constant non-zero differences in every pair: T = +-inf, no round dropped
        df = _long_df([[2, 3, 4], [3, 4, 5], [4, 5, 6]])  # B - A = C - B = 1
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        t_ref, stats = _ref_exact([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
        p_maxt_ref = _ref_maxt_p(stats, t_ref)
        for j in range(3):
            assert pt.loc[j, "T"] == t_ref[j]
            assert pt.loc[j, "p_unc"] == _ref_tail(stats[:, j], t_ref[j])
            assert pt.loc[j, "p_corr"] == p_maxt_ref[j]
        assert np.isinf(pt["T"]).all()

    def test_missing_subjects_listwise(self):
        """Subjects with any missing value are removed from the whole family."""
        values = [row[:] for row in DATA3]
        df = _long_df(values + [[9, 9, np.nan], [8, 8, 8]])  # s5 incomplete, s6 complete
        df.loc[len(df)] = ["s7", "A", np.nan]  # s7 has a single missing row
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        # Same as the family without the incomplete subjects s5 and s7
        pt_ref = pairwise_tests(
            data=_long_df(DATA3 + [[8, 8, 8]]),
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        pd.testing.assert_frame_equal(pt.reset_index(drop=True), pt_ref.reset_index(drop=True))

    def test_repeated_measures_averaged(self):
        """Repeated measurements of a subject-condition cell are averaged first."""
        df = _long_df(DATA2)
        # Give subject s0 two measurements in condition A: 1 and 5 (mean 3 = original)
        df = pd.concat(
            [df, pd.DataFrame({"subj": ["s0", "s0"], "cond": ["A", "A"], "dv": [1.0, 5.0]})],
            ignore_index=True,
        )
        df = df[df["dv"] != 3.0]  # remove the original s0-A = 3 measurement
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
        )
        pt_ref = pairwise_tests(
            data=_long_df(DATA2),
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
        )
        pd.testing.assert_frame_equal(pt, pt_ref)

    def test_shuffled_long_table(self):
        """The row order of the long-format table does not change the results."""
        df = _long_df(DATA3)
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation(999, seed=3),
            padjust="maxT",
        )
        df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
        pt_shuffled = pairwise_tests(
            data=df_shuffled,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation(999, seed=3),
            padjust="maxT",
        )
        pd.testing.assert_frame_equal(pt, pt_shuffled)

    def test_categorical_condition_order(self):
        """The custom order of a categorical condition column is respected."""
        df = _long_df(DATA3)
        df["cond"] = pd.Categorical(df["cond"], categories=["C", "A", "B"], ordered=True)
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        # Pairs follow the categorical order: (C, A), (C, B), (A, B)
        assert pt["A"].tolist() == ["C", "C", "A"]
        assert pt["B"].tolist() == ["A", "B", "B"]
        # Reference with the columns in the same order
        values = [[r[2], r[0], r[1]] for r in DATA3]  # columns C, A, B
        t_ref, stats = _ref_exact(values)
        p_maxt_ref = _ref_maxt_p(stats, t_ref)
        for j in range(3):
            assert pt.loc[j, "T"] == t_ref[j]
            assert pt.loc[j, "p_unc"] == _ref_tail(stats[:, j], t_ref[j])
            assert pt.loc[j, "p_corr"] == p_maxt_ref[j]

    def test_one_sided_alternatives(self):
        """One-sided p_unc of every pair, checked against the reference."""
        df = _long_df(DATA3)
        t_ref, stats = _ref_exact(DATA3)
        for alt in ["greater", "less"]:
            pt = pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                alternative=alt,
                resampling=WithinPermutation("exact"),
            )
            assert (pt["alternative"] == alt).all()
            for j in range(3):
                assert pt.loc[j, "T"] == t_ref[j]
                assert pt.loc[j, "p_unc"] == _ref_tail(stats[:, j], t_ref[j], alt)

    def test_rounding_option_does_not_leak(self):
        """Display rounding options never affect the p-values or their correction."""
        df = _long_df(DATA3)
        cfg = WithinPermutation("exact")
        ref = pairwise_tests(
            data=df, dv="dv", within="cond", subject="subj", resampling=cfg, padjust="holm"
        )
        pg.options["round"] = 3
        try:
            rounded = pairwise_tests(
                data=df, dv="dv", within="cond", subject="subj", resampling=cfg, padjust="holm"
            )
        finally:
            pg.set_default_options()
        # p_corr is computed from the full-precision p_unc, then rounded for display
        np.testing.assert_allclose(rounded["p_corr"], ref["p_corr"], atol=5e-4)
        np.testing.assert_allclose(rounded["p_unc"], ref["p_unc"], atol=5e-4)
        # Same for the single ttest
        x, y = [v[0] for v in DATA2], [v[1] for v in DATA2]
        pg.options["round"] = 3
        try:
            tt = ttest(x, y, paired=True, resampling=cfg)
        finally:
            pg.set_default_options()
        assert tt.loc["T_test", "p_val"] == round(
            ttest(x, y, paired=True, resampling=cfg).loc["T_test", "p_val"], 3
        )

    def test_maxt_requires_resampling_and_two_sided(self):
        df = _long_df(DATA3)
        with pytest.raises(ValueError, match="maxT"):
            pairwise_tests(data=df, dv="dv", within="cond", subject="subj", padjust="maxT")
        for alt in ["greater", "less"]:
            with pytest.raises(ValueError, match="two-sided"):
                pairwise_tests(
                    data=df,
                    dv="dv",
                    within="cond",
                    subject="subj",
                    alternative=alt,
                    padjust="maxT",
                    resampling=WithinPermutation(100),
                )

    def test_invalid_designs(self):
        df = _long_df(DATA3)
        cfg = WithinPermutation(100, seed=1)
        # Between-subject design
        with pytest.raises(ValueError, match="within"):
            pairwise_tests(data=df, dv="dv", between="cond", resampling=cfg)
        # Mixed design
        df2 = df.copy()
        df2["group"] = (
            ["g1", "g2"] * (len(df2) // 2)
            if len(df2) % 2 == 0
            else (["g1", "g2", "g1"] * 5)[: len(df2)]
        )
        with pytest.raises(ValueError, match="within"):
            pairwise_tests(
                data=df2, dv="dv", within="cond", between="group", subject="subj", resampling=cfg
            )
        # Two within-subject factors
        df3 = _long_df([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        df3["cond2"] = ["u", "u", "v", "v"] * 3
        with pytest.raises(ValueError, match="within"):
            pairwise_tests(
                data=df3, dv="dv", within=["cond", "cond2"], subject="subj", resampling=cfg
            )
        # Non-parametric
        with pytest.raises(ValueError, match="parametric"):
            pairwise_tests(
                data=df, dv="dv", within="cond", subject="subj", parametric=False, resampling=cfg
            )
        # Pairwise deletion of missing values
        with pytest.raises(ValueError, match="listwise"):
            pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                nan_policy="pairwise",
                resampling=cfg,
            )
        # Missing subject identifier
        with pytest.raises(ValueError, match="subject"):
            pairwise_tests(data=df, dv="dv", within="cond", resampling=cfg)
        # resampling must be a WithinPermutation object
        with pytest.raises(ValueError, match="WithinPermutation"):
            pairwise_tests(data=df, dv="dv", within="cond", subject="subj", resampling=100)

    def test_invalid_data(self):
        cfg = WithinPermutation(100, seed=1)
        # Fewer than two complete subjects
        with pytest.raises(ValueError, match="two complete subjects"):
            pairwise_tests(
                data=_long_df([[1, 2]]), dv="dv", within="cond", subject="subj", resampling=cfg
            )
        # No complete subject at all
        df = _long_df([[1, np.nan], [np.nan, 2]])
        with pytest.raises(ValueError, match="two complete subjects"):
            pairwise_tests(data=df, dv="dv", within="cond", subject="subj", resampling=cfg)
        # A single condition
        df = _long_df([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="two unique values"):
            pairwise_tests(
                data=df[df["cond"] == "A"], dv="dv", within="cond", subject="subj", resampling=cfg
            )
        # Infinite values
        df = _long_df([[1, np.inf], [2, 3]])
        with pytest.raises(ValueError, match="infinite"):
            pairwise_tests(data=df, dv="dv", within="cond", subject="subj", resampling=cfg)

    def test_holm_on_permutation_pvalues(self):
        """Classical corrections are applied to the randomization p_unc."""
        df = _long_df(DATA3)
        for method in ["holm", "bonf", "fdr_bh", "sidak"]:
            pt = pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                padjust=method,
                resampling=WithinPermutation("exact"),
            )
            expected = multicomp(pt["p_unc"].to_numpy(), alpha=0.05, method=method)[1]
            assert_array_equal(pt["p_corr"].to_numpy(), expected)
            assert (pt["p_adjust"] == method).all()

    def test_maxt_joint_vs_independent_pairs(self):
        """Counterexample: permuting each pair independently and then taking the max
        gives a wrong joint max-|T| distribution (3 conditions)."""
        t_ref, stats = _ref_exact(DATA3_COUNTER)
        n = len(DATA3_COUNTER)
        # Correct joint distribution of the family max |T| over the (3!)**n space
        mx_joint = np.abs(stats).max(axis=1)
        # Wrong "independent pairs" distribution: for each pair, the 2**n sign flips of
        # its observed differences, combined as if the pairs were independent
        pair_dists = []
        for a, b in combinations(range(3), 2):
            d0 = [row[a] - row[b] for row in DATA3_COUNTER]
            flips = np.array(
                [
                    abs(_ref_tstat([s * v for s, v in zip(signs, d0)]))
                    for signs in product([1, -1], repeat=n)
                ]
            )
            pair_dists.append(flips)
        grid = np.sort(np.unique(mx_joint))
        surv_correct = np.array([np.mean(mx_joint >= c) for c in grid])
        surv_wrong = np.array([1.0 - np.prod([np.mean(d < c) for d in pair_dists]) for c in grid])
        gap = np.abs(surv_correct - surv_wrong).max()
        # The two distributions differ by more than 0.3 in survival probability:
        # the maxT correction must use the joint (synchronized) rounds
        assert gap > 0.3
        # And the engine matches the correct joint distribution
        pt = pairwise_tests(
            data=_long_df(DATA3_COUNTER),
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        for j in range(3):
            c = abs(t_ref[j])
            assert pt.loc[j, "p_corr"] == np.mean(mx_joint >= c)

    def test_random_statistical_bias(self):
        """Random mode converges to the exact p_unc and maxT p_corr.

        n_resamples = 20,000 over 5 seeds; the binomial standard error at p = 0.5 is
        sqrt(0.25 / 20000) = 0.0035, so the tolerance 0.02 is about 5.7 SE.
        """
        df = _long_df(DATA3)
        exact = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
            padjust="maxT",
        )
        for seed in [0, 1, 2, 3, 4]:
            rnd = pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                resampling=WithinPermutation(20000, seed=seed),
                padjust="maxT",
            )
            assert np.abs(rnd["p_unc"] - exact["p_unc"]).max() < 0.02
            assert np.abs(rnd["p_corr"] - exact["p_corr"]).max() < 0.02

    def test_batch_invariance_and_prefix(self):
        """Batch size never changes the results; n_resamples only extends the stream."""
        df = _long_df(DATA3)
        ref = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation(500, seed=5, batch=1),
            padjust="maxT",
        )
        for batch in [2, 3, 128, 1000]:
            out = pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                resampling=WithinPermutation(500, seed=5, batch=batch),
                padjust="maxT",
            )
            pd.testing.assert_frame_equal(ref, out)

    def test_config_reuse_and_state(self):
        """The config object is reusable; no global state is modified."""
        df = _long_df(DATA3)
        df0 = df.copy(deep=True)
        opts0 = dict(pg.options)
        cfg = WithinPermutation(300, seed=17, batch=16)
        np.random.seed(99)
        r1 = pairwise_tests(data=df, dv="dv", within="cond", subject="subj", resampling=cfg)
        draws = np.random.rand(4)
        np.random.seed(99)
        assert_array_equal(draws, np.random.rand(4))  # global RNG untouched
        r2 = pairwise_tests(data=df, dv="dv", within="cond", subject="subj", resampling=cfg)
        pd.testing.assert_frame_equal(r1, r2)
        pd.testing.assert_frame_equal(df, df0)  # input data unchanged
        assert dict(pg.options) == opts0
        assert (cfg.n_resamples, cfg.seed, cfg.batch) == (300, 17, 16)

    def test_exact_overflow(self):
        """(k!)**n > 100,000 raises OverflowError before any resampling."""
        df = _long_df([[float(i), float(i + 1), float(i + 2)] for i in range(7)])  # 6**7
        with pytest.raises(OverflowError, match="279936"):
            pairwise_tests(
                data=df,
                dv="dv",
                within="cond",
                subject="subj",
                resampling=WithinPermutation("exact"),
            )
        # 6**6 = 46,656 is still allowed
        df = _long_df([[float(i), float(i + 1), float(i + 2)] for i in range(6)])
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
        )
        assert pt.loc[0, "resamples"] == 46656

    def test_return_desc_and_effsize(self):
        """Descriptives and effect sizes use the same complete subjects."""
        df = _long_df(DATA3)
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            return_desc=True,
            effsize="cohen",
            resampling=WithinPermutation("exact"),
        )
        values = np.array(DATA3, dtype=float)
        for j, (a, b) in enumerate(combinations(range(3), 2)):
            x, y = values[:, a], values[:, b]
            assert pt.loc[j, "mean_A"] == x.mean()
            assert pt.loc[j, "std_A"] == x.std(ddof=1)
            assert pt.loc[j, "mean_B"] == y.mean()
            assert pt.loc[j, "std_B"] == y.std(ddof=1)
            assert pt.loc[j, "cohen"] == pg.compute_effsize(x, y, paired=True, eftype="cohen")

    def test_observed_T_matches_parametric(self):
        """The observed T-values are the same statistic as the parametric branch."""
        df = _long_df(DATA3)
        para = pairwise_tests(data=df, dv="dv", within="cond", subject="subj")
        perm = pairwise_tests(
            data=df,
            dv="dv",
            within="cond",
            subject="subj",
            resampling=WithinPermutation("exact"),
        )
        np.testing.assert_allclose(perm["T"], para["T"], rtol=1e-12)
        assert_array_equal(perm["dof"], para["dof"])

    def test_single_element_within_list(self):
        """within=['cond'] (one-element list) is also supported."""
        df = _long_df(DATA2)
        pt = pairwise_tests(
            data=df,
            dv="dv",
            within=["cond"],
            subject="subj",
            resampling=WithinPermutation("exact"),
        )
        assert pt.shape[0] == 1
        assert pt.loc[0, "resamples"] == 16

    @pytest.mark.slow
    def test_memory_bounded_by_batch(self):
        """Peak memory is bounded by the batch, not by n_resamples.

        Runs 1,000 and 10,000 random rounds (batch=32, 100 subjects, 8 conditions) in a
        clean subprocess: an implementation that secretly stores per-round data of size
        n_resamples * n (or * m) would need tens of MB more for 10,000 rounds.
        """
        code = textwrap.dedent(
            """
            import resource
            import numpy as np
            import pingouin as pg

            rng = np.random.default_rng(0)
            n, k = 100, 8
            rows = []
            vals = rng.normal(size=(n, k))
            import pandas as pd
            for i in range(n):
                for j in range(k):
                    rows.append({"subj": i, "cond": j, "dv": vals[i, j]})
            df = pd.DataFrame(rows)
            peaks = []
            for n_res in [1000, 10000]:
                before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                pg.pairwise_tests(
                    data=df, dv="dv", within="cond", subject="subj",
                    resampling=pg.WithinPermutation(n_res, seed=0, batch=32),
                )
                after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peaks.append(after - before)
            print(peaks[0], peaks[1])
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        peak_1000, peak_10000 = map(float, out.stdout.strip().split())
        # 10x more rounds must not grow the peak memory by more than 20 MB
        assert peak_10000 - peak_1000 < 20 * 1024, (peak_1000, peak_10000)
