"""Tests for group-sequential designs (``sequential_bounds``, ``sequential_ttest`` and the
sequential options of ``power_ttest`` / ``power_ttest2n``)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pingouin import sequential_bounds, sequential_ttest, ttest
from pingouin.power import power_ttest, power_ttest2n

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _corr(t):
    """Correlation matrix of the interim Z-statistics: corr(Z_j, Z_k) = sqrt(t_j / t_k)."""
    t = np.asarray(t, float)
    out = np.eye(len(t))
    for i in range(len(t)):
        for j in range(len(t)):
            out[i, j] = np.sqrt(min(t[i], t[j]) / max(t[i], t[j]))
    return out


def _multivariate_crossing(t, bound, alternative):
    """Independent reference: cumulative crossing probability, from scipy's MVN CDF."""
    t = np.asarray(t, float)
    bound = np.asarray(bound, float)
    k = len(t)
    if alternative == "greater":
        lower, upper = np.full(k, -np.inf), bound
    elif alternative == "less":
        lower, upper = bound, np.full(k, np.inf)
    else:
        lower, upper = -bound, bound
    cdf = stats.multivariate_normal.cdf(
        upper,
        mean=np.zeros(k),
        cov=_corr(t),
        lower_limit=lower,
        maxpts=10_000_000,
        abseps=1e-12,
        releps=1e-12,
    )
    return 1 - cdf


def _multivariate_crossing_prefixes(t, bound, alternative):
    """Cumulative crossing probability at every look, from an independent MVN calculation."""
    return np.array(
        [_multivariate_crossing(t[: k + 1], bound[: k + 1], alternative) for k in range(len(t))]
    )


def _fixed_power(d, n, tsample, alpha, alternative):
    """Fixed-sample non-central T power (same convention as ``power_ttest``)."""
    nc = d * np.sqrt(n / tsample)
    dof = (n - 1) * tsample
    if alternative == "greater":
        crit = stats.t.ppf(1 - alpha, dof)
        return stats.nct.sf(crit, dof, nc)
    if alternative == "less":
        crit = stats.t.ppf(alpha, dof)
        return 1 - stats.nct.sf(crit, dof, nc)
    crit = stats.t.ppf(1 - alpha / 2, dof)
    return stats.nct.sf(crit, dof, nc) + 1 - stats.nct.sf(-crit, dof, nc)


def _sequential_power_reference(d, n, t, tsample, alpha, alternative, spending):
    """Independent reference sequential power, from scipy's MVN CDF."""
    bounds = sequential_bounds(t, alpha=alpha, alternative=alternative, spending=spending)
    nominal = bounds["nominal_p"].to_numpy()
    t = np.asarray(t, float)
    # Marginal power of each look, converted to the shift of a standard normal statistic.
    marginal = np.array(
        [_fixed_power(d, n * tk, tsample, nominal[k], alternative) for k, tk in enumerate(t)]
    )
    if alternative == "greater":
        shift = bounds["bound"].to_numpy() + stats.norm.ppf(np.clip(marginal, 1e-15, 1 - 1e-16))
    elif alternative == "less":
        shift = bounds["bound"].to_numpy() - stats.norm.ppf(np.clip(marginal, 1e-15, 1 - 1e-16))
    else:
        shift = np.zeros_like(marginal)
        for k in range(len(t)):
            lo, hi = 0.0, 60.0
            target = min(max(marginal[k], 1e-15), 1 - 1e-16)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                prob = stats.norm.cdf(-bounds["bound"].iloc[k] - mid) + stats.norm.sf(
                    bounds["bound"].iloc[k] - mid
                )
                if prob < target:
                    lo = mid
                else:
                    hi = mid
            shift[k] = 0.5 * (lo + hi)
    bound = bounds["bound"].to_numpy()
    if alternative == "greater":
        lower, upper = np.full(len(t), -np.inf), bound
    elif alternative == "less":
        lower, upper = bound, np.full(len(t), np.inf)
    else:
        lower, upper = -bound, bound
    cdf = stats.multivariate_normal.cdf(
        upper - shift,
        mean=np.zeros(len(t)),
        cov=_corr(t),
        lower_limit=lower - shift,
        maxpts=20_000_000,
        abseps=1e-13,
        releps=1e-13,
    )
    return 1 - cdf


# --------------------------------------------------------------------------------------------
# sequential_bounds
# --------------------------------------------------------------------------------------------


class TestSequentialBounds:
    def test_reference_values_greater(self):
        """Published reference boundaries (Jennison & Turnbull 2000)."""
        looks = [0.2, 0.4, 0.6, 0.8, 1.0]
        obf = sequential_bounds(looks, alpha=0.025, alternative="greater", spending="obf")
        assert np.allclose(obf["bound"], [4.877, 3.357, 2.680, 2.290, 2.031], atol=1e-3)
        pocock = sequential_bounds(looks, alpha=0.025, alternative="greater", spending="pocock")
        assert np.allclose(pocock["bound"], [2.438, 2.427, 2.410, 2.397, 2.386], atol=1e-3)

    def test_reference_values_greater_precision(self):
        """Higher-precision values, obtained from an independent MVN calculation."""
        looks = [0.2, 0.4, 0.6, 0.8, 1.0]
        obf = sequential_bounds(looks, alpha=0.025, alternative="greater", spending="obf")
        assert np.allclose(
            obf["bound"],
            [4.8768849488, 3.3570119216, 2.6802800645, 2.2898167677, 2.0310320435],
            atol=1e-6,
        )
        pocock = sequential_bounds(looks, alpha=0.025, alternative="greater", spending="pocock")
        assert np.allclose(
            pocock["bound"],
            [2.4379766881, 2.4268138584, 2.4101941425, 2.3966492654, 2.3859996498],
            atol=1e-6,
        )

    def test_columns_and_dtypes(self):
        bounds = sequential_bounds([0.25, 0.5, 1.0], alpha=0.05)
        assert list(bounds.columns) == [
            "look",
            "t",
            "bound",
            "nominal_p",
            "alpha_look",
            "alpha_spent",
        ]
        assert bounds["look"].tolist() == [1, 2, 3]
        assert np.array_equal(bounds["t"].to_numpy(), [0.25, 0.5, 1.0])
        assert bounds["alpha_spent"].iloc[-1] == 0.05
        # alpha_look is the increment of alpha_spent
        assert np.allclose(bounds["alpha_look"], np.diff(np.r_[0.0, bounds["alpha_spent"]]))

    def test_single_look(self):
        for alternative in ["two-sided", "greater", "less"]:
            bounds = sequential_bounds([1.0], alpha=0.05, alternative=alternative)
            assert bounds["nominal_p"].iloc[0] == 0.05
            assert bounds["alpha_look"].iloc[0] == 0.05
            assert bounds["alpha_spent"].iloc[0] == 0.05

    def test_symmetric_two_sided_and_less(self):
        looks = [0.3, 0.6, 1.0]
        greater = sequential_bounds(looks, alpha=0.05, alternative="greater")
        less = sequential_bounds(looks, alpha=0.05, alternative="less")
        # The lower boundary is the mirror image of the upper boundary
        assert np.allclose(less["bound"], -greater["bound"], atol=1e-12)
        assert np.allclose(less["nominal_p"], greater["nominal_p"], atol=1e-15)
        # Symmetric two-sided boundaries (one per tail)
        two = sequential_bounds(looks, alpha=0.05, alternative="two-sided")
        assert np.all(two["bound"] > 0)
        # For a linear spending function (Pocock), a two-sided test at alpha has the same
        # boundaries as a one-sided test at alpha / 2 in the mirrored tail.
        two_poc = sequential_bounds(looks, alpha=0.05, alternative="two-sided", spending="pocock")
        one_poc = sequential_bounds(looks, alpha=0.025, alternative="greater", spending="pocock")
        assert np.allclose(two_poc["bound"], one_poc["bound"], atol=1e-12)

    def test_nominal_p_matches_bound(self):
        bounds = sequential_bounds([0.2, 0.5, 0.75, 1.0], alpha=0.05, alternative="two-sided")
        expected = 2 * stats.norm.sf(bounds["bound"])
        assert np.allclose(bounds["nominal_p"], expected, atol=1e-15)
        bounds = sequential_bounds([0.2, 0.5, 0.75, 1.0], alpha=0.05, alternative="greater")
        assert np.allclose(bounds["nominal_p"], stats.norm.sf(bounds["bound"]), atol=1e-15)
        bounds = sequential_bounds([0.2, 0.5, 0.75, 1.0], alpha=0.05, alternative="less")
        assert np.allclose(bounds["nominal_p"], stats.norm.cdf(bounds["bound"]), atol=1e-15)

    def test_spending_function(self):
        alpha, looks = 0.05, [0.2, 0.4, 0.6, 0.8, 1.0]
        t = np.asarray(looks, float)
        for spending, expected in [
            ("obf", 2 * stats.norm.sf(stats.norm.isf(alpha / 2) / np.sqrt(t))),
            ("pocock", alpha * np.log(1 + (np.e - 1) * t)),
            (0.5, alpha * t**0.5),
            (2.0, alpha * t**2.0),
        ]:
            bounds = sequential_bounds(looks, alpha=alpha, spending=spending)
            assert np.allclose(bounds["alpha_spent"], expected, atol=1e-14)

    def test_cumulative_crossing_probability(self):
        """Cumulative crossing probability must equal alpha(t_k), checked independently."""
        designs = [
            ([0.2, 0.4, 0.6, 0.8, 1.0], 0.05, "two-sided", "obf"),
            ([0.2, 0.4, 0.6, 0.8, 1.0], 0.025, "greater", "pocock"),
            ([0.1, 0.3, 0.55, 0.8, 1.0], 0.05, "less", 0.5),
            ([0.3, 0.6, 1.0], 0.05, "two-sided", "pocock"),
        ]
        for looks, alpha, alternative, spending in designs:
            bounds = sequential_bounds(
                looks, alpha=alpha, alternative=alternative, spending=spending
            )
            reference = _multivariate_crossing_prefixes(
                looks, bounds["bound"].to_numpy(), alternative
            )
            assert np.allclose(reference, bounds["alpha_spent"], atol=1e-6)

    def test_prefix_invariance(self):
        """Appending looks must not change the boundaries of the previous looks."""
        looks = [0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
        full = sequential_bounds(looks, alpha=0.05)
        # A single-look design is a fixed-sample test (it spends the whole alpha), so prefix
        # invariance only applies from two looks onward.
        for m in range(2, len(looks) + 1):
            prefix = sequential_bounds(looks[:m], alpha=0.05)
            assert np.array_equal(prefix["bound"].to_numpy(), full["bound"].to_numpy()[:m])
            assert np.array_equal(prefix["nominal_p"].to_numpy(), full["nominal_p"].to_numpy()[:m])
            assert np.array_equal(
                prefix["alpha_spent"].to_numpy(), full["alpha_spent"].to_numpy()[:m]
            )
        # Same property for other spending functions and alternatives
        for alternative, spending in [("greater", "pocock"), ("less", 0.5)]:
            full = sequential_bounds(looks, alpha=0.025, alternative=alternative, spending=spending)
            for m in range(2, len(looks) + 1):
                prefix = sequential_bounds(
                    looks[:m], alpha=0.025, alternative=alternative, spending=spending
                )
                assert np.array_equal(prefix["bound"].to_numpy(), full["bound"].to_numpy()[:m])

    def test_deterministic(self):
        a = sequential_bounds([0.2, 0.4, 0.7, 1.0], alpha=0.05, spending=0.7)
        b = sequential_bounds([0.2, 0.4, 0.7, 1.0], alpha=0.05, spending=0.7)
        pd.testing.assert_frame_equal(a, b)

    def test_twenty_looks_speed(self):
        import time

        t = np.linspace(0.05, 1.0, 20)
        start = time.time()
        bounds = sequential_bounds(t, alpha=0.05)
        assert time.time() - start < 2.0
        assert np.all(np.diff(bounds["bound"]) < 0)  # OBF boundaries decrease
        assert bounds["alpha_spent"].iloc[-1] == 0.05

    def test_validation(self):
        with pytest.raises(ValueError):
            sequential_bounds(None)
        with pytest.raises(ValueError):
            sequential_bounds([])
        with pytest.raises(ValueError):
            sequential_bounds([[0.5, 1.0]])
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 0.4])
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 0.5])
        with pytest.raises(ValueError):
            sequential_bounds([0.0, 1.0])
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.5])
        with pytest.raises(ValueError):
            sequential_bounds([0.5, np.inf])
        with pytest.raises(ValueError):
            sequential_bounds([True, 1.0])
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], alpha=0.0)
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], alpha=0.5)
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], alpha=1.0)
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], alternative="both")
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], spending="bad")
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], spending=0)
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], spending=-1.0)
        with pytest.raises(ValueError):
            sequential_bounds([0.5, 1.0], spending=np.inf)


# --------------------------------------------------------------------------------------------
# Sequential power
# --------------------------------------------------------------------------------------------


class TestSequentialPower:
    def test_looks_none_regression(self):
        """looks=None must reproduce the historical values exactly (bit for bit)."""
        cases = [
            dict(d=0.5, n=20, contrast="one-sample"),
            dict(d=0.5, n=20),
            dict(d=0.5, power=0.80, alternative="greater"),
            dict(d=0.5, power=0.80),
            dict(n=20, power=0.80, contrast="paired"),
            dict(n=20, power=0.80),
            dict(d=0.5, n=20, power=0.80, alpha=None),
            dict(d=0.5, n=20, alternative="less"),
        ]
        expected = [
            0.5645044184390209,
            0.3379390289250411,
            50.150783386861136,
            63.76561,
            0.6604416546230263,
            0.9091290326820792,
            0.44301676584494126,
            0.0006909466675297793,
        ]
        for kwargs, value in zip(cases, expected):
            result = power_ttest(**kwargs)
            assert result == result  # not NaN
            # The last case is a tiny power: use an absolute tolerance
            assert np.isclose(result, value, rtol=1e-5, atol=1e-7)

    def test_looks_one_is_identical(self):
        """looks=[1] must be bit-for-bit identical to looks=None, in every solve mode."""
        cases = [
            dict(d=0.5, n=20, contrast="one-sample"),
            dict(d=0.5, n=20),
            dict(d=0.5, power=0.80, alternative="greater"),
            dict(d=0.5, power=0.80),
            dict(n=20, power=0.80, contrast="paired"),
            dict(n=20, power=0.80),
            dict(d=0.5, n=20, power=0.80, alpha=None),
            dict(d=-0.5, n=20, alternative="less"),
            dict(d=0.5, n=20, contrast="one-sample", alternative="greater"),
        ]
        for kwargs in cases:
            fixed = power_ttest(**kwargs)
            single = power_ttest(looks=[1], **kwargs)
            assert fixed == single, kwargs

        cases2n = [
            dict(nx=20, ny=15, d=0.5, alternative="greater"),
            dict(nx=20, ny=15, power=0.80),
            dict(nx=20, ny=15, d=0.5, power=0.80, alpha=None),
        ]
        for kwargs in cases2n:
            fixed = power_ttest2n(**kwargs)
            single = power_ttest2n(looks=[1], **kwargs)
            assert fixed == single, kwargs

    def test_errors_looks_none_unchanged(self):
        """Validation errors of the fixed-sample path must be unchanged."""
        with pytest.raises(ValueError):
            power_ttest(d=0.5)
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=20, looks=[0.5, 1.5])
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=20, looks=[0.5, 0.8])
        with pytest.raises(ValueError):
            power_ttest2n(nx=20, ny=20)

    def test_power_equals_marginal_bell_power(self):
        """At each look, the marginal rejection probability equals the fixed-sample power."""
        bounds = sequential_bounds([0.25, 0.5, 0.75, 1.0], alpha=0.05, alternative="two-sided")
        nominal = bounds["nominal_p"].to_numpy()
        d, n, tsample = 0.4, 120, 2
        marginal = np.array(
            [
                _fixed_power(d, n * tk, tsample, nominal[k], "two-sided")
                for k, tk in enumerate(bounds["t"])
            ]
        )
        from pingouin.sequential import sequential_power

        # Sequential power must be a probability between the max marginal and their sum.
        power = sequential_power(
            bounds["t"].to_numpy(), bounds["bound"].to_numpy(), marginal, "two-sided"
        )
        assert marginal.max() <= power <= min(1.0, marginal.sum()) + 1e-12

    def test_sequential_power_reference(self):
        """Sequential power must match an independent MVN reference (absolute error < 1e-6)."""
        designs = [
            (0.4, 120, [0.25, 0.5, 0.75, 1.0], 0.05, "two-sided", "obf", 2),
            (0.5, 60, [0.2, 0.4, 0.6, 0.8, 1.0], 0.025, "greater", "pocock", 1),
            (-0.4, 80, [0.25, 0.5, 1.0], 0.025, "less", "obf", 1),
            (0.3, 100, [0.5, 1.0], 0.05, "two-sided", 0.5, 2),
        ]
        for d, n, looks, alpha, alternative, spending, tsample in designs:
            contrast = "two-samples" if tsample == 2 else "one-sample"
            computed = power_ttest(
                d=d,
                n=n,
                contrast=contrast,
                alpha=alpha,
                alternative=alternative,
                looks=looks,
                spending=spending,
            )
            reference = _sequential_power_reference(
                d, n, looks, tsample, alpha, alternative, spending
            )
            assert abs(computed - reference) < 1e-6, (d, n, looks, alternative, spending)

    def test_power_ttest2n_sequential_reference(self):
        designs = [
            (60, 50, 0.4, [0.25, 0.5, 0.75, 1.0], 0.05, "two-sided", "obf"),
            (30, 60, 0.5, [0.5, 1.0], 0.025, "greater", "pocock"),
        ]
        for nx, ny, d, looks, alpha, alternative, spending in designs:
            computed = power_ttest2n(
                nx=nx,
                ny=ny,
                d=d,
                alpha=alpha,
                alternative=alternative,
                looks=looks,
                spending=spending,
            )
            # Reference: fixed-sample power at every look, MVN integration.
            bounds = sequential_bounds(
                looks, alpha=alpha, alternative=alternative, spending=spending
            )
            nominal = bounds["nominal_p"].to_numpy()
            t = np.asarray(looks, float)
            marginal = np.array(
                [
                    _fixed_power(d, (nx + ny) * tk / 2, 2, nominal[k], alternative)
                    if nx == ny
                    else _nct_two_sample(d, nx * tk, ny * tk, nominal[k], alternative)
                    for k, tk in enumerate(t)
                ]
            )
            reference = _mvn_power_from_marginal(
                t, bounds["bound"].to_numpy(), marginal, alternative
            )
            assert abs(computed - reference) < 1e-6

    def test_solve_round_trip(self):
        """Solved n, d and alpha must reproduce the requested power (< 1e-6)."""
        looks = [0.2, 0.4, 0.6, 0.8, 1.0]
        target = 0.8

        n = power_ttest(d=0.4, power=target, contrast="one-sample", looks=looks)
        assert abs(power_ttest(d=0.4, n=n, contrast="one-sample", looks=looks) - target) < 1e-6
        d = power_ttest(n=60, power=target, contrast="one-sample", looks=looks)
        assert abs(power_ttest(d=d, n=60, contrast="one-sample", looks=looks) - target) < 1e-6
        alpha = power_ttest(
            d=0.4, n=60, power=target, alpha=None, contrast="one-sample", looks=looks
        )
        assert 0 < alpha < 0.5
        assert (
            abs(power_ttest(d=0.4, n=60, alpha=alpha, contrast="one-sample", looks=looks) - target)
            < 1e-6
        )

        # Two-sample unequal sizes
        d2 = power_ttest2n(nx=60, ny=50, power=target, looks=[0.25, 0.5, 1.0])
        assert abs(power_ttest2n(nx=60, ny=50, d=d2, looks=[0.25, 0.5, 1.0]) - target) < 1e-6
        a2 = power_ttest2n(nx=60, ny=50, d=0.4, power=target, alpha=None, looks=[0.25, 0.5, 1.0])
        assert (
            abs(power_ttest2n(nx=60, ny=50, d=0.4, alpha=a2, looks=[0.25, 0.5, 1.0]) - target)
            < 1e-6
        )

    def test_alpha_solve_stays_below_half(self):
        alpha = power_ttest(
            d=0.2, n=500, power=0.5, alpha=None, contrast="one-sample", looks=[0.5, 1.0]
        )
        assert 0 < alpha < 0.5
        alpha = power_ttest2n(nx=200, ny=200, d=0.2, power=0.5, alpha=None, looks=[0.5, 1.0])
        assert 0 < alpha < 0.5

    def test_first_look_dof_error(self):
        # First look has zero or negative degrees of freedom
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=4, contrast="one-sample", looks=[0.25, 1.0])
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=4, contrast="two-samples", looks=[0.25, 1.0])
        with pytest.raises(ValueError):
            power_ttest2n(nx=2, ny=2, d=0.5, looks=[0.5, 1.0])
        # Solving for n only searches in the valid range (does not raise)
        n = power_ttest(d=0.5, power=0.8, contrast="one-sample", looks=[0.25, 1.0])
        assert (n * 0.25 - 1) > 0
        assert abs(power_ttest(d=0.5, n=n, contrast="one-sample", looks=[0.25, 1.0]) - 0.8) < 1e-6

    def test_validation(self):
        looks = [0.5, 1.0]
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=20, looks=looks, spending="bad")
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=20, looks=looks, alpha=0.6)
        with pytest.raises(ValueError):
            power_ttest(d=0.5, n=20, looks=[0.5, 1.5])
        with pytest.raises(ValueError):
            power_ttest2n(nx=20, ny=20, d=0.5, looks=[0.5, 1.5])
        with pytest.raises(ValueError):
            power_ttest2n(nx=20, ny=20, d=0.5, looks=looks, spending=-2.0)

    def test_speed(self):
        import time

        looks = [0.2, 0.4, 0.6, 0.8, 1.0]
        start = time.time()
        power_ttest(d=0.4, n=120, looks=looks)
        assert time.time() - start < 0.5
        start = time.time()
        power_ttest(d=0.4, power=0.8, looks=looks)
        assert time.time() - start < 5.0
        start = time.time()
        power_ttest(n=120, power=0.8, looks=looks)
        assert time.time() - start < 5.0
        start = time.time()
        power_ttest(d=0.4, n=120, power=0.8, alpha=None, looks=looks)
        assert time.time() - start < 5.0


def _nct_two_sample(d, nx, ny, alpha, alternative):
    """Fixed-sample power of an independent two-sample T-test with unequal sizes."""
    nc = d / np.sqrt(1 / nx + 1 / ny)
    dof = nx + ny - 2
    if alternative == "greater":
        crit = stats.t.ppf(1 - alpha, dof)
        return stats.nct.sf(crit, dof, nc)
    if alternative == "less":
        crit = stats.t.ppf(alpha, dof)
        return 1 - stats.nct.sf(crit, dof, nc)
    crit = stats.t.ppf(1 - alpha / 2, dof)
    return stats.nct.sf(crit, dof, nc) + 1 - stats.nct.sf(-crit, dof, nc)


def _mvn_power_from_marginal(t, bound, marginal, alternative):
    """Sequential power from marginal rejection probabilities, using scipy's MVN CDF."""
    marginal = np.clip(marginal, 1e-15, 1 - 1e-16)
    shift = np.zeros_like(marginal)
    for k in range(len(t)):
        if alternative == "greater":
            shift[k] = bound[k] + stats.norm.ppf(marginal[k])
        elif alternative == "less":
            shift[k] = bound[k] - stats.norm.ppf(marginal[k])
        else:
            lo, hi = 0.0, 60.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                prob = stats.norm.cdf(-bound[k] - mid) + stats.norm.sf(bound[k] - mid)
                if prob < marginal[k]:
                    lo = mid
                else:
                    hi = mid
            shift[k] = 0.5 * (lo + hi)
    if alternative == "greater":
        lower, upper = np.full(len(t), -np.inf), bound
    elif alternative == "less":
        lower, upper = bound, np.full(len(t), np.inf)
    else:
        lower, upper = -bound, bound
    cdf = stats.multivariate_normal.cdf(
        upper - shift,
        mean=np.zeros(len(t)),
        cov=_corr(t),
        lower_limit=lower - shift,
        maxpts=20_000_000,
        abseps=1e-13,
        releps=1e-13,
    )
    return 1 - cdf


# --------------------------------------------------------------------------------------------
# sequential_ttest
# --------------------------------------------------------------------------------------------


class TestSequentialTtest:
    def test_single_look_identical_to_ttest(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=30)
        y = rng.normal(loc=0.5, size=30)
        for paired in [False, True]:
            if paired:
                y_ = x + rng.normal(loc=0.4, size=30)
            else:
                y_ = y
            res = sequential_ttest(x, y_, looks=[(30, 30)] if not paired else [30], paired=paired)
            ref = ttest(x, y_, paired=paired)
            assert res["T"].iloc[0] == ref["T"].iloc[0]
            assert res["dof"].iloc[0] == ref["dof"].iloc[0]
            assert res["p_unc"].iloc[0] == ref["p_val"].iloc[0]
            assert res["nominal_p"].iloc[0] == 0.05
            assert res["reject"].iloc[0] == bool(ref["p_val"].iloc[0] <= 0.05)

    def test_ttest_agreement_at_each_look(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=50)
        y = rng.normal(loc=0.6, size=50)
        looks = [(10, 10), (20, 25), (35, 40), (50, 50)]
        # Use a very small alpha so that the test does not stop early (all looks are reported)
        res = sequential_ttest(x, y, looks=looks, alpha=1e-10)
        bounds = sequential_bounds(_info_from_pairs(looks), alpha=1e-10)
        assert len(res) == len(looks)
        assert not res["reject"].any()
        for k, (nx, ny) in enumerate(looks):
            ref = ttest(x[:nx], y[:ny])
            assert res["T"].iloc[k] == ref["T"].iloc[0]
            assert res["dof"].iloc[k] == ref["dof"].iloc[0]
            assert res["p_unc"].iloc[k] == ref["p_val"].iloc[0]
            assert res["nominal_p"].iloc[k] == bounds["nominal_p"].iloc[k]
        assert list(res.columns) == ["look", "nx", "ny", "T", "dof", "p_unc", "nominal_p", "reject"]

    def test_nominal_matches_bounds_paired(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=40)
        y = x + rng.normal(loc=0.2, size=40)
        looks = [10, 20, 30, 40]
        res = sequential_ttest(
            x, y, looks=looks, paired=True, alternative="greater", alpha=0.025, spending="pocock"
        )
        bounds = sequential_bounds(
            [0.25, 0.5, 0.75, 1.0], alpha=0.025, alternative="greater", spending="pocock"
        )
        assert np.array_equal(
            res["nominal_p"].to_numpy(), bounds["nominal_p"].to_numpy()[: len(res)]
        )

    def test_stopping(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=40)
        y = x + rng.normal(loc=1.0, size=40)
        res = sequential_ttest(x, y, looks=[10, 20, 30, 40], paired=True)
        assert res["reject"].iloc[-1] is True or res["reject"].iloc[-1] == True  # noqa: E712
        assert res["reject"].sum() == 1
        # No analysis after the rejection
        assert res["look"].iloc[-1] == res["reject"].idxmax() + 1

    def test_information_fraction_independent(self):
        # Unequal sizes: t_k = (nx ny / (nx + ny)) / (nX nY / (nX + nY))
        looks = [(10, 20), (20, 30), (40, 60)]
        t = _info_from_pairs(looks)
        assert np.allclose(
            t, [(10 * 20 / 30) / (40 * 60 / 100), (20 * 30 / 50) / (40 * 60 / 100), 1.0]
        )

    def test_validation(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=30)
        y = rng.normal(size=30)
        with pytest.raises(ValueError):
            sequential_ttest(np.r_[x, np.nan], y, looks=[5, 10], paired=True)
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[20, 40], paired=True)  # too many
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[10, 10], paired=True)  # not strictly increasing
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[1, 2], paired=True)  # too few pairs
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(10, 10), (10, 10)])  # no increase
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(10, 10), (5, 20)])  # decrease
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(10, 10), (40, 10)])  # too many
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(1, 10), (10, 10)])  # too few obs
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(10, 10), (20, 20)], alpha=0.5)
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(10, 10), (20, 20)], spending="bad")
        with pytest.raises(ValueError):
            sequential_ttest(x, y, looks=[(10, 10), (20, 20)], alternative="both")

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=40)
        y = rng.normal(loc=0.3, size=40)
        a = sequential_ttest(x, y, looks=[(10, 10), (20, 20), (40, 40)], alpha=0.001)
        b = sequential_ttest(x, y, looks=[(10, 10), (20, 20), (40, 40)], alpha=0.001)
        pd.testing.assert_frame_equal(a, b)

    def test_simulation_power_matches(self):
        """Rejection rate of sequential_ttest matches the sequential power within MC error."""
        rng = np.random.default_rng(2024)
        n = 60
        looks = [15, 30, 45, 60]
        d = 0.5
        nrep = 1500
        rejections = 0
        for _ in range(nrep):
            x = rng.normal(size=n)
            y = x + rng.normal(loc=d, size=n)  # paired design
            res = sequential_ttest(x, y, looks=looks, paired=True)
            rejections += int(res["reject"].iloc[-1])
        rate = rejections / nrep
        expected = power_ttest(d=d, n=n, contrast="paired", looks=[0.25, 0.5, 0.75, 1.0])
        se = np.sqrt(expected * (1 - expected) / nrep)
        assert abs(rate - expected) < 4 * se

    def test_simulation_null_type_one(self):
        """Under the null, the rejection rate is close to alpha."""
        rng = np.random.default_rng(4321)
        n = 40
        looks = [10, 20, 30, 40]
        alpha = 0.05
        nrep = 1200
        rejections = 0
        for _ in range(nrep):
            x = rng.normal(size=n)
            y = x + rng.normal(size=n)  # null paired design
            res = sequential_ttest(x, y, looks=looks, paired=True, alpha=alpha)
            rejections += int(res["reject"].iloc[-1])
        rate = rejections / nrep
        se = np.sqrt(alpha * (1 - alpha) / nrep)
        assert abs(rate - alpha) < 4 * se

    def test_naive_alpha_inflation(self):
        """Sanity check motivating the feature: naive repeated t-tests inflate type-I error."""
        rng = np.random.default_rng(99)
        n = 100
        looks = [20, 40, 60, 80, 100]
        nrep = 600
        naive = 0
        sequential = 0
        for _ in range(nrep):
            x = rng.normal(size=n)
            y = x + rng.normal(size=n)  # null
            naive += int(
                any(ttest(x[:k], y[:k], paired=True)["p_val"].iloc[0] <= 0.05 for k in looks)
            )
            res = sequential_ttest(x, y, looks=looks, paired=True)
            sequential += int(res["reject"].iloc[-1])
        assert naive / nrep > 0.10
        assert abs(sequential / nrep - 0.05) < 0.03


def _info_from_pairs(looks):
    """Information fractions of a sequence of (nx, ny) pairs."""
    nx = np.array([p[0] for p in looks], float)
    ny = np.array([p[1] for p in looks], float)
    info = nx * ny / (nx + ny)
    return info / info[-1]


# --------------------------------------------------------------------------------------------
# Export / integration
# --------------------------------------------------------------------------------------------


def test_public_exports():
    import pingouin as pg

    assert hasattr(pg, "sequential_bounds")
    assert hasattr(pg, "sequential_ttest")
    assert "sequential_bounds" in pg.power.__all__
    assert "sequential_ttest" in pg.parametric.__all__
