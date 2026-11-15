# Author: Raphael Vallat <raphaelvallat9@gmail.com>
# Date: April 2018
import warnings
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.optimize import brenth

from . import sequential as _seq
from .utils import _check_alternative

__all__ = [
    "power_ttest",
    "power_ttest2n",
    "power_anova",
    "power_rm_anova",
    "power_corr",
    "power_chi2",
    "sequential_bounds",
]

# Bracketing interval of the Cohen d when solving for the effect size of a T-test
_D_BOUNDS = {"two-sided": (1e-07, 10), "less": (-10, 5), "greater": (-5, 10)}


def _solve(func, lower, upper):
    """Find the root of ``func`` in [lower, upper] with Brent's method, or NaN if it fails."""
    try:
        return brenth(func, lower, upper)
    except ValueError:  # pragma: no cover
        return np.nan


def _ncf_sf(fcrit, dof1, dof2, nc):
    """Survival function of the non-central F distribution, i.e. the power of an F-test."""
    return stats.ncf.sf(fcrit, dof1, dof2, nc)


def _power_nct(nc, dof, alpha, alternative):
    """Power of a T-test, given the non-centrality parameter and degrees of freedom."""
    if alternative == "less":
        tcrit = stats.t.ppf(alpha, dof)
        return 1 - stats.nct.sf(tcrit, dof, nc)
    elif alternative == "two-sided":
        tcrit = stats.t.ppf(1 - alpha / 2, dof)
        return stats.nct.sf(tcrit, dof, nc) + (1 - stats.nct.sf(-tcrit, dof, nc))
    else:  # alternative = "greater"
        tcrit = stats.t.ppf(1 - alpha, dof)
        return stats.nct.sf(tcrit, dof, nc)


@lru_cache(maxsize=32)
def _seq_bounds_table(t, alpha, alternative, spending):
    """Cached sequential boundaries, nominal p-values and cumulative spending.

    ``t`` is a hashable tuple of floats. The returned arrays must not be mutated by callers.
    """
    return _seq.sequential_bound_table(np.asarray(t, float), alpha, alternative, spending)


def _seq_bounds_cached(t, alpha, alternative, spending):
    """Copies of the cached sequential boundaries, nominal p-values and cumulative spending."""
    bound, nominal, spent = _seq_bounds_table(t, float(alpha), alternative, spending)
    return bound.copy(), nominal.copy(), spent.copy()


def _sequential_power_ttest(d, n, t, tsample, alpha, alternative, spending):
    """Power of a group-sequential T-test with equal (or one-sample) group sizes.

    ``n`` is the sample size of the final analysis (number of pairs or number of observations in
    each group for a two-sample test); the sample size of the ``k``-th look is ``n * t[k]``. The
    marginal rejection probability of each look is the fixed-sample non-central T power at that
    look and at the nominal level of the look; the dependence between looks follows the joint
    normal distribution of the interim statistics, with correlation :math:`\\sqrt{t_j / t_k}`.
    """
    t = np.asarray(t, float)
    dof = (n * t - 1) * tsample
    if dof[0] <= 0:
        raise ValueError(
            "The degrees of freedom of the first analysis must be positive. Increase the sample "
            "size or the first information fraction."
        )
    bound, nominal, _ = _seq_bounds_cached(tuple(t), float(alpha), alternative, spending)
    marginal = np.array(
        [
            _power_nct(d * np.sqrt(nk / tsample), df, nom, alternative)
            for nk, df, nom in zip(n * t, dof, nominal)
        ]
    )
    return _seq.sequential_power(t, bound, marginal, alternative)


def _sequential_power_ttest2n(nx, ny, d, t, alpha, alternative, spending):
    """Power of a group-sequential two-sample T-test with unequal group sizes."""
    t = np.asarray(t, float)
    dof = (nx + ny) * t - 2
    if dof[0] <= 0:
        raise ValueError(
            "The degrees of freedom of the first analysis must be positive. Increase the sample "
            "sizes or the first information fraction."
        )
    bound, nominal, _ = _seq_bounds_cached(tuple(t), float(alpha), alternative, spending)
    marginal = np.array(
        [
            _power_nct(d / np.sqrt(1 / (nx * tk) + 1 / (ny * tk)), df, nom, alternative)
            for tk, df, nom in zip(t, dof, nominal)
        ]
    )
    return _seq.sequential_power(t, bound, marginal, alternative)


def sequential_bounds(looks, alpha=0.05, alternative="two-sided", spending="obf"):
    """
    Group-sequential efficacy boundaries and nominal p-values.

    In a group-sequential design, the data are analyzed several times and the trial may stop as
    soon as a boundary is crossed. Using the same nominal significance level at every look
    inflates the overall type-I error rate well above ``alpha``. This function computes the
    efficacy boundaries that spend ``alpha`` over the planned analyses, according to a spending
    function.

    Parameters
    ----------
    looks : sequence of float
        Information fractions of the successive analyses, i.e. the proportion of the maximum
        planned information available at each look. Must be a one-dimensional, strictly
        increasing sequence of finite numbers in (0, 1]. The last value is typically 1.
    alpha : float
        Overall significance level (type-I error probability), in (0, 0.5). For a two-sided
        alternative, ``alpha`` is the total error rate of both tails.
    alternative : str
        Defines the alternative hypothesis, or tail of the test. Must be one of "two-sided"
        (default), "greater" or "less".
    spending : str or float
        Type-I error spending function:

        * ``"obf"``: O'Brien-Fleming-like spending,
          :math:`\\alpha(t) = 2 (1 - \\Phi(\\Phi^{-1}(1 - \\alpha / 2) / \\sqrt{t}))`
        * ``"pocock"``: Pocock-like spending,
          :math:`\\alpha(t) = \\alpha \\ln(1 + (e - 1) t)`
        * a positive, finite float :math:`\\rho`: power-family spending,
          :math:`\\alpha(t) = \\alpha t^{\\rho}`

        where :math:`t` is the information fraction and :math:`\\Phi` the standard normal CDF.

    Returns
    -------
    bounds : :py:class:`pandas.DataFrame`
        One row per analysis, with the columns:

        * ``'look'``: analysis number, starting at 1
        * ``'t'``: information fraction
        * ``'bound'``: efficacy boundary on the standardized Z scale. For ``"greater"`` the null
          is rejected when :math:`Z \\geq` ``bound``; for ``"less"`` when :math:`Z \\leq`
          ``bound`` (the boundary is negative); for ``"two-sided"`` when
          :math:`|Z| \\geq` ``bound`` (symmetric boundaries).
        * ``'nominal_p'``: nominal p-value of that look (one-sided for "greater"/"less",
          two-sided for "two-sided").
        * ``'alpha_look'``: type-I error probability spent at that look (increment).
        * ``'alpha_spent'``: cumulative type-I error probability spent up to that look.

    Notes
    -----
    Under the null hypothesis, the interim statistics :math:`Z_1, \\ldots, Z_K` follow a joint
    standard normal distribution whose correlation between looks :math:`j` and :math:`k` is
    :math:`\\sqrt{t_j / t_k}`. The boundaries are defined so that the probability of crossing for
    the first time at look :math:`k` equals :math:`\\alpha(t_k) - \\alpha(t_{k-1})`, with
    :math:`t_0 = 0`.

    The calculation is exact up to numerical quadrature. The boundaries of a given look only
    depend on the analyses up to that look: appending further analyses does not change the
    boundaries of the previous ones.

    The same boundaries (and nominal p-values) are used by :py:func:`power_ttest`,
    :py:func:`power_ttest2n` and :py:func:`sequential_ttest`, which guarantees that a design
    planned with this function has the intended type-I error rate when analyzed with
    :py:func:`sequential_ttest`.

    When only one analysis is provided, the design reduces to a fixed-sample test: ``nominal_p``,
    ``alpha_look`` and ``alpha_spent`` are all equal to ``alpha``.

    Examples
    --------
    O'Brien-Fleming boundaries for five equally spaced analyses (these values match the published
    ones, e.g. Jennison & Turnbull 2000)::

        >>> from pingouin import sequential_bounds
        >>> bounds = sequential_bounds([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.025,
        ...                            alternative="greater", spending="obf")
        >>> bounds.round(3)
           look    t  bound  nominal_p  alpha_look  alpha_spent
        0     1  0.2  4.877      0.000       0.000        0.000
        1     2  0.4  3.357      0.000       0.000        0.000
        2     3  0.6  2.680      0.004       0.003        0.004
        3     4  0.8  2.290      0.011       0.008        0.012
        4     5  1.0  2.031      0.021       0.013        0.025

    Pocock boundaries for the same design::

        >>> sequential_bounds([0.2, 0.4, 0.6, 0.8, 1.0], alpha=0.025,
        ...                   alternative="greater", spending="pocock").round(3)
           look    t  bound  nominal_p  alpha_look  alpha_spent
        0     1  0.2  2.438      0.007       0.007        0.007
        1     2  0.4  2.427      0.008       0.006        0.013
        2     3  0.6  2.410      0.008       0.005        0.018
        3     4  0.8  2.397      0.008       0.004        0.022
        4     5  1.0  2.386      0.009       0.003        0.025

    See also
    --------
    power_ttest, power_ttest2n, sequential_ttest

    References
    ----------
    * Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with Applications to
      Clinical Trials. Chapman & Hall/CRC.

    * Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
      Biometrika, 70(3), 659-663.
    """
    import pandas as pd

    t = _seq._check_looks(looks)
    alpha = _seq._check_alpha(alpha)
    _seq._check_alternative(alternative)
    spending = _seq._check_spending(spending)

    # Use the same table as power_ttest / power_ttest2n / sequential_ttest (single source of truth).
    bound, nominal, spent = _seq_bounds_cached(
        tuple(float(v) for v in t), float(alpha), alternative, spending
    )

    alpha_look = np.diff(np.concatenate([[0.0], spent]))
    return pd.DataFrame(
        {
            "look": np.arange(1, t.size + 1),
            "t": t,
            "bound": bound,
            "nominal_p": nominal,
            "alpha_look": alpha_look,
            "alpha_spent": spent,
        }
    )


def power_ttest(
    d=None,
    n=None,
    power=None,
    alpha=0.05,
    contrast="two-samples",
    alternative="two-sided",
    looks=None,
    spending="obf",
):
    """
    Evaluate power, sample size, effect size or significance level of a one-sample T-test,
    a paired T-test or an independent two-samples T-test with equal sample sizes.

    Parameters
    ----------
    d : float
        Cohen d effect size
    n : int
        Sample size
        In case of a two-sample T-test, sample sizes are assumed to be equal.
        Otherwise, see the :py:func:`power_ttest2n` function.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.
    contrast : str
        Can be `"one-sample"`, `"two-samples"` or `"paired"`.
        Note that `"one-sample"` and `"paired"` have the same behavior.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less".
    looks : sequence of float, optional
        Information fractions of a group-sequential design (interim analyses), e.g.
        ``[0.25, 0.5, 0.75, 1.0]``. If None (default), a fixed-sample design is used and the
        function behaves exactly as before. If provided, must be a one-dimensional, strictly
        increasing sequence of finite numbers in (0, 1] whose **last value is exactly 1**; the
        sequential power is then computed (see Notes).
    spending : str or float
        Type-I error spending function used when ``looks`` is not None (ignored otherwise). Must
        be "obf" (default), "pocock", or a positive finite float ``rho``. See
        :py:func:`sequential_bounds` for the formulas.

    Notes
    -----
    Exactly ONE of the parameters ``d``, ``n``, ``power`` and ``alpha`` must be passed as None, and
    that parameter is determined from the others.

    For a paired T-test, the sample size ``n`` corresponds to the number of pairs. For an
    independent two-sample T-test with equal sample sizes, ``n`` corresponds to the sample size of
    each group (i.e. number of observations in one group). If the sample sizes are unequal, please
    use the :py:func:`power_ttest2n` function instead.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    This function is a Python adaptation of the `pwr.t.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    The first step is to use the Cohen's d to calculate the non-centrality parameter
    :math:`\\delta` and degrees of freedom :math:`v`. In case of paired groups, this is:

    .. math:: \\delta = d * \\sqrt n
    .. math:: v = n - 1

    and in case of independent groups with equal sample sizes:

    .. math:: \\delta = d * \\sqrt{\\frac{n}{2}}
    .. math:: v = (n - 1) * 2

    where :math:`d` is the Cohen d and :math:`n` the sample size.

    The critical value is then found using the percent point function of the T distribution with
    :math:`q = 1 - alpha` and :math:`v` degrees of freedom.

    Finally, the power of the test is given by the survival function of the non-central
    distribution using the previously calculated critical value, degrees of freedom and
    non-centrality parameter.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    **Group-sequential designs.** When ``looks`` is provided, ``n`` still refers to the sample size
    of the *final* analysis (or to the sample size of each group for a two-sample test). The
    sample size of the ``k``-th analysis is ``n * looks[k]`` (which may be non-integer). The
    efficacy boundaries and nominal p-values are obtained from :py:func:`sequential_bounds` with
    the same ``alpha``, ``alternative`` and ``spending``. The test performed at the ``k``-th
    analysis is the usual fixed-sample T-test at that look, declared significant when its p-value
    is at most the nominal p-value of the look. The sequential power is the probability of
    rejecting the null hypothesis at *any* look. It is computed by combining:

    * the marginal rejection probability of each look, which is exactly the fixed-sample T-test
      power (non-central T distribution, same degrees of freedom and ``alternative``) evaluated at
      the nominal level of that look and at the sample size of that look, and
    * the dependence between looks, via the joint normal distribution of the standardized
      statistics with correlation :math:`\\sqrt{t_j / t_k}`.

    For a two-sided alternative, rejection in either tail counts and ``d`` is taken in absolute
    value, as in the fixed-sample case. With a single look (``looks=[1]``), the result is identical
    to the fixed-sample calculation (``looks=None``).

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute power of a one-sample T-test given ``d``, ``n`` and ``alpha``

    >>> from pingouin import power_ttest
    >>> print("power: %.4f" % power_ttest(d=0.5, n=20, contrast="one-sample"))
    power: 0.5645

    2. Compute required sample size given ``d``, ``power`` and ``alpha``

    >>> print("n: %.4f" % power_ttest(d=0.5, power=0.80, alternative="greater"))
    n: 50.1508

    3. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level

    >>> print("d: %.4f" % power_ttest(n=20, power=0.80, alpha=0.05, contrast="paired"))
    d: 0.6604

    4. Compute achieved alpha level given ``d``, ``n`` and ``power``

    >>> print("alpha: %.4f" % power_ttest(d=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.4430

    5. One-sided tests

    >>> from pingouin import power_ttest
    >>> print("power: %.4f" % power_ttest(d=0.5, n=20, alternative="greater"))
    power: 0.4634

    >>> print("power: %.4f" % power_ttest(d=0.5, n=20, alternative="less"))
    power: 0.0007
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, n, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of n, d, power, and alpha must be None.")

    # Safety checks
    _check_alternative(alternative)
    assert contrast.lower() in ["one-sample", "paired", "two-samples"]
    tsample = 2 if contrast.lower() == "two-samples" else 1
    if d is not None and alternative == "two-sided":
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    t = None
    if looks is not None:
        # Group-sequential design: validate the design before doing anything else.
        t = _seq._check_looks(looks, require_last_one=True)
        spending = _seq._check_spending(spending)
        if alpha is not None:
            alpha = _seq._check_alpha(alpha)
        if t.size == 1:
            # A single look is a fixed-sample design: use the exact same code path, so that the
            # results are bit-for-bit identical to ``looks=None``.
            t = None
        elif n is not None and (n * t[0] - 1) * tsample <= 0:
            # Validate the first analysis degrees of freedom when the final size is known.
            raise ValueError(
                "The degrees of freedom of the first analysis must be positive. Increase the "
                "sample size or the first information fraction."
            )

    def func(d, n, alpha):
        if t is None:
            return _power_nct(d * np.sqrt(n / tsample), (n - 1) * tsample, alpha, alternative)
        return _sequential_power_ttest(d, n, t, tsample, alpha, alternative, spending)

    def check_dof(n):
        """Raise if the first analysis of a sequential design has non-positive degrees of freedom."""
        if t is not None and (n * t[0] - 1) * tsample <= 0:
            raise ValueError(
                "The degrees of freedom of the first analysis must be positive. Increase the "
                "sample size or the first information fraction."
            )

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        check_dof(n)
        return func(d, n, alpha)
    elif n is None:
        # Compute required sample size given d, power and alpha
        if t is None:
            return _solve(lambda n: func(d, n, alpha) - power, 2 + 1e-10, 1e07)
        # Search only in the range where the first analysis has positive degrees of freedom.
        # The lower bound is chosen so that the first analysis has at least one degree of freedom
        # (well above the degenerate dof -> 0 regime) and at least two observations.
        n_lo = max(2 + 1e-10, (2 + 1e-10) / t[0])
        return _solve(lambda n: func(d, n, alpha) - power, n_lo, 1e07)
    elif d is None:
        # Compute achieved d given sample size, power and alpha level
        check_dof(n)
        return _solve(lambda d: func(d, n, alpha) - power, *_D_BOUNDS[alternative])
    else:
        # Compute achieved alpha (significance) level given d, n and power
        if t is None:
            return _solve(lambda alpha: func(d, n, alpha) - power, 1e-10, 1 - 1e-10)
        check_dof(n)
        # For a sequential design, alpha must lie in (0, 0.5).
        return _solve(lambda alpha: func(d, n, alpha) - power, 1e-10, 0.5 - 1e-10)


def power_ttest2n(
    nx, ny, d=None, power=None, alpha=0.05, alternative="two-sided", looks=None, spending="obf"
):
    """
    Evaluate power, effect size or  significance level of an independent two-samples T-test
    with unequal sample sizes.

    Parameters
    ----------
    nx, ny : int
        Sample sizes. Must be specified. If the sample sizes are equal, you should use the
        :py:func:`power_ttest` function instead.
    d : float
        Cohen d effect size
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability). The default is 0.05.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of "two-sided"
        (default), "greater" or "less".
    looks : sequence of float, optional
        Information fractions of a group-sequential design (interim analyses), e.g.
        ``[0.25, 0.5, 0.75, 1.0]``. If None (default), a fixed-sample design is used and the
        function behaves exactly as before. If provided, must be a one-dimensional, strictly
        increasing sequence of finite numbers in (0, 1] whose **last value is exactly 1**; the
        sequential power is then computed (see Notes).
    spending : str or float
        Type-I error spending function used when ``looks`` is not None (ignored otherwise). Must
        be "obf" (default), "pocock", or a positive finite float ``rho``. See
        :py:func:`sequential_bounds` for the formulas.

    Notes
    -----
    Exactly ONE of the parameters ``d``, ``power`` and ``alpha`` must be passed as None, and that
    parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to compute
    it.

    This function is a Python adaptation of the `pwr.t2n.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    The first step is to use the Cohen's d to calculate the non-centrality parameter
    :math:`\\delta` and degrees of freedom :math:`v`.cIn case of two independent groups with
    unequal sample sizes, this is:

    .. math:: \\delta = d * \\sqrt{\\frac{n_i * n_j}{n_i + n_j}}
    .. math:: v = n_i + n_j - 2

    where :math:`d` is the Cohen d, :math:`n` the sample size,
    :math:`n_i` the sample size of the first group and
    :math:`n_j` the sample size of the second group,

    The critical value is then found using the percent point function of the T distribution with
    :math:`q = 1 - alpha` and :math:`v` degrees of freedom.

    Finally, the power of the test is given by the survival function of the non-central
    distribution using the previously calculated critical value, degrees of freedom and
    non-centrality parameter.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    **Group-sequential designs.** When ``looks`` is provided, ``nx`` and ``ny`` still refer to the
    sample sizes of the *final* analysis. The sample sizes of the ``k``-th analysis are
    ``nx * looks[k]`` and ``ny * looks[k]``. The efficacy boundaries and nominal p-values come
    from :py:func:`sequential_bounds` with the same ``alpha``, ``alternative`` and ``spending``;
    each analysis is the usual fixed-sample T-test at that look, declared significant when its
    p-value is at most the nominal p-value of the look. The sequential power is the probability of
    rejecting the null hypothesis at *any* look. Its marginal probabilities are the fixed-sample
    non-central T powers at each look (see :py:func:`power_ttest`), combined through the joint
    normal distribution of the interim statistics. With a single look (``looks=[1]``), the result
    is identical to the fixed-sample calculation (``looks=None``).

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------

    2. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level

    >>> print("d: %.4f" % power_ttest2n(nx=20, ny=15, power=0.80, alpha=0.05))
    d: 0.9859

    3. Compute achieved alpha level given ``d``, ``n`` and ``power``

    >>> print("alpha: %.4f" % power_ttest2n(nx=20, ny=15, d=0.5, power=0.80, alpha=None))
    alpha: 0.5000
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of d, power, and alpha must be None")

    # Safety checks
    _check_alternative(alternative)
    if d is not None and alternative == "two-sided":
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    t = None
    if looks is not None:
        # Group-sequential design: validate the design before doing anything else.
        t = _seq._check_looks(looks, require_last_one=True)
        spending = _seq._check_spending(spending)
        if alpha is not None:
            alpha = _seq._check_alpha(alpha)
        if t.size == 1:
            # A single look is a fixed-sample design: use the exact same code path, so that the
            # results are bit-for-bit identical to ``looks=None``.
            t = None
        elif (nx + ny) * t[0] - 2 <= 0:
            raise ValueError(
                "The degrees of freedom of the first analysis must be positive. Increase the "
                "sample sizes or the first information fraction."
            )

    def func(d, alpha):
        if t is None:
            return _power_nct(d * (1 / np.sqrt(1 / nx + 1 / ny)), nx + ny - 2, alpha, alternative)
        return _sequential_power_ttest2n(nx, ny, d, t, alpha, alternative, spending)

    def check_dof():
        """Raise if the first analysis has non-positive degrees of freedom."""
        if t is not None and (nx + ny) * t[0] - 2 <= 0:
            raise ValueError(
                "The degrees of freedom of the first analysis must be positive. Increase the "
                "sample sizes or the first information fraction."
            )

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, alpha)
    elif d is None:
        # Compute achieved d given sample size, power and alpha level
        check_dof()
        return _solve(lambda d: func(d, alpha) - power, *_D_BOUNDS[alternative])
    else:
        # Compute achieved alpha (significance) level given d, n and power
        if t is None:
            return _solve(lambda alpha: func(d, alpha) - power, 1e-10, 1 - 1e-10)
        check_dof()
        # For a sequential design, alpha must lie in (0, 0.5).
        return _solve(lambda alpha: func(d, alpha) - power, 1e-10, 0.5 - 1e-10)


def power_anova(eta_squared=None, k=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or significance level of a one-way balanced ANOVA.

    Parameters
    ----------
    eta_squared : float
        ANOVA effect size (eta-squared, :math:`\\eta^2`).
    k : int
        Number of groups
    n : int
        Sample size per group. Groups are assumed to be balanced (i.e. same sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level :math:`\\alpha` (type I error probability). The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters ``eta_squared``, ``k``, ``n``, ``power`` and ``alpha``
    must be passed as None, and that parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    This function is a Python adaptation of the `pwr.anova.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an
    effect there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    For one-way ANOVA, eta-squared is the same as partial eta-squared. It can be evaluated from the
    F-value (:math:`F^*`) and the degrees of freedom of the ANOVA (:math:`v_1, v_2`) using the
    following formula:

    .. math:: \\eta^2 = \\frac{v_1 F^*}{v_1 F^* + v_2}

    GPower uses the :math:`f` effect size instead of the :math:`\\eta^2`. The formula to convert
    from one to the other are given below:

    .. math:: f = \\sqrt{\\frac{\\eta^2}{1 - \\eta^2}}

    .. math:: \\eta^2 = \\frac{f^2}{1 + f^2}

    Using :math:`\\eta^2` and the total sample size :math:`N`, the non-centrality parameter is
    defined by:

    .. math:: \\delta = N * \\frac{\\eta^2}{1 - \\eta^2}

    Then the critical value of the non-central F-distribution is computed using the percentile
    point function of the F-distribution with:

    .. math:: q = 1 - \\alpha
    .. math:: v_1 = k - 1
    .. math:: v_2 = N - k

    where :math:`k` is the number of groups.

    Finally, the power of the ANOVA is calculated using the survival function of the non-central
    F-distribution using the previously computed critical value, non-centrality parameter, and
    degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_anova
    >>> print("power: %.4f" % power_anova(eta_squared=0.1, k=3, n=20))
    power: 0.6082

    2. Compute required number of groups

    >>> print("k: %.4f" % power_anova(eta_squared=0.1, n=20, power=0.80))
    k: 6.0944

    3. Compute required sample size

    >>> print("n: %.4f" % power_anova(eta_squared=0.1, k=3, power=0.80))
    n: 29.9256

    4. Compute achieved effect size

    >>> print("eta-squared: %.4f" % power_anova(n=20, k=4, power=0.80, alpha=0.05))
    eta-squared: 0.1255

    5. Compute achieved alpha (significance)

    >>> print("alpha: %.4f" % power_anova(eta_squared=0.1, n=20, k=4, power=0.80, alpha=None))
    alpha: 0.1085
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [eta_squared, k, n, power, alpha]])
    if n_none != 1:
        err = "Exactly one of eta, k, n, power, and alpha must be None."
        raise ValueError(err)

    # Safety checks
    if eta_squared is not None:
        eta_squared = abs(eta_squared)
        f_sq = eta_squared / (1 - eta_squared)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(f_sq, k, n, alpha):
        nc = (n * k) * f_sq
        dof1 = k - 1
        dof2 = (n * k) - k
        fcrit = stats.f.ppf(1 - alpha, dof1, dof2)
        return _ncf_sf(fcrit, dof1, dof2, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(f_sq, k, n, alpha)
    elif k is None:
        # Compute required number of groups
        return _solve(lambda k: func(f_sq, k, n, alpha) - power, 2, 100)
    elif n is None:
        # Compute required sample size
        return _solve(lambda n: func(f_sq, k, n, alpha) - power, 2, 1e07)
    elif eta_squared is None:
        # Compute achieved eta-squared. Solve directly on eta-squared in (0, 1): solving on
        # f_sq in (0, 1) would restrict eta-squared to be lower than 0.5.
        return _solve(lambda eta: func(eta / (1 - eta), k, n, alpha) - power, 1e-10, 1 - 1e-10)
    else:
        # Compute achieved alpha
        return _solve(lambda alpha: func(f_sq, k, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_rm_anova(eta_squared=None, m=None, n=None, power=None, alpha=0.05, corr=0.5, epsilon=1):
    """
    Evaluate power, sample size, effect size or significance level of a balanced one-way
    repeated measures ANOVA.

    Parameters
    ----------
    eta_squared : float
        ANOVA effect size (eta-squared, :math:`\\eta^2`).
    m : int
        Number of repeated measurements.
    n : int
        Sample size per measurement. All measurements must have the same sample size.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level :math:`\\alpha` (type I error probability). The default is 0.05.
    corr : float
        Average correlation coefficient among repeated measurements. The default is :math:`r=0.5`.
    epsilon : float
        Epsilon adjustement factor for sphericity. This can be calculated using the
        :py:func:`pingouin.epsilon` function.

    Notes
    -----
    Exactly ONE of the parameters ``eta_squared``, ``m``, ``n``, ``power`` and ``alpha`` must be
    passed as None, and that parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    GPower uses the :math:`f` effect size instead of the :math:`\\eta^2`. The formula to convert
    from one to the other are given below:

    .. math:: f = \\sqrt{\\frac{\\eta^2}{1 - \\eta^2}}

    .. math:: \\eta^2 = \\frac{f^2}{1 + f^2}

    Using :math:`\\eta^2`, the sample size :math:`N`, the number of repeated measurements
    :math:`m`, the epsilon correction factor :math:`\\epsilon` (see :py:func:`pingouin.epsilon`),
    and the average correlation between the repeated measures :math:`c`, one can then calculate the
    non-centrality parameter as follow:

    .. math:: \\delta = \\frac{f^2 * N * m * \\epsilon}{1 - c}

    Then the critical value of the non-central F-distribution is computed using the percentile
    point function of the F-distribution with:

    .. math:: q = 1 - \\alpha
    .. math:: v_1 = (m - 1) * \\epsilon
    .. math:: v_2 = (N - 1) * v_1

    Finally, the power of the ANOVA is calculated using the survival function of the non-central
    F-distribution using the previously computed critical value, non-centrality parameter,
    and degrees of freedom.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables
    (i.e. sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_rm_anova
    >>> print("power: %.4f" % power_rm_anova(eta_squared=0.1, m=3, n=20))
    power: 0.8913

    2. Compute required number of groups

    >>> print("m: %.4f" % power_rm_anova(eta_squared=0.1, n=20, power=0.90))
    m: 3.1347

    3. Compute required sample size

    >>> print("n: %.4f" % power_rm_anova(eta_squared=0.1, m=3, power=0.80))
    n: 15.9979

    4. Compute achieved effect size

    >>> print("eta-squared: %.4f" % power_rm_anova(n=20, m=4, power=0.80, alpha=0.05))
    eta-squared: 0.0680

    5. Compute achieved alpha (significance)

    >>> print("alpha: %.4f" % power_rm_anova(eta_squared=0.1, n=20, m=4, power=0.80, alpha=None))
    alpha: 0.0081

    Let's take a more concrete example. First, we'll load a repeated measures
    dataset in wide-format. Each row is an observation (e.g. a subject), and
    each column a successive repeated measurements (e.g t=0, t=1, ...).

    >>> import pingouin as pg
    >>> data = pg.read_dataset("rm_anova_wide")
    >>> data.head()
       Before  1 week  2 week  3 week
    0     4.3     5.3     4.8     6.3
    1     3.9     2.3     5.6     4.3
    2     4.5     2.6     4.1     NaN
    3     5.1     4.2     6.0     6.3
    4     3.8     3.6     4.8     6.8

    Note that this dataset has some missing values. We'll simply delete any row with one or more
    missing values, and then compute a repeated measures ANOVA:

    >>> data = data.dropna()
    >>> pg.rm_anova(data, effsize="n2").round(3)
       Source  ddof1  ddof2      F  p_unc     n2    eps
    0  Within      3     24  5.201  0.007  0.346  0.694

    The repeated measures ANOVA is significant at the 0.05 level. Now, we can
    easily compute the power of the ANOVA with the information in the ANOVA table:

    >>> # n is the sample size and m is the number of repeated measures
    >>> n, m = data.shape
    >>> round(pg.power_rm_anova(eta_squared=0.346, m=m, n=n, epsilon=0.694), 3)
    0.99

    Our ANOVA has a very high statistical power. However, to be even more accurate in our power
    calculation, we should also fill in the average correlation among repeated measurements.
    Since our dataframe is in wide-format (with each column being a successive measurement), this
    can be done by taking the mean of the superdiagonal of the correlation matrix, which is similar
    to manually calculating the correlation between each successive pairwise measurements and then
    taking the mean. Since correlation coefficients are not normally distributed, we use the
    *r-to-z* transform prior to averaging (:py:func:`numpy.arctanh`), and then the *z-to-r*
    transform (:py:func:`numpy.tanh`) to convert back to a correlation coefficient. This gives a
    more precise estimate of the mean.

    >>> import numpy as np
    >>> corr = np.diag(data.corr(), k=1)
    >>> avgcorr = np.tanh(np.arctanh(corr).mean())
    >>> round(avgcorr, 4)
    -0.1996

    In this example, we're using a fake dataset and the average correlation is negative. However,
    it will most likely be positive with real data. Let's now compute the final power of the
    repeated measures ANOVA:

    >>> round(pg.power_rm_anova(eta_squared=0.346, m=m, n=n, epsilon=0.694, corr=avgcorr), 3)
    0.771
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [eta_squared, m, n, power, alpha]])
    if n_none != 1:
        msg = "Exactly one of eta, m, n, power, and alpha must be None."
        raise ValueError(msg)

    # Safety checks
    assert 0 < epsilon <= 1, "epsilon must be between 0 and 1."
    assert -1 < corr < 1, "corr must be between -1 and 1."
    if eta_squared is not None:
        eta_squared = abs(eta_squared)
        f_sq = eta_squared / (1 - eta_squared)
    if alpha is not None:
        assert 0 < alpha <= 1, "alpha must be between 0 and 1."
    if power is not None:
        assert 0 < power <= 1, "power must be between 0 and 1."
    if n is not None:
        assert n > 1, "The sample size n must be > 1."
    if m is not None:
        assert m > 1, "The number of repeated measures m must be > 1."

    def func(f_sq, m, n, alpha):
        dof1 = (m - 1) * epsilon
        dof2 = (n - 1) * dof1
        nc = (f_sq * n * m * epsilon) / (1 - corr)
        fcrit = stats.f.ppf(1 - alpha, dof1, dof2)
        return _ncf_sf(fcrit, dof1, dof2, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(f_sq, m, n, alpha)
    elif m is None:
        # Compute required number of repeated measures
        return _solve(lambda m: func(f_sq, m, n, alpha) - power, 2, 100)
    elif n is None:
        # Compute required sample size
        return _solve(lambda n: func(f_sq, m, n, alpha) - power, 2, 1e6)
    elif eta_squared is None:
        # Compute achieved eta-squared. Solve directly on eta-squared in (0, 1): solving on
        # f_sq in (0, 1) would restrict eta-squared to be lower than 0.5.
        return _solve(lambda eta: func(eta / (1 - eta), m, n, alpha) - power, 1e-10, 1 - 1e-10)
    else:
        # Compute achieved alpha
        return _solve(lambda alpha: func(f_sq, m, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_corr(r=None, n=None, power=None, alpha=0.05, alternative="two-sided"):
    """
    Evaluate power, sample size, correlation coefficient or significance level of a correlation
    test.

    Parameters
    ----------
    r : float
        Correlation coefficient.
    n : int
        Number of observations (sample size).
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability). The default is 0.05.
    alternative : string
        Defines the alternative hypothesis, or tail of the correlation. Must be one of
        "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided
        p-value. "greater" tests against the alternative hypothesis that the correlation is
        positive (greater than zero), "less" tests against the hypothesis that the correlation is
        negative.

    Notes
    -----
    Exactly ONE of the parameters ``r``, ``n``, ``power`` and ``alpha`` must be passed as None,
    and that parameter is determined from the others.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    This function is a Python adaptation of the `pwr.r.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Examples
    --------
    1. Compute achieved power given ``r``, ``n`` and ``alpha``

    >>> from pingouin import power_corr
    >>> print("power: %.4f" % power_corr(r=0.5, n=20))
    power: 0.6379

    2. Same but one-sided test

    >>> print("power: %.4f" % power_corr(r=0.5, n=20, alternative="greater"))
    power: 0.7510

    >>> print("power: %.4f" % power_corr(r=0.5, n=20, alternative="less"))
    power: 0.0000

    3. Compute required sample size given ``r``, ``power`` and ``alpha``

    >>> print("n: %.4f" % power_corr(r=0.5, power=0.80))
    n: 28.2484

    4. Compute achieved ``r`` given ``n``, ``power`` and ``alpha`` level

    >>> print("r: %.4f" % power_corr(n=20, power=0.80, alpha=0.05))
    r: 0.5822

    5. Compute achieved alpha level given ``r``, ``n`` and ``power``

    >>> print("alpha: %.4f" % power_corr(r=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.1377
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [r, n, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of n, r, power, and alpha must be None")

    # Safety checks
    _check_alternative(alternative)

    if r is not None:
        assert -1 <= r <= 1
        if alternative == "two-sided":
            r = abs(r)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1
    if n is not None:
        if n <= 4:
            warnings.warn("Sample size is too small to estimate power (n <= 4). Returning NaN.")
            return np.nan

    # Define main function. A "less" test is a "greater" test on the negated correlation.
    tside = 2 if alternative == "two-sided" else 1

    def func(r, n, alpha):
        if alternative == "less":
            r = -r
        dof = n - 2
        ttt = stats.t.ppf(1 - alpha / tside, dof)
        rc = np.sqrt(ttt**2 / (ttt**2 + dof))
        zr = np.arctanh(r) + r / (2 * (n - 1))
        zrc = np.arctanh(rc)
        pwr = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3))
        if alternative == "two-sided":
            pwr = pwr + stats.norm.cdf((-zr - zrc) * np.sqrt(n - 3))
        return pwr

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given r, n and alpha
        return func(r, n, alpha)
    elif n is None:
        # Compute required sample size given r, power and alpha
        return _solve(lambda n: func(r, n, alpha) - power, 4 + 1e-10, 1e09)
    elif r is None:
        # Compute achieved r given sample size, power and alpha level
        lower = 1e-10 if alternative == "two-sided" else -1 + 1e-10
        return _solve(lambda r: func(r, n, alpha) - power, lower, 1 - 1e-10)
    else:
        # Compute achieved alpha (significance) level given r, n and power
        return _solve(lambda alpha: func(r, n, alpha) - power, 1e-10, 1 - 1e-10)


def power_chi2(dof, w=None, n=None, power=None, alpha=0.05):
    """
    Evaluate power, sample size, effect size or significance level of chi-squared tests.

    Parameters
    ----------
    dof : float
        Degree of freedom (depends on the chosen test).
    w : float
        Cohen's w effect size [1]_.
    n : int
        Total number of observations.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability). The default is 0.05.

    Notes
    -----
    Exactly ONE of the parameters ``w``, ``n``, ``power`` and ``alpha`` must be passed as None,
    and that parameter is determined from the others. The degrees of freedom ``dof`` must always
    be specified.

    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.

    This function is a Python adaptation of the `pwr.chisq.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.

    The non-centrality parameter is defined by:

    .. math:: \\delta = N * w^2

    Then the critical value is computed using the percentile point function of the :math:`\\chi^2`
    distribution with the alpha level and degrees of freedom.

    Finally, the power of the chi-squared test is calculated using the survival function of the
    non-central :math:`\\chi^2` distribution using the previously computed critical value,
    non-centrality parameter, and the degrees of freedom of the test.

    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.

    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).

    Examples
    --------
    1. Compute achieved power

    >>> from pingouin import power_chi2
    >>> print("power: %.4f" % power_chi2(dof=1, w=0.3, n=20))
    power: 0.2687

    2. Compute required sample size

    >>> print("n: %.4f" % power_chi2(dof=3, w=0.3, power=0.80))
    n: 121.1396

    3. Compute achieved effect size

    >>> print("w: %.4f" % power_chi2(dof=2, n=20, power=0.80, alpha=0.05))
    w: 0.6941

    4. Compute achieved alpha (significance)

    >>> print("alpha: %.4f" % power_chi2(dof=1, w=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.1630
    """
    assert isinstance(dof, (int, float))
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [w, n, power, alpha]])
    if n_none != 1:
        err = "Exactly one of w, n, power, and alpha must be None."
        raise ValueError(err)

    # Safety checks
    if w is not None:
        w = abs(w)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    def func(w, n, alpha):
        k = stats.chi2.ppf(1 - alpha, dof)
        nc = n * w**2
        return stats.ncx2.sf(k, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power
        return func(w, n, alpha)
    elif n is None:
        # Compute required sample size
        return _solve(lambda n: func(w, n, alpha) - power, 1, 1e07)
    elif w is None:
        # Compute achieved effect size
        return _solve(lambda w: func(w, n, alpha) - power, 1e-10, 100)
    else:
        # Compute achieved alpha
        return _solve(lambda alpha: func(w, n, alpha) - power, 1e-10, 1 - 1e-10)
