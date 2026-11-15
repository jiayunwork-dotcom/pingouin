# Internal numerical helpers for group-sequential (interim analysis) designs.
#
# The public entry points are ``pingouin.sequential_bounds`` (in ``power.py``) and
# ``pingouin.sequential_ttest`` (in ``parametric.py``). Both rely on the exact same boundary
# calculation implemented here, so that the nominal p-values used by the test are bit-for-bit
# identical to those returned by ``sequential_bounds``.
from functools import lru_cache

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import ndtri

# Number of nodes of the Gauss-Legendre quadrature used to propagate the joint distribution of
# the interim test statistics. 320 nodes give an absolute error well below 1e-9 for up to ~20
# analyses, while keeping a single call below a few hundred milliseconds.
_NODES = 320

# Half-width (in standard deviation units) of the continuation region, measured from the
# boundary. The Gaussian density is negligible further away, so the region is truncated.
_TAIL = 14.0

# Largest magnitude (in standard deviation units) of the non-centrality parameter used when
# calibrating the joint distribution of the sequential T-test statistics.
_MAX_MU = 60.0

# Smallest bound considered by the root finder.
_TINY = 1e-14

# Smallest cumulative type-I error considered when computing boundaries. Spending functions can
# underflow to exactly zero for extremely small information fractions (e.g. O'Brien-Fleming at
# t = 1e-8), which would put the corresponding boundary at infinity. Flooring at 1e-15 keeps the
# boundary finite (~7.9 standard deviations); the induced error is far below the practical
# precision of the design.
_SPEND_FLOOR = 1e-15

# Largest magnitude of the efficacy boundary, in standard deviation units. It only matters when
# the type-I error of the first look is so small that its normal quantile is not representable.
_MAX_BOUND = 38.0


def _check_looks(looks, require_last_one=False):
    """Validate and return the information fractions of a group-sequential design.

    The values must form a one-dimensional, strictly increasing sequence of finite numbers in
    (0, 1]. Booleans are rejected. If ``require_last_one`` is True, the last value must be exactly
    1 (i.e. the final analysis uses the full sample).
    """
    if looks is None:
        raise ValueError("looks must be provided.")
    if isinstance(looks, (str, bytes)):
        raise ValueError("looks must be a one-dimensional sequence of numbers in (0, 1].")
    try:
        items = list(looks)
    except TypeError as err:
        raise ValueError("looks must be a one-dimensional sequence of numbers in (0, 1].") from err
    if not items:
        raise ValueError("looks must contain at least one value.")
    for item in items:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError("looks must contain numbers in (0, 1], not booleans.")
    try:
        arr = np.asarray(items, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("looks must be a one-dimensional sequence of numbers in (0, 1].") from err

    if arr.ndim != 1:
        raise ValueError("looks must be a one-dimensional sequence of numbers in (0, 1].")
    if not np.all(np.isfinite(arr)):
        raise ValueError("looks must only contain finite numbers.")
    if np.any(arr <= 0) or np.any(arr > 1):
        raise ValueError("Every value in looks must be in the interval (0, 1].")
    if arr.size > 1 and not np.all(np.diff(arr) > 0):
        raise ValueError("looks must be strictly increasing.")
    if require_last_one and arr[-1] != 1.0:
        raise ValueError("The last value in looks must be 1.")
    return arr


def _check_spending(spending):
    """Validate and normalize the type-I error spending function."""
    if isinstance(spending, (bool, np.bool_)):
        raise ValueError("spending must be 'obf', 'pocock', or a positive finite float rho.")
    if isinstance(spending, str):
        if spending not in ("obf", "pocock"):
            raise ValueError("spending must be 'obf', 'pocock', or a positive finite float rho.")
        return spending
    if isinstance(spending, (int, float, np.integer, np.floating)):
        rho = float(spending)
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError("spending must be 'obf', 'pocock', or a positive finite float rho.")
        return rho
    raise ValueError("spending must be 'obf', 'pocock', or a positive finite float rho.")


def _check_alpha(alpha):
    """Validate a sequential significance level, which must lie in (0, 0.5)."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise ValueError("alpha must be a number in (0, 0.5).")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not (0 < alpha < 0.5):
        raise ValueError("alpha must be a number in (0, 0.5).")
    return alpha


def _check_alternative(alternative):
    """Validate the alternative hypothesis."""
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be one of 'two-sided', 'greater' or 'less'.")


def _spending_increment(t, alpha, spending):
    """Cumulative type-I error spending function, evaluated at each information fraction."""
    t = np.asarray(t, float)
    if spending == "obf":
        z = ndtri(1 - alpha / 2)
        spent = 2 * stats.norm.sf(z / np.sqrt(t))
    elif spending == "pocock":
        spent = alpha * np.log(1 + (np.e - 1) * t)
    else:  # rho > 0
        spent = alpha * t**spending
    spent = np.asarray(spent, float)
    # The spending function is exact at t = 1; enforce it to avoid quadrature noise.
    spent = np.where(t == 1, alpha, spent)
    return spent


def _quantile(prob):
    """Standard normal quantile, clipped to a representable, finite value."""
    prob = min(max(prob, 1e-300), 1 - 1e-16)
    return float(np.clip(ndtri(prob), -_MAX_BOUND, _MAX_BOUND))


def _leggauss(n):
    """Cached Gauss-Legendre nodes and weights on [-1, 1]."""
    return _leggauss_cached(n)


@lru_cache(maxsize=4)
def _leggauss_cached(n):
    return np.polynomial.legendre.leggauss(n)


def _quad(lo, hi, n=_NODES):
    """Gauss-Legendre nodes and weights mapped to the interval [lo, hi]."""
    x, w = _leggauss(n)
    x = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    w = w * (0.5 * (hi - lo))
    return x, w


def _continuation_region(bound, alternative):
    """Continuation region of a single look, truncated at ``_TAIL`` standard deviations.

    The region is made to always contain the interval [-``_TAIL``, ``_TAIL``] (for one-sided
    tests), so that it captures the bulk of the standard normal density even when the boundary is
    far in the tail.
    """
    if alternative == "greater":
        return min(bound - _TAIL, -_TAIL), bound
    if alternative == "less":
        return bound, max(bound + _TAIL, _TAIL)
    return -bound, bound


def _crossing_probability(bound, x, v, rho, s, alternative):
    """Probability of crossing at the current look, for the given continuation density ``v``."""
    m = (bound - rho * x) / s
    if alternative == "greater":
        prob = stats.norm.sf(m)
    elif alternative == "less":
        prob = stats.norm.cdf(m)
    else:
        prob = stats.norm.cdf((-bound - rho * x) / s) + stats.norm.sf(m)
    return np.dot(v, prob)


def sequential_bound_table(t, alpha, alternative, spending):
    """Boundaries, nominal p-values and cumulative spending of a validated design.

    Parameters are assumed to be already validated. With a single look, the design is a
    fixed-sample test: the nominal p-value and the cumulative spending are exactly ``alpha``.
    """
    t = np.asarray(t, float)
    if t.size == 1:
        if alternative == "greater":
            bound = np.array([ndtri(1 - alpha)], float)
        elif alternative == "less":
            bound = np.array([ndtri(alpha)], float)
        else:
            bound = np.array([ndtri(1 - alpha / 2)], float)
        return bound, np.array([alpha], float), np.array([alpha], float)
    bound, nominal, spent = sequential_bounds(t, alpha, alternative, spending, return_spent=True)
    return bound, nominal, spent


def sequential_bounds(t, alpha, alternative, spending, return_spent=False):
    """
    Compute the group-sequential efficacy boundaries and their nominal p-values.

    Parameters
    ----------
    t : array_like
        Information fractions of each look, strictly increasing in (0, 1].
    alpha : float
        Overall two-sided (``alternative="two-sided"``) or one-sided significance level.
    alternative : str
        One of "two-sided", "greater" or "less".
    spending : str or float
        "obf", "pocock", or a positive float ``rho``.
    return_spent : bool
        If True, also return the cumulative type-I error spending at each look.

    Returns
    -------
    bound : np.ndarray
        Efficacy boundaries, in the standardized Z scale.
    nominal : np.ndarray
        Nominal p-value of each look (one-sided or two-sided, matching ``alternative``).
    spent : np.ndarray
        Cumulative type-I error spent up to and including each look.
    """
    t = np.asarray(t, float)
    spent = _spending_increment(t, alpha, spending)
    k_last = t.size
    bound = np.empty(k_last)

    if alternative == "greater":
        bound[0] = _quantile(1 - spent[0])
    elif alternative == "less":
        bound[0] = _quantile(spent[0])
    else:
        bound[0] = _quantile(1 - spent[0] / 2)

    lo, hi = _continuation_region(bound[0], alternative)
    x, w = _quad(lo, hi)
    # Density of Z_1 restricted to the continuation region (unnormalized).
    v = w * stats.norm.pdf(x)

    for k in range(1, k_last):
        rho = np.sqrt(t[k - 1] / t[k])
        s = np.sqrt(1 - rho**2)
        if alternative == "less":
            bracket = (-_MAX_BOUND, -_TINY)
        else:
            bracket = (_TINY, _MAX_BOUND)
        target = max(spent[k] - spent[k - 1], _SPEND_FLOOR)
        bound[k] = brentq(
            lambda b: _crossing_probability(b, x, v, rho, s, alternative) - target,
            *bracket,
            xtol=1e-15,
            rtol=4 * np.finfo(float).eps,
            maxiter=200,
        )
        bound[k] = np.clip(bound[k], -_MAX_BOUND, _MAX_BOUND)
        lo, hi = _continuation_region(bound[k], alternative)
        x2, w2 = _quad(lo, hi)
        # Transition density: Z_k = rho * Z_{k-1} + sqrt(1 - rho^2) * Y.
        m = stats.norm.pdf((x2[:, None] - rho * x[None, :]) / s) / s
        v = w2 * (m @ v)
        x = x2

    if alternative == "greater":
        nominal = stats.norm.sf(bound)
    elif alternative == "less":
        nominal = stats.norm.cdf(bound)
    else:
        nominal = 2 * stats.norm.sf(bound)

    if return_spent:
        return bound, nominal, spent
    return bound, nominal


def _calibrate_mean(bound, rejection_prob, alternative):
    """Non-centrality parameter whose marginal rejection probability equals ``rejection_prob``.

    The marginal distribution of the standardized statistic at a given look is a standard normal
    shifted by this non-centrality parameter. It is derived from the marginal T-test power, so
    that the sequential power is exactly consistent with the fixed-sample power at every look.
    """
    if alternative == "greater":
        return bound - ndtri(1 - rejection_prob)
    if alternative == "less":
        return bound - ndtri(rejection_prob)
    base = 2 * stats.norm.sf(bound)
    if rejection_prob <= base:
        # No positive shift can lower the two-sided rejection probability below its null value;
        # this only happens through round-off for a vanishing effect size.
        return 0.0
    # Solve on the log scale for numerical accuracy when the marginal probability is tiny.
    lo, hi = 0.0, _MAX_MU
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if stats.norm.cdf(-bound - mid) + stats.norm.sf(bound - mid) < rejection_prob:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13:
            break
    return 0.5 * (lo + hi)


def sequential_power(t, bound, rejection_prob, alternative):
    """
    Probability of rejecting the null hypothesis at any look.

    Parameters
    ----------
    t : array_like
        Information fractions.
    bound : array_like
        Efficacy boundaries, in the standardized Z scale.
    rejection_prob : array_like
        Marginal rejection probability of each look, under the alternative. These are obtained
        from the fixed-sample T-test power evaluated at the nominal level of that look.
    alternative : str
        One of "two-sided", "greater" or "less".
    """
    t = np.asarray(t, float)
    rejection_prob = np.asarray(rejection_prob, float)
    if np.any(rejection_prob >= 1.0):
        return 1.0
    rejection_prob = np.clip(rejection_prob, 1e-15, 1 - 1e-16)

    k_last = t.size
    mu = np.array(
        [_calibrate_mean(bound[k], rejection_prob[k], alternative) for k in range(k_last)]
    )

    # Crossing probability of the first look, under the alternative.
    if alternative == "greater":
        power = stats.norm.sf(bound[0] - mu[0])
    elif alternative == "less":
        power = stats.norm.cdf(bound[0] - mu[0])
    else:
        power = stats.norm.cdf(-bound[0] - mu[0]) + stats.norm.sf(bound[0] - mu[0])

    lo, hi = _continuation_region(bound[0], alternative)
    x, w = _quad(lo, hi)
    v = w * stats.norm.pdf(x - mu[0])

    for k in range(1, k_last):
        rho = np.sqrt(t[k - 1] / t[k])
        s = np.sqrt(1 - rho**2)
        # Conditional mean of Z_k given the continuation value at look k - 1.
        shift = mu[k] - rho * mu[k - 1]
        m = (bound[k] - rho * x - shift) / s
        if alternative == "greater":
            prob = stats.norm.sf(m)
        elif alternative == "less":
            prob = stats.norm.cdf(m)
        else:
            prob = stats.norm.cdf((-bound[k] - rho * x - shift) / s) + stats.norm.sf(m)
        power += np.dot(v, prob)

        lo, hi = _continuation_region(bound[k], alternative)
        x2, w2 = _quad(lo, hi)
        dens = stats.norm.pdf((x2[:, None] - shift - rho * x[None, :]) / s) / s
        v = w2 * (dens @ v)
        x = x2

    return float(np.clip(power, 0.0, 1.0))
