"""Regression tests for partial correlation with redundant covariates and for
non-estimable pairs in pairwise_corr.

Covers:
- redundant / constant / proportional / linearly dependent / recoded covariates
  leaving r, p_val and CI95 unchanged (Pearson and Spearman, one- and two-sided);
- non-estimable partial correlations (control information fully explains a
  controlled variable) returning NaN instead of plausible-looking numbers;
- pairwise_corr keeping non-estimable pairs as NaN rows without interrupting
  the other pairs, and mapping multiple-comparison corrections back to the
  correct rows;
- numerical accuracy against an independent reference implementation
  (scipy.linalg.lstsq residualization + textbook t/Fisher formulas), including
  full-rank covariate matrices with condition number up to 1e6.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sst
from scipy.linalg import lstsq

import pingouin as pg
from pingouin import corr, multicomp, pairwise_corr, partial_corr


@pytest.fixture
def no_rounding():
    """Disable all output rounding (restored after the test)."""
    old = pg.options.copy()
    pg.options["round"] = None
    pg.options["round.column.CI95"] = None
    yield
    pg.options.clear()
    pg.options.update(old)


def _ref_partial(x, y, Z=None, Zx=None, Zy=None, k=None, alternative="two-sided"):
    """Independent (semi-)partial correlation reference.

    Residualizes x on [1, Zx] and y on [1, Zy] with scipy.linalg.lstsq (GELSD,
    a different code path than the SVD-basis kernel tested here), then applies
    the textbook t-test (as in R's ppcor) and Fisher confidence interval
    formulas. Only valid for full-rank covariate matrices. ``Z`` is a shortcut
    for a full partial correlation (same covariates on both sides).
    """
    n = x.size
    if Z is not None:
        Zx = Zy = Z

    def resid(v, Z):
        if Z is None or Z.shape[1] == 0:
            return v - v.mean()
        A = np.column_stack([np.ones(n), Z])
        return v - A @ lstsq(A, v)[0]

    rx, ry = resid(x, Zx), resid(y, Zy)
    r = float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))
    if k is None:  # effective number of control directions
        k = Zx.shape[1] if Zx is not None else (Zy.shape[1] if Zy is not None else 0)
    dof = n - k - 2
    tval = r * np.sqrt(dof / (1 - r**2))
    if alternative == "two-sided":
        p = 2 * sst.t.sf(abs(tval), dof)
    elif alternative == "greater":
        p = sst.t.sf(tval, dof)
    else:
        p = sst.t.cdf(tval, dof)
    zz, se = np.arctanh(r), 1 / np.sqrt(n - k - 3)
    if alternative == "two-sided":
        crit = abs(sst.norm.ppf(0.025))
        ci = np.tanh([zz - crit * se, zz + crit * se])
    elif alternative == "greater":
        ci = np.array([np.tanh(zz - sst.norm.ppf(0.95) * se), 1.0])
    else:
        ci = np.array([-1.0, np.tanh(zz + sst.norm.ppf(0.95) * se)])
    return r, p, ci


def _repro_data():
    """The synthetic dataset of the bug report."""
    rng = np.random.default_rng(16)
    z = rng.normal(size=40)
    w = rng.normal(size=40)
    x = 2 * z + rng.normal(size=40)
    y = 3 * z + 0.4 * x + rng.normal(size=40)
    return pd.DataFrame(dict(x=x, y=y, z=z, z2=2 * z, w=w, absorbed=z + 2 * w, exact=z.copy()))


def _stats(row, method="pearson"):
    """(r, p_val, CI95) of a one-row partial_corr output, as floats."""
    return (
        row.at[method, "r"],
        row.at[method, "p_val"],
        np.asarray(row.at[method, "CI95"], dtype=float),
    )


class TestPartialCorrRedundant:
    """Redundant covariates and non-estimable partial correlations."""

    def test_repro_redundant_covariate_same_inference(self, no_rounding):
        """Bug report, case 1-2: an extra proportional covariate changed the
        p-value because the degrees of freedom counted it. Both calls must now
        give identical r, p_val and CI95."""
        df = _repro_data()
        r1, p1, c1 = _stats(partial_corr(data=df, x="x", y="y", covar=["z"]))
        with pytest.warns(UserWarning, match="rank-deficient"):
            res2 = partial_corr(data=df, x="x", y="y", covar=["z", "z2"])
        r2, p2, c2 = _stats(res2)
        assert r2 == pytest.approx(r1, abs=1e-12)
        assert p2 == pytest.approx(p1, abs=1e-12)
        assert np.allclose(c1, c2, atol=1e-12)
        # Anchor on the value reported as correct in the bug report
        # (previously 0.000170862 with one covariate, wrongly 0.000209900 with
        # the duplicated one).
        assert p1 == pytest.approx(0.000170862, abs=1e-6)
        # Independent reference (lstsq residualization on [1, z])
        r_ref, p_ref, ci_ref = _ref_partial(
            df["x"].to_numpy(), df["y"].to_numpy(), df[["z"]].to_numpy()
        )
        assert r1 == pytest.approx(r_ref, abs=1e-12)
        assert p1 == pytest.approx(p_ref, abs=1e-12)
        assert np.allclose(c1, ci_ref, atol=1e-12)

    def test_repro_absorbed_variable_not_estimable(self):
        """Bug report, case 3: 'absorbed' is an exact linear combination of
        the covariates. The correlation is not estimable: NaN statistics (not
        a finite r ~ 0.72 with a significant p-value), but n is kept."""
        df = _repro_data()
        res = partial_corr(data=df, x="absorbed", y="y", covar=["z", "w"])
        assert res.at["pearson", "n"] == 40
        assert np.isnan(res.at["pearson", "r"])
        assert np.isnan(res.at["pearson", "p_val"])
        assert np.isnan(res.at["pearson", "CI95"])
        # Same if y is the absorbed variable
        res = partial_corr(data=df, x="x", y="absorbed", covar=["z", "w"])
        assert res.at["pearson", "n"] == 40
        assert np.isnan(res.at["pearson", "r"])
        # A covariate numerically identical to x or y (issue #375) is a data
        # condition, not an input error: NaN, not an exception.
        res = partial_corr(data=df, x="exact", y="y", covar=["z"])
        assert res.at["pearson", "n"] == 40
        assert np.isnan(res.at["pearson", "r"])
        res = partial_corr(data=df, x="x", y="exact", covar=["z"])
        assert res.at["pearson", "n"] == 40
        assert np.isnan(res.at["pearson", "r"])

    def test_repro_pairwise_keeps_valid_pairs(self, no_rounding):
        """Bug report, case 4: pairwise_corr must not raise when one pair is
        not estimable; the valid pairs are still returned and adjusted."""
        df = _repro_data()
        out = pg.pairwise_corr(df, columns=["x", "y", "exact"], covar=["z"], padjust="holm")
        assert out.shape[0] == 3
        assert list(out["X"]) == ["x", "x", "y"]
        assert list(out["Y"]) == ["y", "exact", "exact"]
        # The (x, y) pair is valid and matches a direct partial_corr call
        direct = partial_corr(data=df, x="x", y="y", covar=["z"])
        row_xy = out[(out["X"] == "x") & (out["Y"] == "y")].iloc[0]
        assert row_xy["n"] == 40
        assert row_xy["r"] == direct.at["pearson", "r"]
        assert row_xy["p_unc"] == direct.at["pearson", "p_val"]
        # A single finite p-value forms the correction family: holm of one
        # p-value is the p-value itself
        assert row_xy["p_corr"] == row_xy["p_unc"]
        assert row_xy["p_adjust"] == "holm"
        # The two pairs involving 'exact' are not estimable but kept as NaN rows
        for _, row in out[out["Y"] == "exact"].iterrows():
            assert row["n"] == 40
            assert np.isnan(row["r"])
            assert np.isnan(row["p_unc"])
            assert np.isnan(row["p_corr"])

    @pytest.mark.parametrize("method", ["pearson", "spearman"])
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_equivalent_control_representations(self, method, alternative, no_rounding):
        """Duplicated, proportional, exactly linearly dependent or constant
        covariate columns, and invertible linear recodings, do not change
        r, p_val or CI95 — as long as the control space *actually used by the
        method* is unchanged (raw values for Pearson, ranks for Spearman)."""
        rng = np.random.default_rng(7)
        n = 60
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        x = 1.5 * z1 - 0.7 * z2 + rng.normal(size=n)
        y = 0.9 * z1 + 0.4 * z2 + 0.5 * x + rng.normal(size=n)
        df = pd.DataFrame(dict(x=x, y=y, z1=z1, z2=z2))
        df["z1b"] = 2 * df["z1"]
        df["zsum"] = df["z1"] + df["z2"]
        df["zmix"] = 3 * df["z1"] - df["z2"]
        df["const"] = 7.5
        df["z1c"] = df["z1"] * 1e4
        df["z2c"] = df["z2"] * 1e-4
        df["z1cube"] = df["z1"] ** 3  # strictly increasing: same ranks as z1
        kw = dict(data=df, x="x", y="y", method=method, alternative=alternative)
        ref = _stats(partial_corr(covar=["z1", "z2"], **kw), method)
        # Variants that leave the control space unchanged for both methods
        variants = {
            "duplicated": ["z1", "z2", "z1"],
            "proportional": ["z1", "z2", "z1b"],
            "constant": ["z1", "z2", "const"],
            "rescaled": ["z1c", "z2c"],
        }
        if method == "pearson":
            # Raw-value linear dependence / recodings: Pearson-only equivalences
            variants["linear_dependent"] = ["z1", "z2", "zsum", "zmix"]
        else:
            # Strictly increasing transforms preserve the ranks exactly
            variants["monotone_transform"] = ["z1cube", "z2"]
        for name, covs in variants.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # rank-deficiency warning
                got = _stats(partial_corr(covar=covs, **kw), method)
            for i, what in enumerate(["r", "p_val", "CI95"]):
                assert np.allclose(got[i], ref[i], atol=1e-8), (name, what, got[i], ref[i])
        if method == "pearson":
            # Invertible linear recoding of the covariates
            df[["r1", "r2"]] = df[["z1", "z2"]].to_numpy() @ np.array([[2.0, 1.0], [1.0, 1.0]])
            got = _stats(partial_corr(covar=["r1", "r2"], **kw), method)
            for i, what in enumerate(["r", "p_val", "CI95"]):
                assert np.allclose(got[i], ref[i], atol=1e-8), ("recoded", what)

    def test_spearman_uses_rank_space(self, no_rounding):
        """For Spearman, the control space is defined by the ranks of the
        observations actually used: raw linear dependence does not imply rank
        dependence (and vice versa, dependence is checked on the ranks)."""
        rng = np.random.default_rng(7)
        n = 60
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        x = 1.5 * z1 - 0.7 * z2 + rng.normal(size=n)
        y = 0.9 * z1 + 0.4 * z2 + 0.5 * x + rng.normal(size=n)
        df = pd.DataFrame(dict(x=x, y=y, z1=z1, z2=z2))
        df["z1b"] = 2 * df["z1"]  # same ranks as z1 -> redundant for Spearman
        df["zsum"] = df["z1"] + df["z2"]  # raw-dependent, but rank-INdependent
        s1 = _stats(
            partial_corr(data=df, x="x", y="y", covar=["z1", "z2"], method="spearman"), "spearman"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s2 = _stats(
                partial_corr(data=df, x="x", y="y", covar=["z1", "z2", "z1b"], method="spearman"),
                "spearman",
            )
            s3 = _stats(
                partial_corr(data=df, x="x", y="y", covar=["z1", "z2", "zsum"], method="spearman"),
                "spearman",
            )
        # rank(2*z1) == rank(z1): identical control space -> identical results
        for i in range(3):
            assert np.allclose(s2[i], s1[i], atol=1e-8)
        # rank(z1+z2) is a genuinely new direction: 3 effective covariates
        assert not np.isclose(s3[1], s1[1])
        # ... whereas for Pearson zsum is exactly linearly dependent
        p1 = partial_corr(data=df, x="x", y="y", covar=["z1", "z2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p3 = partial_corr(data=df, x="x", y="y", covar=["z1", "z2", "zsum"])
        assert p3.at["pearson", "r"] == pytest.approx(p1.at["pearson", "r"], abs=1e-12)
        assert p3.at["pearson", "p_val"] == pytest.approx(p1.at["pearson", "p_val"], abs=1e-12)
        # Spearman reference: the Pearson reference computed on the ranks
        r_ref, p_ref, _ = _ref_partial(
            sst.rankdata(x), sst.rankdata(y), np.column_stack([sst.rankdata(v) for v in [z1, z2]])
        )
        assert s1[0] == pytest.approx(r_ref, abs=1e-12)
        assert s1[1] == pytest.approx(p_ref, abs=1e-12)

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_full_rank_matches_independent_reference(self, alternative, no_rounding):
        """Full-rank inputs (incl. condition numbers up to 1e6) match the
        independent lstsq reference to 1e-8 on r, p_val and CI95."""
        rng = np.random.default_rng(31)
        n = 60
        for eps in [None, 1e-2, 1e-4, 1e-6]:  # eps=None: well-separated covariates
            u, v = rng.normal(size=n), rng.normal(size=n)
            if eps is None:
                c1, c2 = u, v
            else:
                c1, c2 = u + eps * v, u - eps * v
            Z = np.column_stack([c1, c2])
            Zs = (Z - Z.mean(0)) / Z.std(0, ddof=1)
            cond = np.linalg.cond(Zs)
            if eps is not None:
                assert 0.5 / eps < cond < 5 / eps  # sanity check on the construction
            x = 0.8 * c1 + rng.normal(size=n)
            y = 0.5 * c2 + 0.3 * x + rng.normal(size=n)
            df = pd.DataFrame(dict(x=x, y=y, c1=c1, c2=c2))
            got = partial_corr(data=df, x="x", y="y", covar=["c1", "c2"], alternative=alternative)
            r_ref, p_ref, ci_ref = _ref_partial(x, y, Z, alternative=alternative)
            assert got.at["pearson", "r"] == pytest.approx(r_ref, abs=1e-8)
            assert got.at["pearson", "p_val"] == pytest.approx(p_ref, abs=1e-8)
            assert np.allclose(
                np.asarray(got.at["pearson", "CI95"], dtype=float), ci_ref, atol=1e-8
            )

    def test_equivalent_representations_numerical_agreement(self, no_rounding):
        """Exactly rank-deficient equivalent control representations agree to
        1e-8 (not only after rounding) on r, p_val and CI95."""
        rng = np.random.default_rng(13)
        n = 80
        z1, z2 = rng.normal(size=n), rng.normal(size=n)
        x = z1 - z2 + rng.normal(size=n)
        y = 0.3 * z2 + 0.6 * x + rng.normal(size=n)
        df = pd.DataFrame(dict(x=x, y=y, z1=z1, z2=z2))
        df["z3"] = df["z1"] + 2 * df["z2"]
        df["z4"] = -df["z1"] + df["z2"]
        base = _stats(partial_corr(data=df, x="x", y="y", covar=["z1", "z2"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aug = _stats(partial_corr(data=df, x="x", y="y", covar=["z1", "z2", "z3", "z4"]))
            rec = _stats(partial_corr(data=df, x="x", y="y", covar=["z3", "z4"]))
        for got in [aug, rec]:
            for i in range(3):
                assert np.allclose(got[i], base[i], atol=1e-8)

    def test_semi_partial_one_side_only(self, no_rounding):
        """Semi-partial correlation controls only one side: it differs from
        the full partial correlation, obeys the same equivalence rules on its
        own control space, and only the controlled side can be absorbed."""
        rng = np.random.default_rng(7)
        n = 60
        z1, z2 = rng.normal(size=n), rng.normal(size=n)
        x = 1.5 * z1 - 0.7 * z2 + rng.normal(size=n)
        y = 0.9 * z1 + 0.4 * z2 + 0.5 * x + rng.normal(size=n)
        df = pd.DataFrame(dict(x=x, y=y, z1=z1, z2=z2))
        df["z1b"] = 2 * df["z1"]
        # Semi-partial != full partial (controlling one side is not both)
        semi_x = partial_corr(data=df, x="x", y="y", x_covar=["z1", "z2"])
        semi_y = partial_corr(data=df, x="x", y="y", y_covar=["z1", "z2"])
        full = partial_corr(data=df, x="x", y="y", covar=["z1", "z2"])
        assert not np.isclose(semi_x.at["pearson", "r"], full.at["pearson", "r"])
        assert not np.isclose(semi_y.at["pearson", "r"], full.at["pearson", "r"])
        # Independent references: residualize one side only
        Z = df[["z1", "z2"]].to_numpy()
        r_ref, p_ref, _ = _ref_partial(x, y, Zx=Z, Zy=None, k=2)
        assert semi_x.at["pearson", "r"] == pytest.approx(r_ref, abs=1e-12)
        assert semi_x.at["pearson", "p_val"] == pytest.approx(p_ref, abs=1e-12)
        r_ref, p_ref, _ = _ref_partial(x, y, Zx=None, Zy=Z, k=2)
        assert semi_y.at["pearson", "r"] == pytest.approx(r_ref, abs=1e-12)
        # Redundant covariate in the semi-partial control space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            semi_x2 = partial_corr(data=df, x="x", y="y", x_covar=["z1", "z2", "z1b"])
        assert semi_x2.at["pearson", "r"] == pytest.approx(semi_x.at["pearson", "r"], abs=1e-12)
        assert semi_x2.at["pearson", "p_val"] == pytest.approx(
            semi_x.at["pearson", "p_val"], abs=1e-12
        )
        # Only the controlled side can be absorbed: x_covar identical to x
        df["exact"] = df["z1"] * 2
        res = partial_corr(data=df, x="exact", y="y", x_covar=["z1"])
        assert np.isnan(res.at["pearson", "r"])
        assert res.at["pearson", "n"] == n
        # ... but with y_covar, x is not controlled and keeps its variance
        res = partial_corr(data=df, x="exact", y="y", y_covar=["z1"])
        assert np.isfinite(res.at["pearson", "r"])
        # Spearman semi-partial with redundant covariate
        s1 = partial_corr(data=df, x="x", y="y", x_covar=["z1", "z2"], method="spearman")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s2 = partial_corr(data=df, x="x", y="y", x_covar=["z1", "z2", "z1b"], method="spearman")
        assert s2.at["spearman", "r"] == pytest.approx(s1.at["spearman", "r"], abs=1e-12)
        assert s2.at["spearman", "p_val"] == pytest.approx(s1.at["spearman", "p_val"], abs=1e-12)

    def test_insufficient_information(self, no_rounding):
        """When the residuals are estimable but the degrees of freedom are not
        positive, r is kept and p_val / CI95 are NaN. With fewer than 3
        complete observations, everything is NaN (n is always reported)."""
        rng = np.random.default_rng(3)
        n = 5
        z1, z2 = rng.normal(size=n), rng.normal(size=n)
        df = pd.DataFrame(
            dict(x=z1 + 0.1 * rng.normal(size=n), y=2 * z1 + rng.normal(size=n), z1=z1, z2=z2)
        )
        # dof = 5 - 2 - 2 = 1: p-value defined, CI95 not (n - k - 3 = 0)
        res = partial_corr(data=df, x="x", y="y", covar=["z1", "z2"])
        assert np.isfinite(res.at["pearson", "r"])
        assert np.isfinite(res.at["pearson", "p_val"])
        assert np.isnan(res.at["pearson", "CI95"])
        # dof = 4 - 2 - 2 = 0: r kept, p_val and CI95 are NaN
        res = partial_corr(data=df.iloc[:4], x="x", y="y", covar=["z1", "z2"])
        assert np.isfinite(res.at["pearson", "r"])
        assert np.isnan(res.at["pearson", "p_val"])
        assert np.isnan(res.at["pearson", "CI95"])
        # Fewer than 3 complete observations: nothing is estimable, no error
        res = partial_corr(data=df.iloc[:2], x="x", y="y", covar=["z1"])
        assert res.at["pearson", "n"] == 2
        assert np.isnan(res.at["pearson", "r"])
        df_nan = df.copy()
        df_nan.loc[df_nan.index[2:], "x"] = np.nan
        res = partial_corr(data=df_nan, x="x", y="y", covar=["z1"])
        assert res.at["pearson", "n"] == 2
        assert np.isnan(res.at["pearson", "r"])

    def test_perfect_partial_correlation(self, no_rounding):
        """An exact +/- 1 partial correlation is estimable: r = +/- 1,
        p_val = 0 and a degenerate CI95, not NaN."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=30)
        z = rng.normal(size=30)
        df = pd.DataFrame(dict(x=x, y=2 * x + z, z=z))
        res = partial_corr(data=df, x="x", y="y", covar="z")
        assert res.at["pearson", "r"] == pytest.approx(1.0, abs=1e-12)
        assert res.at["pearson", "p_val"] == 0.0
        assert np.allclose(np.asarray(res.at["pearson", "CI95"], dtype=float), [1.0, 1.0])

    def test_input_errors_unchanged(self):
        """Configuration errors still raise; they are not swallowed into NaN
        rows: missing columns, invalid method/alternative, conflicting covar
        specifications, x == y, x/y among the covariates."""
        df = _repro_data()
        with pytest.raises(AssertionError):
            partial_corr(data=df, x="x", y="y", covar=["not_a_column"])
        with pytest.raises(AssertionError):
            partial_corr(data=df, x="x", y="y", covar=["z"], method="kendall")
        with pytest.raises(AssertionError):
            partial_corr(data=df, x="x", y="y", covar=["z"], alternative="bogus")
        with pytest.raises(ValueError):
            partial_corr(data=df, x="x", y="y", covar="z", x_covar="w")
        with pytest.raises(ValueError):
            partial_corr(data=df, x="x", y="y", x_covar="z", y_covar="w")
        with pytest.raises(AssertionError):
            partial_corr(data=df, x="x", y="x", covar=["z"])
        with pytest.raises(AssertionError):
            partial_corr(data=df, x="z", y="y", covar=["z"])


class TestPairwiseCorrPartial:
    """pairwise_corr with covariates: per-pair missingness, non-estimable
    pairs and multiple-comparison correction."""

    def _nan_data(self):
        rng = np.random.default_rng(11)
        n = 50
        z = rng.normal(size=n)
        df = pd.DataFrame(
            dict(
                a=z + rng.normal(size=n),
                b=0.5 * z + rng.normal(size=n),
                c=rng.normal(size=n),
                z=z,
            )
        )
        df.loc[[1, 5, 9], "a"] = np.nan
        df.loc[[2, 3], "b"] = np.nan
        df.loc[[7, 8, 9, 10], "z"] = np.nan
        return df

    @pytest.mark.parametrize("method", ["pearson", "spearman"])
    def test_pairwise_masks_match_direct_calls(self, method, no_rounding):
        """In pairwise mode each pair uses its own complete observations, and
        every row matches a direct partial_corr call on the same samples."""
        df = self._nan_data()
        out = pairwise_corr(df, columns=["a", "b", "c"], covar=["z"], method=method)
        assert out["n"].nunique() > 1  # different masks -> different n
        for _, row in out.iterrows():
            sub = df[[row["X"], row["Y"], "z"]].dropna()
            ref = partial_corr(data=sub, x=row["X"], y=row["Y"], covar=["z"], method=method)
            assert row["n"] == ref.at[method, "n"]
            assert row["r"] == ref.at[method, "r"]
            assert row["p_unc"] == ref.at[method, "p_val"]
            assert np.allclose(
                np.asarray(row["CI95"], dtype=float),
                np.asarray(ref.at[method, "CI95"], dtype=float),
            )

    def test_listwise_uses_common_samples(self, no_rounding):
        """In listwise mode, deletion is done once on the selected variables
        and covariates; all rows share the same n and control information."""
        df = self._nan_data()
        out = pairwise_corr(df, columns=["a", "b", "c"], covar=["z"], nan_policy="listwise")
        n_common = df[["a", "b", "c", "z"]].dropna().shape[0]
        assert (out["n"] == n_common).all()
        sub = df[["a", "b", "c", "z"]].dropna()
        for _, row in out.iterrows():
            ref = partial_corr(data=sub, x=row["X"], y=row["Y"], covar=["z"])
            assert row["r"] == ref.at["pearson", "r"]
            assert row["p_unc"] == ref.at["pearson", "p_val"]

    @pytest.mark.parametrize("padjust", ["holm", "fdr_bh"])
    def test_padjust_maps_to_correct_rows(self, padjust, no_rounding):
        """The correction family is made of the finite p-values only, and the
        corrected values are mapped back to the correct rows."""
        df = _repro_data()
        out = pairwise_corr(df, columns=["x", "y", "exact"], covar=["z"], padjust=padjust)
        finite = out["p_unc"].dropna()
        expect = multicomp(finite.to_numpy(), method=padjust)[1]
        assert np.allclose(out.loc[finite.index, "p_corr"].to_numpy(), expect)
        # NaN p-values stay NaN after correction
        assert out["p_corr"].isna().equals(out["p_unc"].isna())
        # A batch with several finite p-values: family = all finite p-values
        df2 = self._nan_data()
        out2 = pairwise_corr(df2, columns=["a", "b", "c"], covar=["z"], padjust=padjust)
        expect2 = multicomp(out2["p_unc"].to_numpy(), method=padjust)[1]
        assert np.allclose(out2["p_corr"].to_numpy(), expect2)

    def test_all_pairs_non_estimable(self, no_rounding):
        """When the whole batch is not estimable, the output keeps its columns
        (X, Y, n and the NaN statistics) and the original pair order."""
        rng = np.random.default_rng(41)
        z = rng.normal(size=50)
        df = pd.DataFrame(dict(u=z.copy(), v=2 * z, w=z + 0 * rng.normal(size=50), z=z))
        out = pairwise_corr(df, columns=["u", "v", "w"], covar=["z"], padjust="holm")
        for col in [
            "X",
            "Y",
            "method",
            "covar",
            "alternative",
            "n",
            "r",
            "CI95",
            "p_unc",
            "p_corr",
            "p_adjust",
        ]:
            assert col in out.columns
        assert list(out["X"]) == ["u", "u", "v"]
        assert list(out["Y"]) == ["v", "w", "w"]
        assert (out["n"] == 50).all()
        assert out["r"].isna().all()
        assert out["p_unc"].isna().all()
        assert out["p_corr"].isna().all()

    def test_pair_with_too_few_samples(self, no_rounding):
        """A pair with fewer than 3 complete observations gets a NaN row (with
        its actual n) and does not interrupt the other pairs."""
        df = self._nan_data()
        df.loc[df.index[:48], "a"] = np.nan  # 'a' has only 2 complete rows
        out = pairwise_corr(df, columns=["a", "b", "c"], covar=["z"])
        row_a = out[out["X"] == "a"].iloc[0]
        assert row_a["n"] == 2
        assert np.isnan(row_a["r"])
        assert np.isnan(row_a["p_unc"])
        row_bc = out[out["X"] == "b"].iloc[0]
        assert np.isfinite(row_bc["r"])

    def test_constant_covariate(self, no_rounding):
        """A covariate that is constant (on the whole table or on a pair's
        valid rows) is treated according to the information it provides: none.
        """
        df = self._nan_data()
        df["zc"] = 4.2  # constant covariate on the whole dataframe
        out_const = pairwise_corr(df, columns=["a", "b", "c"], covar=["zc"])
        out_plain = pairwise_corr(df, columns=["a", "b", "c"])
        assert np.allclose(out_const["r"].to_numpy(), out_plain["r"].to_numpy())
        # Covariate constant on the valid rows of one pair only
        df2 = pd.DataFrame(
            dict(
                a=np.random.default_rng(1).normal(size=30),
                b=np.random.default_rng(2).normal(size=30),
            )
        )
        df2["z"] = 1.7
        df2.loc[:4, "z"] = np.arange(5.0)  # z varies only on the first 5 rows
        df2.loc[:4, "b"] = np.nan  # ... which are missing for pair (a, b)
        out = pairwise_corr(df2, columns=["a", "b"], covar=["z"])
        sub = df2[["a", "b", "z"]].dropna()
        assert sub["z"].nunique() == 1
        ref = corr(sub["a"], sub["b"])  # no control information left
        assert out.iloc[0]["r"] == pytest.approx(ref.at["pearson", "r"], abs=1e-12)
        assert out.iloc[0]["p_unc"] == pytest.approx(ref.at["pearson", "p_val"], abs=1e-12)

    def test_pairwise_redundant_covariates(self, no_rounding):
        """Redundant covariates do not change pairwise results either, and the
        rank-deficiency warning is emitted (once per unique mask)."""
        rng = np.random.default_rng(23)
        n = 60
        df = pd.DataFrame({f"v{i}": rng.normal(size=n) for i in range(4)})
        df["z"] = rng.normal(size=n)
        df["z2"] = 2 * df["z"]
        o1 = pairwise_corr(df, columns=[f"v{i}" for i in range(4)], covar=["z"])
        with pytest.warns(UserWarning, match="rank-deficient"):
            o2 = pairwise_corr(df, columns=[f"v{i}" for i in range(4)], covar=["z", "z2"])
        assert np.allclose(o1["r"].to_numpy(), o2["r"].to_numpy())
        assert np.allclose(o1["p_unc"].to_numpy(), o2["p_unc"].to_numpy())

    def test_pairwise_input_errors(self):
        """Input configuration errors still raise in pairwise_corr (no
        catch-all turning an erroneous call into a successful NaN table)."""
        df = _repro_data()
        with pytest.raises(AssertionError):
            pairwise_corr(df, columns=["x", "y"], covar=["not_a_column"])
        with pytest.raises(AssertionError):
            pairwise_corr(df, columns=["x", "y"], covar=["z"], method="kendall")
        with pytest.raises(AssertionError):
            pairwise_corr(df, columns=["x", "y"], covar=["z"], alternative="bogus")
        with pytest.raises(ValueError):
            pairwise_corr(df, columns=["x", "z"], covar=["z"])
