# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Analysis of variance (analysis of deviance) for GLM and GAM models.

This module provides ANOVA functionality for analyzing model effects
and comparing nested models.

Single-model table
------------------
For a single fitted GLM, ``anova_glm`` reports partial (Type III-style)
Wald tests per coefficient:

    W_j = β̂_j² / Var(β̂_j)

where Var(β̂_j) is the diagonal of the model covariance matrix (already
scaled by the dispersion estimate φ̂). For Gaussian models with
``test='F'`` the reference distribution is F(1, df_resid) (equivalent to
the squared t-statistic, matching ``statsmodels.stats.anova_lm`` with
``typ=3``); otherwise W_j ~ χ²₁ asymptotically.

Model comparison
----------------
For several nested models, ``anova_glm`` reports sequential deviance
differences (analysis of deviance, R ``anova.glm`` convention):

- ``test='Chisq'`` / ``'LRT'``: ΔD ~ χ²_Δdf — valid for nested GLMs of
  the same family with known dispersion (Poisson, Binomial), where the
  deviance difference equals the likelihood-ratio statistic.
- ``test='F'``: F = (ΔD/Δdf) / φ̂ with φ̂ taken from the largest model —
  the convention for Gaussian models with estimated dispersion. For
  non-Gaussian families ``test='F'`` falls back to the χ² test with a
  warning.

Examples
--------
>>> from aurora.inference.anova import anova_glm, likelihood_ratio_test
>>>
>>> # Single-model Wald table (Type III-style)
>>> anova_glm(result, type=3)
>>>
>>> # Compare nested models
>>> anova_glm(reduced_model, full_model, test="Chisq")
>>> likelihood_ratio_test(reduced_model, full_model)

References
----------
.. [1] Fox, J. (2015). Applied Regression Analysis and GLMs.
.. [2] Venables, W.N. & Ripley, B.D. (2002). Modern Applied Statistics with S.
.. [3] Self, S.G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
       likelihood estimators and likelihood ratio test when some parameters
       are on the boundary of the parameter space. JASA 82, 605-610.
.. [4] Stram, D.O. & Lee, J.W. (1994). Variance components testing in the
       longitudinal mixed effects model. Biometrics 50, 1171-1177.
.. [5] Pinheiro, J.C. & Bates, D.M. (2000). Mixed-Effects Models in S and
       S-PLUS. Springer. §2.4 (PQL likelihood is conditional).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy import stats


@dataclass
class ANOVAResult:
    """Result of an ANOVA test.

    Attributes
    ----------
    df : ndarray
        Degrees of freedom for each source.
    ss : ndarray
        Sum of squares for each source. For single-model Wald tables this
        holds the per-coefficient Wald statistics (χ² scale); for model
        comparisons it holds the (non-negative) deviance differences.
    ms : ndarray
        Mean squares (SS / df).
    f_statistic : ndarray
        Test statistics: F values when ``test='F'``, χ² values otherwise.
    p_value : ndarray
        P-values for each test.
    source : list of str
        Names of sources of variation.
    anova_type : int
        Type of ANOVA performed (1, 2, or 3).
    residual_df : int
        Residual degrees of freedom of the (largest) model.
    residual_ss : float
        Residual deviance (Gaussian: residual sum of squares) of the
        (largest) model.
    test : str
        Reference distribution used for the p-values ('F' or 'Chisq').
    """

    df: np.ndarray
    ss: np.ndarray
    ms: np.ndarray
    f_statistic: np.ndarray
    p_value: np.ndarray
    source: list[str]
    anova_type: int
    residual_df: int
    residual_ss: float
    test: str = "F"

    def __repr__(self) -> str:
        return f"ANOVAResult(type={self.anova_type}, sources={self.source})"

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        """Return formatted ANOVA table."""
        lines = []
        sep = "=" * 75

        stat_label = "F" if self.test == "F" else "Chi2"
        p_label = "Pr(>F)" if self.test == "F" else "Pr(>Chi)"

        lines.append(sep)
        lines.append(f"{'ANOVA Table (Type ' + str(self.anova_type) + ')':^75}")
        lines.append(sep)
        lines.append(
            f"{'Source':>15} {'Df':>8} {'Sum Sq':>12} {'Mean Sq':>12} "
            f"{stat_label:>10} {p_label:>12}"
        )
        lines.append("-" * 75)

        for i, src in enumerate(self.source):
            p_str = (
                f"{self.p_value[i]:.4e}" if self.p_value[i] < 0.0001 else f"{self.p_value[i]:.4f}"
            )
            sig = ""
            if self.p_value[i] < 0.001:
                sig = " ***"
            elif self.p_value[i] < 0.01:
                sig = " **"
            elif self.p_value[i] < 0.05:
                sig = " *"
            elif self.p_value[i] < 0.1:
                sig = " ."

            lines.append(
                f"{src:>15} {int(self.df[i]):>8} {self.ss[i]:>12.4f} "
                f"{self.ms[i]:>12.4f} {self.f_statistic[i]:>10.4f} {p_str:>12}{sig}"
            )

        # Residuals row
        residual_ms = self.residual_ss / self.residual_df
        lines.append(
            f"{'Residuals':>15} {self.residual_df:>8} {self.residual_ss:>12.4f} "
            f"{residual_ms:>12.4f}"
        )

        lines.append(sep)
        lines.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "df": self.df.tolist(),
            "ss": self.ss.tolist(),
            "ms": self.ms.tolist(),
            "f_statistic": self.f_statistic.tolist(),
            "p_value": self.p_value.tolist(),
            "type": self.anova_type,
            "residual_df": self.residual_df,
            "residual_ss": self.residual_ss,
            "test": self.test,
        }


@dataclass
class LRTResult:
    """Result of a likelihood ratio test.

    Attributes
    ----------
    statistic : float
        Chi-squared test statistic (2 * (LL_full - LL_reduced)).
    df : int
        Degrees of freedom (difference in parameters).
    p_value : float
        P-value from chi-squared distribution.
    model_names : tuple of str
        Names of the compared models.
    boundary_conditions : list of str
        Variance parameters detected at or near boundary.
    boundary_correction_applied : bool
        Whether Self & Liang (1987) mixture correction was applied.
    """

    statistic: float
    df: int
    p_value: float
    model_names: tuple[str, str]
    ll_reduced: float
    ll_full: float
    boundary_conditions: list[str] | None = None
    boundary_correction_applied: bool = False

    def __repr__(self) -> str:
        return f"LRTResult(χ²={self.statistic:.4f}, df={self.df}, p={self.p_value:.4e})"

    def summary(self) -> str:
        """Return formatted LRT summary."""
        lines = []
        sep = "=" * 60

        lines.append(sep)
        lines.append(f"{'Likelihood Ratio Test':^60}")
        lines.append(sep)
        lines.append(f"Model 1 (reduced): {self.model_names[0]}")
        lines.append(f"Model 2 (full):    {self.model_names[1]}")
        lines.append("-" * 60)
        lines.append(f"Log-Lik (reduced): {self.ll_reduced:>15.4f}")
        lines.append(f"Log-Lik (full):    {self.ll_full:>15.4f}")
        lines.append("-" * 60)
        lines.append(f"Chi-squared:       {self.statistic:>15.4f}")
        lines.append(f"Df:                {self.df:>15}")
        lines.append(f"P-value:           {self.p_value:>15.4e}")
        lines.append(sep)

        if self.p_value < 0.05:
            lines.append("The full model is significantly better (p < 0.05).")
        else:
            lines.append("No significant difference between models.")

        return "\n".join(lines)


def anova_glm(
    *models: Any,
    type: Literal[1, 2, 3] = 3,
    test: Literal["F", "Chisq", "LRT"] = "F",
) -> ANOVAResult:
    """Perform ANOVA on one or more GLM models.

    Parameters
    ----------
    *models : ModelResult
        One or more fitted model results. With one model, produces a table
        of partial Wald tests per coefficient. With multiple models,
        performs a sequential analysis of deviance (models are sorted by
        number of parameters).
    type : {1, 2, 3}, default=3
        Type of sum of squares. Only ``type=3`` (partial tests) is
        supported for a single model: it requires no refits and is
        order-independent. For multiple models the comparison is always
        sequential (Type I style).
    test : {'F', 'Chisq', 'LRT'}, default='F'
        Test statistic / reference distribution:

        - ``'F'``: F test. For a single Gaussian model this is the squared
          t-statistic with F(1, df_resid) reference. For model comparison
          of Gaussian models, F = (ΔD/Δdf)/φ̂ with φ̂ from the largest
          model (R ``anova.glm`` convention). For non-Gaussian families it
          falls back to the χ² test with a warning.
        - ``'Chisq'`` / ``'LRT'``: χ² test on the Wald statistics (single
          model) or on the deviance differences (model comparison). The
          deviance difference equals the likelihood-ratio statistic for
          families with unit dispersion (Poisson, Binomial).

    Returns
    -------
    result : ANOVAResult
        ANOVA result with table and statistics.

    Examples
    --------
    >>> from aurora import fit_glm, Gaussian
    >>> from aurora.inference.anova import anova_glm
    >>>
    >>> result = fit_glm(X, y, family=Gaussian())
    >>> anova_glm(result, type=3)

    >>> # Compare nested models
    >>> m1 = fit_glm(X[:, :2], y)
    >>> m2 = fit_glm(X, y)
    >>> anova_glm(m1, m2, test="Chisq")

    Notes
    -----
    Type III (partial) tests: each coefficient is tested adjusted for all
    other terms via its Wald statistic. Sequential (Type I) tests for a
    single model would require refitting sub-models and are not currently
    implemented.
    """
    if len(models) == 0:
        raise ValueError("At least one model is required.")

    if len(models) == 1:
        # Single model ANOVA
        return _anova_single(models[0], type=type, test=test)
    else:
        # Model comparison
        return _anova_compare(models, test=test)


def _anova_single(
    model: Any,
    type: int,
    test: str,
) -> ANOVAResult:
    """Partial (Type III-style) Wald tests per coefficient of one model.

    The Wald statistic for coefficient j is

        W_j = β̂_j² / Var(β̂_j)

    with Var(β̂_j) taken from the model covariance matrix, which already
    incorporates the dispersion estimate φ̂. For Gaussian models with
    ``test='F'`` the reference distribution is F(1, df_resid) (the squared
    t-statistic); otherwise W_j ~ χ²₁.
    """
    if type != 3:
        raise NotImplementedError(
            f"Single-model ANOVA supports type=3 only (partial Wald tests, "
            f"no refits); got type={type}. Use multiple nested models for "
            "sequential (Type I) comparisons."
        )

    # Extract coefficients (including intercept when present)
    if hasattr(model, "coef_"):
        coef = np.atleast_1d(np.asarray(model.coef_, dtype=float))
        intercept = getattr(model, "intercept_", None)
    elif hasattr(model, "fixed_effects_"):
        coef = np.atleast_1d(np.asarray(model.fixed_effects_, dtype=float))
        intercept = None
    else:
        raise ValueError("Cannot extract coefficients from model.")

    if intercept is not None:
        coef_full = np.concatenate(([float(intercept)], coef))
        sources = ["intercept"] + [f"X{i}" for i in range(len(coef))]
    else:
        coef_full = coef
        sources = [f"X{i}" for i in range(len(coef))]

    n_params = len(coef_full)

    # Covariance matrix of the full parameter vector (includes dispersion).
    cov = _get_coef_covariance(model, n_params)
    var = np.clip(np.diag(cov), 1e-300, None)

    # Wald statistics: β̂_j² / Var(β̂_j)
    wald = coef_full**2 / var

    n_obs = _get_n_obs(model)
    rank = getattr(model, "rank_", None) or n_params
    residual_df = max(int(n_obs - rank), 0)

    # Residual deviance (Gaussian: residual sum of squares)
    residual_dev = _get_deviance(model)
    if residual_dev is None:
        residual_dev = np.nan

    gaussian = _is_gaussian_family(model)

    df = np.ones(n_params, dtype=int)

    if test == "F":
        if gaussian and residual_df > 0:
            # Squared t-statistics with F(1, df_resid) reference.
            f_stat = wald
            p_values = stats.f.sf(f_stat, 1, residual_df)
            dispersion = _get_dispersion(model, residual_dev, residual_df)
            ss = wald * dispersion
            test_used = "F"
        else:
            if gaussian:
                warnings.warn(
                    "F test requested but residual degrees of freedom are "
                    "zero; falling back to the chi-squared test.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "F test is only appropriate for Gaussian family with "
                    "estimated dispersion; using the chi-squared test "
                    "(R anova.glm convention).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            f_stat = wald
            p_values = stats.chi2.sf(wald, 1)
            ss = wald.copy()
            test_used = "Chisq"
    elif test in ("Chisq", "LRT"):
        f_stat = wald
        p_values = stats.chi2.sf(wald, 1)
        ss = wald.copy()
        test_used = "Chisq"
    else:
        raise ValueError(f"Unknown test: {test!r}. Valid options: 'F', 'Chisq', 'LRT'.")

    ms = ss / df

    return ANOVAResult(
        df=df,
        ss=np.asarray(ss, dtype=float),
        ms=np.asarray(ms, dtype=float),
        f_statistic=np.asarray(f_stat, dtype=float),
        p_value=np.asarray(p_values, dtype=float),
        source=sources,
        anova_type=type,
        residual_df=residual_df,
        residual_ss=float(residual_dev),
        test=test_used,
    )


def _anova_compare(
    models: tuple[Any, ...],
    test: str,
) -> ANOVAResult:
    """Sequential analysis of deviance for nested models.

    Models are sorted by number of parameters and compared pairwise. The
    table reports deviance differences ΔD = D_reduced − D_full (truncated
    at zero with a warning when negative beyond numerical noise).

    - ``test='Chisq'``/``'LRT'``: ΔD ~ χ²_Δdf. Valid for nested GLMs of
      the same family with unit dispersion (Poisson, Binomial), where ΔD
      equals the likelihood-ratio statistic.
    - ``test='F'``: F = (ΔD/Δdf)/φ̂ ~ F(Δdf, df_resid) with φ̂ from the
      largest model — appropriate for Gaussian models with estimated
      dispersion (R ``anova.glm`` convention). Non-Gaussian families fall
      back to the χ² test with a warning.
    """
    if test not in ("F", "Chisq", "LRT"):
        raise ValueError(f"Unknown test: {test!r}. Valid options: 'F', 'Chisq', 'LRT'.")

    # Sort models by number of parameters
    model_info = []
    for i, m in enumerate(models):
        n_params = _get_n_params(m)
        dev = _get_deviance(m)
        model_info.append((n_params, dev, m, f"Model {i + 1}"))

    model_info.sort(key=lambda x: x[0])

    largest_params, largest_dev, largest_model, _ = model_info[-1]

    n_obs = _get_n_obs(largest_model)
    residual_df = max(int(n_obs - largest_params), 0)

    if largest_dev is None:
        # Fall back to a pure log-likelihood-ratio comparison
        return _anova_compare_loglik(model_info, residual_df=residual_df)

    gaussian = _is_gaussian_family(largest_model)
    use_f = test == "F" and gaussian and residual_df > 0
    if test == "F" and not use_f:
        warnings.warn(
            "F test is only appropriate for Gaussian family with estimated "
            "dispersion; using the chi-squared test on deviance differences "
            "(R anova.glm convention).",
            RuntimeWarning,
            stacklevel=2,
        )

    dispersion = _get_dispersion(largest_model, largest_dev, residual_df)

    sources = []
    df_list: list[int] = []
    ddev_list: list[float] = []

    for i in range(1, len(model_info)):
        prev_params, prev_dev, _, prev_name = model_info[i - 1]
        curr_params, curr_dev, _, curr_name = model_info[i]

        if prev_dev is None or curr_dev is None:
            # A nested model without deviance info: fall back to LRT.
            return _anova_compare_loglik(model_info, residual_df=residual_df)

        df_diff = int(curr_params - prev_params)
        ddev = float(prev_dev - curr_dev)

        tol = 1e-8 * max(1.0, abs(float(prev_dev)), abs(float(curr_dev)))
        if ddev < -tol:
            warnings.warn(
                f"Deviance of the larger model exceeds that of the smaller "
                f"model by {-ddev:.4g} ({curr_name} vs {prev_name}); models "
                "may not be nested. Truncating the statistic at zero.",
                RuntimeWarning,
                stacklevel=2,
            )
        ddev = max(0.0, ddev)

        sources.append(f"{curr_name} vs {prev_name}")
        df_list.append(df_diff)
        ddev_list.append(ddev)

    df = np.array(df_list, dtype=int)
    ss = np.array(ddev_list, dtype=float)
    ms = ss / np.maximum(df, 1)

    if use_f:
        f_stat = np.where(df > 0, ms / dispersion, np.nan)
        p_values = np.array(
            [
                stats.f.sf(f, d, residual_df) if not np.isnan(f) and d > 0 else np.nan
                for f, d in zip(f_stat, df, strict=False)
            ]
        )
        test_used = "F"
    else:
        f_stat = np.where(df > 0, ss, np.nan)
        p_values = np.array(
            [
                stats.chi2.sf(s, d) if not np.isnan(s) and d > 0 else np.nan
                for s, d in zip(ss, df, strict=False)
            ]
        )
        test_used = "Chisq"

    return ANOVAResult(
        df=df,
        ss=ss,
        ms=ms,
        f_statistic=f_stat,
        p_value=p_values,
        source=sources,
        anova_type=1,  # Sequential
        residual_df=residual_df,
        residual_ss=float(largest_dev),
        test=test_used,
    )


def _anova_compare_loglik(
    model_info: list[tuple[int, Any, Any, str]],
    residual_df: int,
) -> ANOVAResult:
    """Sequential comparison based on log-likelihoods (χ² test).

    Used when the models do not expose a deviance; the statistic is the
    likelihood-ratio statistic 2·(ℓ_full − ℓ_reduced).
    """
    ll_info = []
    for n_params, _, m, name in model_info:
        ll = _get_loglik(m)
        if np.isnan(ll):
            raise ValueError(
                "Cannot compare models: neither deviance nor log-likelihood "
                f"is available for {name}."
            )
        ll_info.append((n_params, ll, name))

    sources = []
    df_list = []
    stat_list = []

    for i in range(1, len(ll_info)):
        prev_params, prev_ll, prev_name = ll_info[i - 1]
        curr_params, curr_ll, curr_name = ll_info[i]

        df_diff = int(curr_params - prev_params)
        stat = 2.0 * (curr_ll - prev_ll)

        tol = 1e-8 * max(1.0, abs(prev_ll), abs(curr_ll))
        if stat < -tol:
            warnings.warn(
                f"Log-likelihood of the larger model is smaller "
                f"({curr_name} vs {prev_name}); models may not be nested. "
                "Truncating the statistic at zero.",
                RuntimeWarning,
                stacklevel=2,
            )
        stat = max(0.0, stat)

        sources.append(f"{curr_name} vs {prev_name}")
        df_list.append(df_diff)
        stat_list.append(stat)

    df = np.array(df_list, dtype=int)
    ss = np.array(stat_list, dtype=float)
    ms = ss / np.maximum(df, 1)
    f_stat = np.where(df > 0, ss, np.nan)
    p_values = np.array(
        [
            stats.chi2.sf(s, d) if not np.isnan(s) and d > 0 else np.nan
            for s, d in zip(ss, df, strict=False)
        ]
    )

    return ANOVAResult(
        df=df,
        ss=ss,
        ms=ms,
        f_statistic=f_stat,
        p_value=p_values,
        source=sources,
        anova_type=1,
        residual_df=residual_df,
        residual_ss=np.nan,
        test="Chisq",
    )


def likelihood_ratio_test(
    model_reduced: Any,
    model_full: Any,
    *,
    names: tuple[str, str] | None = None,
) -> LRTResult:
    """Perform likelihood ratio test comparing two nested models.

    Parameters
    ----------
    model_reduced : ModelResult
        The simpler (nested) model.
    model_full : ModelResult
        The more complex model.
    names : tuple of str, optional
        Names for the models in output.

    Returns
    -------
    result : LRTResult
        Likelihood ratio test result.

    Examples
    --------
    >>> from aurora.inference.anova import likelihood_ratio_test
    >>>
    >>> lrt = likelihood_ratio_test(reduced_model, full_model)
    >>> print(lrt)

    Notes
    -----
    The test statistic is:

    .. math::
        \\chi^2 = 2 (\\ell_{full} - \\ell_{reduced})

    This follows a chi-squared distribution with df equal to the
    difference in the number of parameters. The statistic is truncated at
    zero; a warning is issued when the raw difference is negative beyond
    numerical noise (the models are then likely not nested).

    **Boundary correction.** When a variance component is detected at (or
    numerically near) the boundary of the parameter space (e.g. σ² ≈ 0),
    the null distribution is a 50:50 mixture of χ²_df and χ²_{df-1}
    (Self & Liang 1987, case 5; Stram & Lee 1994). For df = 1 the second
    component is χ²₀, a point mass at zero, whose survival function is 0
    for any positive statistic, giving p = 0.5·P(χ²₁ > t). The correction
    is applied whenever *any* variance parameter of *either* model is at
    the boundary; this is conservative when the boundary parameter is not
    the one being tested.

    **GAMM caveats.** Two warnings apply to ``GAMMResult`` inputs:

    - Non-Gaussian GAMMs are fitted by PQL, whose log-likelihood is a
      *conditional* (quasi-)likelihood given the variance components, not
      a marginal likelihood. Likelihood-ratio tests on it are not valid
      for comparing fixed effects (Pinheiro & Bates 2000, §2.4).
    - Gaussian GAMMs report the REML log-likelihood, which is not
      comparable between models with different fixed-effects structures.
    """
    # Get log-likelihoods
    ll_reduced = _get_loglik(model_reduced)
    ll_full = _get_loglik(model_full)

    if np.isnan(ll_reduced) or np.isnan(ll_full):
        raise ValueError("Both models must have log-likelihood values.")

    # Get number of parameters
    df_reduced = _get_n_params(model_reduced)
    df_full = _get_n_params(model_full)

    df = df_full - df_reduced
    if df < 0:
        raise ValueError("Full model must have at least as many parameters as reduced model.")

    _warn_likelihood_caveats(model_reduced, model_full, df)

    if df == 0:
        # Same model — null result
        model_names = names or ("Reduced", "Full")
        return LRTResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            model_names=model_names,
            ll_reduced=ll_reduced,
            ll_full=ll_full,
            boundary_correction_applied=False,
        )

    # Compute test statistic (truncated at zero; warn if truly negative)
    statistic = 2 * (ll_full - ll_reduced)
    tol = 1e-8 * max(1.0, abs(ll_full), abs(ll_reduced))
    if statistic < -tol:
        warnings.warn(
            f"Full model has a smaller log-likelihood (Δ = {statistic / 2:.4g}); "
            "the models are likely not nested or the likelihoods are not "
            "comparable. Truncating the LRT statistic at zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    statistic = max(0.0, statistic)

    # Check for boundary conditions (Self & Liang, 1987). Conservative:
    # the correction is applied if any variance parameter of either model
    # is at the boundary, not only the parameter being tested.
    boundary_full = _detect_boundary_conditions(model_full)
    boundary_reduced = _detect_boundary_conditions(model_reduced)
    boundary_params = boundary_full + boundary_reduced

    if boundary_params:
        # Self & Liang (1987, case 5) / Stram & Lee (1994): the null
        # distribution is a 50:50 mixture of χ²_df and χ²_{df-1}.
        p_df = stats.chi2.sf(statistic, df)
        if df > 1:
            p_df_minus_1 = stats.chi2.sf(statistic, df - 1)
        else:
            # χ²₀ is a point mass at 0: its survival function is 1 at
            # statistic == 0 and 0 for any statistic > 0.
            p_df_minus_1 = 1.0 if statistic <= 0 else 0.0
        p_value = 0.5 * p_df + 0.5 * p_df_minus_1
        boundary_correction_applied = True
    else:
        # Standard chi-squared test
        p_value = stats.chi2.sf(statistic, df)
        boundary_correction_applied = False

    # Model names
    if names is None:
        names = ("Reduced", "Full")

    return LRTResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        model_names=names,
        ll_reduced=ll_reduced,
        ll_full=ll_full,
        boundary_conditions=boundary_params if boundary_params else None,
        boundary_correction_applied=boundary_correction_applied,
    )


def _warn_likelihood_caveats(model_reduced: Any, model_full: Any, df: int) -> None:
    """Warn about PQL/REML likelihood caveats for GAMM LRTs."""
    try:
        from ...models.gamm.fitting import GAMMResult
    except ImportError:  # pragma: no cover - defensive
        return

    models = [m for m in (model_reduced, model_full) if isinstance(m, GAMMResult)]
    if not models:
        return

    families = {str(m.family).lower() for m in models}
    non_gaussian = any(f not in ("gaussian", "normal") for f in families)

    if non_gaussian:
        warnings.warn(
            "Likelihood ratio test uses a PQL (conditional, quasi-) "
            "log-likelihood from a non-Gaussian GAMM. PQL likelihoods are "
            "not marginal likelihoods: LRTs on them are not valid for "
            "comparing fixed effects (Pinheiro & Bates 2000, §2.4).",
            RuntimeWarning,
            stacklevel=3,
        )
    elif df > 0:
        warnings.warn(
            "Likelihood ratio test uses REML log-likelihoods from Gaussian "
            "GAMMs with different fixed-effects structures. REML likelihoods "
            "are not comparable across different fixed effects; refit with "
            "ML or interpret the p-value with caution.",
            RuntimeWarning,
            stacklevel=3,
        )


def _detect_boundary_conditions(model: Any, threshold: float = 1e-10) -> list[str]:
    """Detect variance components at or near the boundary (e.g., σ² ≈ 0).

    Supports both the ``variance_components_``/``residual_variance_``
    naming and the ``GAMMResult`` naming (``variance_components``,
    ``residual_variance``, without trailing underscore).

    Parameters
    ----------
    model : ModelResult
        A fitted model.
    threshold : float, default=1e-10
        Below this value, a variance parameter is considered at boundary.

    Returns
    -------
    boundary_params : list of str
        Names of parameters detected at the boundary.
    """
    boundary_params = []

    variance_components = getattr(model, "variance_components_", None)
    if variance_components is None:
        variance_components = getattr(model, "variance_components", None)

    if variance_components:
        if isinstance(variance_components, dict):
            for name, value in variance_components.items():
                value_arr = np.asarray(value, dtype=float)
                diag = np.diag(value_arr) if value_arr.ndim == 2 else value_arr.ravel()
                if np.any(np.abs(diag) < threshold):
                    boundary_params.append(name)
        elif isinstance(variance_components, (list, tuple, np.ndarray)):
            for i, val in enumerate(variance_components):
                val_arr = np.asarray(val, dtype=float)
                diag = np.diag(val_arr) if val_arr.ndim == 2 else val_arr.ravel()
                if np.any(np.abs(diag) < threshold):
                    boundary_params.append(f"variance_component_{i}")

    residual_variance = getattr(model, "residual_variance_", None)
    if residual_variance is None:
        residual_variance = getattr(model, "residual_variance", None)
    if residual_variance is not None and abs(residual_variance) < threshold:
        boundary_params.append("residual_variance")

    re_variance = getattr(model, "random_effects_variance_", None)
    if re_variance is None:
        re_variance = getattr(model, "random_effects_variance", None)
    if re_variance is not None:
        for i, var in enumerate(re_variance):
            var_arr = np.asarray(var, dtype=float)
            diag = np.diag(var_arr) if var_arr.ndim == 2 else var_arr.ravel()
            if np.any(np.abs(diag) < threshold):
                boundary_params.append(f"re_{i}")

    if hasattr(model, "psi"):
        psi = np.asarray(model.psi)
        if np.any(np.abs(psi) < threshold):
            boundary_params.append("psi")

    return boundary_params


def _get_loglik(model: Any) -> float:
    """Extract log-likelihood from model."""
    if hasattr(model, "log_likelihood_"):
        return float(model.log_likelihood_)
    if hasattr(model, "log_likelihood"):
        return float(model.log_likelihood)
    if hasattr(model, "loglik"):
        return float(model.loglik)
    if hasattr(model, "llf"):
        return float(model.llf)
    return np.nan


def _get_n_params(model: Any) -> int:
    """Get number of (fixed-effects) parameters from model."""
    if hasattr(model, "df_model"):
        return int(model.df_model) + 1
    if hasattr(model, "coef_"):
        n = len(np.atleast_1d(model.coef_))
        if getattr(model, "intercept_", None) is not None:
            n += 1
        return n
    if hasattr(model, "fixed_effects_"):
        return len(np.atleast_1d(model.fixed_effects_))
    if hasattr(model, "beta_parametric"):
        return len(np.atleast_1d(model.beta_parametric))
    if hasattr(model, "coefficients"):
        return len(np.atleast_1d(model.coefficients))
    return 0


def _get_n_obs(model: Any) -> int:
    """Get number of observations from model."""
    for attr in ("n_obs_", "n_obs", "n_observations", "nobs"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)
    for attr in ("mu_", "_y", "residuals", "fitted_values"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(np.asarray(value, dtype=float).shape[0])
    raise ValueError("Cannot determine the number of observations from the model result.")


def _get_deviance(model: Any) -> float | None:
    """Extract the residual deviance (Gaussian: RSS) from a model."""
    deviance = getattr(model, "deviance_", None)
    if deviance is not None:
        return float(deviance)
    deviance = getattr(model, "deviance", None)
    if deviance is not None and np.isscalar(deviance):
        return float(cast(Any, deviance))
    residuals = getattr(model, "residuals", None)
    if residuals is not None:
        # Response residuals; their sum of squares is the Gaussian deviance.
        return float(np.sum(np.asarray(residuals, dtype=float) ** 2))
    return None


def _get_dispersion(model: Any, deviance: float | None, residual_df: int) -> float:
    """Dispersion estimate φ̂: model attribute, else deviance / df_resid."""
    dispersion = getattr(model, "dispersion_", None)
    if dispersion is not None:
        return float(dispersion)
    if deviance is not None and residual_df > 0:
        return float(deviance) / residual_df
    return 1.0


def _get_coef_covariance(model: Any, n_params: int) -> np.ndarray:
    """Extract the covariance matrix of the full parameter vector."""
    cov = getattr(model, "coef_cov_", None)
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.shape == (n_params, n_params):
            return cov
        raise ValueError(
            f"Coefficient covariance has shape {cov.shape}, expected ({n_params}, {n_params})."
        )

    # Fall back to diagonal covariance from standard errors
    se = getattr(model, "std_errors_", None)
    if se is not None:
        se = np.atleast_1d(np.asarray(se, dtype=float))
        intercept_se = getattr(model, "intercept_std_error_", None)
        if intercept_se is not None:
            se = np.concatenate(([float(intercept_se)], se))
        if len(se) == n_params:
            return np.diag(np.clip(se, 1e-150, None) ** 2)

    raise ValueError(
        "Cannot extract a coefficient covariance matrix from the model "
        "(need `coef_cov_` or `std_errors_`)."
    )


def _is_gaussian_family(model: Any) -> bool:
    """Detect a Gaussian/Normal family from the model result."""
    family = getattr(model, "family", None)
    if family is None:
        # Without family information, assume Gaussian when only residuals
        # are available (least-squares style results).
        return not hasattr(model, "deviance_")
    if isinstance(family, str):
        name = family.lower()
    else:
        name = type(family).__name__.lower()
    return "gaussian" in name or "normal" in name


# Alias for convenience
lrt = likelihood_ratio_test
anova = anova_glm


__all__ = [
    "ANOVAResult",
    "LRTResult",
    "anova_glm",
    "anova",
    "likelihood_ratio_test",
    "lrt",
]
