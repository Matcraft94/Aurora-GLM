# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Diagnostics for Zero-Inflated Models.

This module provides statistical tests and diagnostics for evaluating
zero-inflated models:

- Vuong test: Non-nested model comparison (ZIP vs Poisson, ZINB vs NB)
- Score test: Test for zero-inflation against standard count model
- Rootogram: Visual diagnostic for count model fit

References
----------
.. [1] Vuong, Q. H. (1989).
       "Likelihood ratio tests for model selection and non-nested hypotheses."
       Econometrica, 57(2), 307-333.
.. [2] van den Broek, J. (1995).
       "A score test for zero inflation in a Poisson distribution."
       Biometrics, 51(2), 738-743.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.special import gammaln

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["vuong_test", "score_test_zero_inflation", "rootogram"]


def vuong_test(
    ll_model1: float | NDArray,
    ll_model2: float | NDArray,
    n: int,
    n_params1: int | None = None,
    n_params2: int | None = None,
    correction: str = "aic",
) -> dict:
    """Vuong test for non-nested model comparison.

    The Vuong test compares two non-nested models based on their
    log-likelihood contributions. It tests:

        H0: Models are equivalent (E[m_i] = 0)
        H1: Model 1 is better (E[m_i] > 0) or Model 2 is better (E[m_i] < 0)

    where m_i = log[f1(y_i)/f2(y_i)] is the pointwise log-likelihood ratio.

    Parameters
    ----------
    ll_model1 : float or ndarray
        Log-likelihood of model 1. If float, total log-likelihood.
        If array, pointwise log-likelihoods.
    ll_model2 : float or ndarray
        Log-likelihood of model 2
    n : int
        Number of observations
    n_params1 : int, optional
        Number of parameters in model 1 (for correction)
    n_params2 : int, optional
        Number of parameters in model 2 (for correction)
    correction : str
        Correction type: 'none', 'aic', or 'bic'

    Returns
    -------
    dict
        Test results:
        - 'statistic': Vuong test statistic
        - 'p_value': Two-sided p-value
        - 'p_model1_better': One-sided p-value (model 1 better)
        - 'p_model2_better': One-sided p-value (model 2 better)
        - 'conclusion': String interpretation

    Examples
    --------
    >>> from aurora.models.glm import fit_glm
    >>> from aurora.models.zero_inflated import fit_zip
    >>>
    >>> # Fit competing models
    >>> poisson_result = fit_glm(X, y, family='poisson')
    >>> zip_result = fit_zip(X, y)
    >>>
    >>> # Vuong test
    >>> result = vuong_test(
    ...     zip_result.log_likelihood_,
    ...     poisson_result.deviance_ / -2,  # Convert deviance to log-lik
    ...     n=len(y),
    ...     n_params1=len(zip_result.coef_count_) + len(zip_result.coef_inflate_),
    ...     n_params2=len(poisson_result.coef_),
    ... )
    >>> print(result['conclusion'])

    Notes
    -----
    The test statistic is:

        V = √n × m̄ / s_m

    where m̄ is the mean log-likelihood ratio and s_m is its standard deviation.

    Under H0, V ~ N(0, 1).

    The AIC correction adjusts for different model complexities:
        m̄_adj = m̄ - (k1 - k2) / n
    """
    ll1 = np.asarray(ll_model1)
    ll2 = np.asarray(ll_model2)

    # If scalars, can't compute variance - use approximation
    if ll1.ndim == 0:
        ll1 = np.array([float(ll1)])
    if ll2.ndim == 0:
        ll2 = np.array([float(ll2)])

    # Pointwise log-likelihood difference
    m = ll1 - ll2

    if len(m) == 1:
        # Only have totals, need to approximate
        # Assume variance proportional to mean difference
        m_mean = float(m[0]) / n
        # Conservative estimate of variance
        m_var = abs(m_mean) / np.sqrt(n)
    else:
        m_mean = np.mean(m)
        m_var = np.var(m, ddof=1)

    # Apply correction
    if correction == "aic" and n_params1 is not None and n_params2 is not None:
        m_mean -= (n_params1 - n_params2) / n
    elif correction == "bic" and n_params1 is not None and n_params2 is not None:
        m_mean -= (n_params1 - n_params2) * np.log(n) / (2 * n)

    # Test statistic
    if m_var > 0:
        v_stat = np.sqrt(n) * m_mean / np.sqrt(m_var)
    else:
        v_stat = np.sign(m_mean) * np.inf if m_mean != 0 else 0

    # P-values
    p_two_sided = 2 * stats.norm.sf(abs(v_stat))
    p_model1_better = stats.norm.sf(v_stat)  # P(V > v) if model 1 better
    p_model2_better = stats.norm.cdf(v_stat)  # P(V < v) if model 2 better

    # Interpretation
    if p_two_sided > 0.05:
        conclusion = "Models are not significantly different (p > 0.05)"
    elif v_stat > 0:
        conclusion = f"Model 1 is significantly better (p = {p_model1_better:.4f})"
    else:
        conclusion = f"Model 2 is significantly better (p = {p_model2_better:.4f})"

    return {
        "statistic": float(v_stat),
        "p_value": float(p_two_sided),
        "p_model1_better": float(p_model1_better),
        "p_model2_better": float(p_model2_better),
        "mean_diff": float(m_mean),
        "var_diff": float(m_var),
        "conclusion": conclusion,
    }


def score_test_zero_inflation(
    y: NDArray,
    mu: NDArray,
    family: str = "poisson",
    theta: float | None = None,
) -> dict:
    """Score test for zero-inflation.

    Tests H0: No zero-inflation (π = 0) against H1: Zero-inflation (π > 0).

    This is a one-sided score test based on van den Broek (1995) for
    Poisson and extended to Negative Binomial.

    Parameters
    ----------
    y : ndarray
        Observed counts
    mu : ndarray
        Fitted means from standard count model (Poisson or NB)
    family : str
        Count distribution: 'poisson' or 'negativebinomial'
    theta : float, optional
        Dispersion parameter (required for NB)

    Returns
    -------
    dict
        Test results:
        - 'statistic': Score test statistic
        - 'p_value': One-sided p-value
        - 'conclusion': Interpretation

    Examples
    --------
    >>> from aurora.models.glm import fit_glm
    >>>
    >>> # Fit Poisson model
    >>> result = fit_glm(X, y, family='poisson')
    >>>
    >>> # Test for zero-inflation
    >>> test = score_test_zero_inflation(y, result.mu_)
    >>> print(f"Score statistic: {test['statistic']:.2f}")
    >>> print(f"P-value: {test['p_value']:.4f}")

    Notes
    -----
    For Poisson, the score statistic is:

        S = [Σ(I(y=0) - exp(-μ))]² / [Σexp(-μ)(1 - exp(-μ) - μexp(-μ))]

    Under H0, S ~ χ²(1).

    A significant result suggests zero-inflation is present.
    """
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    n = len(y)

    is_zero = y == 0

    if family.lower() == "poisson":
        # P(Y=0 | Poisson) = exp(-μ)
        p0 = np.exp(-mu)

        # Score numerator: observed zeros - expected zeros
        numerator = np.sum(is_zero) - np.sum(p0)

        # Score denominator (variance under H0)
        # Var = Σ p0(1 - p0) - Σ μ*p0*(something)
        # Simplified: Var ≈ Σ p0(1 - p0)
        variance = np.sum(p0 * (1 - p0 - mu * p0))

    elif family.lower() in ("negativebinomial", "negbin", "nb"):
        if theta is None:
            raise ValueError("theta required for negative binomial")

        # P(Y=0 | NB) = (θ/(θ+μ))^θ
        p0 = (theta / (theta + mu)) ** theta

        # Score numerator
        numerator = np.sum(is_zero) - np.sum(p0)

        # Variance under H0 (approximation)
        dp0_dmu = -theta * p0 / (theta + mu)
        variance = np.sum(p0 * (1 - p0))

    else:
        raise ValueError(f"Unknown family: {family}")

    # Ensure positive variance
    variance = max(variance, 1e-10)

    # Score statistic
    score_stat = numerator**2 / variance

    # P-value (chi-squared with 1 df, one-sided)
    p_value = stats.chi2.sf(score_stat, df=1)

    # Interpretation
    if p_value < 0.01:
        conclusion = f"Strong evidence of zero-inflation (p = {p_value:.4f})"
    elif p_value < 0.05:
        conclusion = f"Moderate evidence of zero-inflation (p = {p_value:.4f})"
    elif p_value < 0.10:
        conclusion = f"Weak evidence of zero-inflation (p = {p_value:.4f})"
    else:
        conclusion = f"No significant zero-inflation (p = {p_value:.4f})"

    return {
        "statistic": float(score_stat),
        "p_value": float(p_value),
        "numerator": float(numerator),
        "variance": float(variance),
        "observed_zeros": int(np.sum(is_zero)),
        "expected_zeros": float(np.sum(p0)),
        "conclusion": conclusion,
    }


def rootogram(
    y: NDArray,
    fitted: NDArray,
    family: str = "poisson",
    theta: float | None = None,
    max_count: int | None = None,
    style: str = "hanging",
    ax=None,
):
    """Create rootogram for count model diagnostics.

    A rootogram compares observed vs expected frequencies on a square-root
    scale, making deviations at all count levels visible.

    Parameters
    ----------
    y : ndarray
        Observed counts
    fitted : ndarray
        Fitted means (μ) from count model
    family : str
        Distribution: 'poisson', 'negativebinomial', 'zip', 'zinb'
    theta : float, optional
        Dispersion (for NB/ZINB)
    max_count : int, optional
        Maximum count to display. Default: 95th percentile of y.
    style : str
        'hanging' (bars hang from expected) or 'standing' (bars from zero)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on

    Returns
    -------
    dict
        Rootogram data:
        - 'counts': Count values (0, 1, 2, ...)
        - 'observed': Observed frequencies
        - 'expected': Expected frequencies
        - 'sqrt_observed': Square-root of observed
        - 'sqrt_expected': Square-root of expected
        - 'ax': Matplotlib axes (if plotting)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from aurora.models.zero_inflated import fit_zip, rootogram
    >>>
    >>> result = fit_zip(X, y)
    >>> data = rootogram(y, result.expected_count_, family='zip')
    >>> plt.show()

    Notes
    -----
    In a hanging rootogram:
    - Bars hang from the expected curve
    - Bar bottoms should touch the zero line if model fits well
    - Bars above zero: model underpredicts that count
    - Bars below zero: model overpredicts that count

    This is particularly useful for detecting:
    - Excess zeros (bar at 0 extends above zero line)
    - Overdispersion (systematic pattern in deviations)
    - Poor fit at specific count values
    """
    y = np.asarray(y, dtype=int)
    fitted = np.asarray(fitted, dtype=float)
    n = len(y)

    if max_count is None:
        max_count = int(np.percentile(y, 95))
    max_count = max(max_count, y.max())

    counts = np.arange(max_count + 1)

    # Observed frequencies
    observed = np.bincount(y, minlength=max_count + 1)[: max_count + 1]

    # Expected frequencies
    expected = np.zeros(max_count + 1)

    if family.lower() == "poisson":
        for k in counts:
            # P(Y=k) = exp(-μ) × μ^k / k!
            log_p = -fitted + k * np.log(fitted + 1e-10) - gammaln(k + 1)
            expected[k] = np.sum(np.exp(log_p))

    elif family.lower() in ("negativebinomial", "negbin", "nb"):
        if theta is None:
            raise ValueError("theta required for negative binomial")
        for k in counts:
            log_p = (
                gammaln(k + theta)
                - gammaln(theta)
                - gammaln(k + 1)
                + theta * np.log(theta / (theta + fitted))
                + k * np.log(fitted / (theta + fitted + 1e-10))
            )
            expected[k] = np.sum(np.exp(log_p))

    elif family.lower() == "zip":
        # Need pi from the model - use observed excess zeros as approximation
        p0_poisson = np.exp(-fitted)
        excess_zeros = max(0, observed[0] - np.sum(p0_poisson))
        pi_approx = excess_zeros / n

        for k in counts:
            if k == 0:
                # P(Y=0) = π + (1-π)exp(-μ)
                expected[0] = np.sum(pi_approx + (1 - pi_approx) * np.exp(-fitted))
            else:
                # P(Y=k) = (1-π) × Poisson(k)
                log_p = -fitted + k * np.log(fitted + 1e-10) - gammaln(k + 1)
                expected[k] = np.sum((1 - pi_approx) * np.exp(log_p))

    elif family.lower() == "zinb":
        if theta is None:
            raise ValueError("theta required for ZINB")
        # Similar approximation for pi
        p0_nb = (theta / (theta + fitted)) ** theta
        excess_zeros = max(0, observed[0] - np.sum(p0_nb))
        pi_approx = excess_zeros / n

        for k in counts:
            if k == 0:
                expected[0] = np.sum(pi_approx + (1 - pi_approx) * p0_nb)
            else:
                log_p = (
                    gammaln(k + theta)
                    - gammaln(theta)
                    - gammaln(k + 1)
                    + theta * np.log(theta / (theta + fitted))
                    + k * np.log(fitted / (theta + fitted + 1e-10))
                )
                expected[k] = np.sum((1 - pi_approx) * np.exp(log_p))

    else:
        raise ValueError(f"Unknown family: {family}")

    # Square root transformation
    sqrt_obs = np.sqrt(observed)
    sqrt_exp = np.sqrt(expected)

    result = {
        "counts": counts,
        "observed": observed,
        "expected": expected,
        "sqrt_observed": sqrt_obs,
        "sqrt_expected": sqrt_exp,
    }

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        width = 0.8

        if style == "hanging":
            # Bars hang from expected curve
            bar_bottoms = sqrt_exp - sqrt_obs
            ax.bar(counts, sqrt_obs, width, bottom=bar_bottoms, color="steelblue", alpha=0.7)
            ax.plot(counts, sqrt_exp, "o-", color="red", markersize=4, label="Expected")
            ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        else:
            # Standing bars from zero
            ax.bar(counts, sqrt_obs, width, color="steelblue", alpha=0.7, label="Observed")
            ax.plot(counts, sqrt_exp, "o-", color="red", markersize=4, label="Expected")

        ax.set_xlabel("Count")
        ax.set_ylabel("√Frequency")
        ax.set_title(f"Rootogram ({family.upper()})")
        ax.legend()

        result["ax"] = ax

    except ImportError:
        pass

    return result
