# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Result classes for Bayesian inference.

This module provides result objects that store posterior samples
and provide methods for posterior summaries and predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["BayesianGLMResult", "BayesianGAMResult"]


@dataclass
class BayesianGLMResult:
    """Results from Bayesian GLM fitting.

    This class stores posterior samples from MCMC and provides
    methods for summarizing the posterior and making predictions.

    Attributes
    ----------
    posterior_samples_ : dict
        Dictionary mapping parameter names to sample arrays.
        Each array has shape (n_chains, n_samples, ...).
    family : str
        Distribution family used
    link : str
        Link function used
    n_samples_ : int
        Number of posterior samples per chain
    n_chains_ : int
        Number of MCMC chains
    n_obs_ : int
        Number of observations in training data
    n_features_ : int
        Number of features (including intercept)
    X_ : ndarray
        Design matrix used for fitting
    y_ : ndarray
        Response variable used for fitting
    backend_ : str
        Backend used ('numpyro' or 'pymc')

    Examples
    --------
    >>> result = fit_glm_bayes(X, y, family='poisson')
    >>> result.summary()
    >>> result.credible_intervals(0.95)
    >>> predictions = result.predict(X_new)
    """

    posterior_samples_: dict[str, NDArray]
    family: str
    link: str
    n_samples_: int
    n_chains_: int
    n_obs_: int
    n_features_: int
    X_: NDArray
    y_: NDArray
    backend_: str

    # Diagnostics (computed lazily)
    _r_hat: dict[str, NDArray] | None = field(default=None, repr=False)
    _ess: dict[str, NDArray] | None = field(default=None, repr=False)

    @property
    def coef_samples_(self) -> NDArray:
        """Posterior samples for coefficients, shape (n_samples_total, n_features)."""
        beta = self.posterior_samples_.get("beta", self.posterior_samples_.get("coef"))
        if beta is None:
            raise KeyError("No coefficient samples found in posterior")
        # Flatten chains: (n_chains, n_samples, p) -> (n_chains * n_samples, p)
        if beta.ndim == 3:
            return beta.reshape(-1, beta.shape[-1])
        return beta

    @property
    def coef_(self) -> NDArray:
        """Posterior mean of coefficients."""
        return np.mean(self.coef_samples_, axis=0)

    @property
    def coef_std_(self) -> NDArray:
        """Posterior standard deviation of coefficients."""
        return np.std(self.coef_samples_, axis=0)

    @property
    def scale_samples_(self) -> NDArray | None:
        """Posterior samples for scale parameter (if applicable)."""
        for key in ["sigma", "scale", "phi"]:
            if key in self.posterior_samples_:
                samples = self.posterior_samples_[key]
                if samples.ndim >= 2:
                    return samples.reshape(-1)
                return samples
        return None

    @property
    def scale_(self) -> float | None:
        """Posterior mean of scale parameter."""
        samples = self.scale_samples_
        return float(np.mean(samples)) if samples is not None else None

    def summary(self, percentiles: tuple[float, ...] = (2.5, 25, 50, 75, 97.5)) -> dict:
        """Generate posterior summary statistics.

        Parameters
        ----------
        percentiles : tuple of float
            Percentiles to compute (default: 2.5, 25, 50, 75, 97.5)

        Returns
        -------
        dict
            Summary with mean, std, percentiles for each parameter
        """
        summary = {}

        # Coefficients
        coef_samples = self.coef_samples_
        n_coef = coef_samples.shape[1]

        coef_summary = {
            "mean": np.mean(coef_samples, axis=0),
            "std": np.std(coef_samples, axis=0),
            "percentiles": {
                p: np.percentile(coef_samples, p, axis=0) for p in percentiles
            },
        }

        # Add R-hat and ESS if available
        if self._r_hat is not None and "beta" in self._r_hat:
            coef_summary["r_hat"] = self._r_hat["beta"]
        if self._ess is not None and "beta" in self._ess:
            coef_summary["ess"] = self._ess["beta"]

        summary["coef"] = coef_summary

        # Scale parameter
        scale_samples = self.scale_samples_
        if scale_samples is not None:
            summary["scale"] = {
                "mean": float(np.mean(scale_samples)),
                "std": float(np.std(scale_samples)),
                "percentiles": {
                    p: float(np.percentile(scale_samples, p)) for p in percentiles
                },
            }

        # Model info
        summary["n_obs"] = self.n_obs_
        summary["n_features"] = self.n_features_
        summary["n_samples"] = self.n_samples_ * self.n_chains_
        summary["n_chains"] = self.n_chains_
        summary["family"] = self.family
        summary["link"] = self.link
        summary["backend"] = self.backend_

        return summary

    def credible_intervals(
        self, level: float = 0.95, method: str = "hdi"
    ) -> dict[str, NDArray]:
        """Compute credible intervals for parameters.

        Parameters
        ----------
        level : float
            Credibility level (e.g., 0.95 for 95% CI)
        method : str
            'hdi' for Highest Density Interval or 'eti' for
            Equal-Tailed Interval

        Returns
        -------
        dict
            Dictionary with 'coef' key containing array of shape (n_features, 2)
            where [:, 0] is lower and [:, 1] is upper bound
        """
        result = {}

        coef_samples = self.coef_samples_
        n_features = coef_samples.shape[1]

        if method == "hdi":
            ci = np.zeros((n_features, 2))
            for i in range(n_features):
                ci[i] = self._compute_hdi(coef_samples[:, i], level)
        else:  # eti
            alpha = (1 - level) / 2
            ci = np.column_stack(
                [
                    np.percentile(coef_samples, 100 * alpha, axis=0),
                    np.percentile(coef_samples, 100 * (1 - alpha), axis=0),
                ]
            )

        result["coef"] = ci

        # Scale parameter
        scale_samples = self.scale_samples_
        if scale_samples is not None:
            if method == "hdi":
                result["scale"] = self._compute_hdi(scale_samples, level)
            else:
                alpha = (1 - level) / 2
                result["scale"] = np.array(
                    [
                        np.percentile(scale_samples, 100 * alpha),
                        np.percentile(scale_samples, 100 * (1 - alpha)),
                    ]
                )

        return result

    def _compute_hdi(self, samples: NDArray, level: float) -> NDArray:
        """Compute Highest Density Interval."""
        samples_sorted = np.sort(samples)
        n = len(samples_sorted)
        n_included = int(np.ceil(level * n))
        n_ci = n - n_included

        # Find narrowest interval
        width = samples_sorted[n_included:] - samples_sorted[:n_ci]
        min_idx = np.argmin(width)

        return np.array([samples_sorted[min_idx], samples_sorted[min_idx + n_included]])

    def predict(
        self,
        X_new: NDArray | None = None,
        type: str = "mean",
        n_samples: int | None = None,
    ) -> NDArray:
        """Generate predictions from posterior.

        Parameters
        ----------
        X_new : ndarray, optional
            New design matrix. If None, uses training data.
        type : str
            'mean' for posterior mean prediction,
            'samples' for posterior predictive samples
        n_samples : int, optional
            Number of posterior samples to use (for 'samples' type)

        Returns
        -------
        ndarray
            If type='mean': shape (n_new,)
            If type='samples': shape (n_samples, n_new)
        """
        if X_new is None:
            X_new = self.X_

        X_new = np.asarray(X_new)
        coef_samples = self.coef_samples_

        if n_samples is not None:
            # Subsample
            idx = np.random.choice(len(coef_samples), size=n_samples, replace=False)
            coef_samples = coef_samples[idx]

        # Linear predictor samples
        eta_samples = X_new @ coef_samples.T  # (n_new, n_samples)

        # Apply inverse link
        mu_samples = self._apply_inverse_link(eta_samples)

        if type == "mean":
            return np.mean(mu_samples, axis=1)
        else:  # samples
            return mu_samples.T  # (n_samples, n_new)

    def _apply_inverse_link(self, eta: NDArray) -> NDArray:
        """Apply inverse link function."""
        if self.link in ("identity", None):
            return eta
        elif self.link == "log":
            return np.exp(eta)
        elif self.link == "logit":
            return 1 / (1 + np.exp(-eta))
        elif self.link == "probit":
            from scipy.stats import norm

            return norm.cdf(eta)
        elif self.link == "inverse":
            return 1 / eta
        elif self.link == "sqrt":
            return eta**2
        else:
            raise ValueError(f"Unknown link function: {self.link}")

    def posterior_predictive(
        self, X_new: NDArray | None = None, n_samples: int = 1000
    ) -> NDArray:
        """Sample from posterior predictive distribution.

        Parameters
        ----------
        X_new : ndarray, optional
            New design matrix. If None, uses training data.
        n_samples : int
            Number of samples to draw

        Returns
        -------
        ndarray
            Shape (n_samples, n_new) samples from predictive distribution
        """
        if X_new is None:
            X_new = self.X_

        X_new = np.asarray(X_new)
        n_new = X_new.shape[0]

        # Get mu samples
        mu_samples = self.predict(X_new, type="samples", n_samples=n_samples)

        # Sample from observation model
        y_pred = np.zeros((n_samples, n_new))

        for i in range(n_samples):
            mu = mu_samples[i]

            if self.family == "gaussian":
                scale = self.scale_ if self.scale_ else 1.0
                y_pred[i] = np.random.normal(mu, scale)
            elif self.family == "poisson":
                y_pred[i] = np.random.poisson(np.maximum(mu, 1e-10))
            elif self.family == "binomial":
                y_pred[i] = np.random.binomial(1, np.clip(mu, 0, 1))
            elif self.family == "gamma":
                scale = self.scale_ if self.scale_ else 1.0
                shape = mu / scale
                y_pred[i] = np.random.gamma(shape, scale)
            elif self.family == "negative_binomial":
                # Parameterized by mu and dispersion
                theta = self.posterior_samples_.get("theta", np.array([1.0]))
                if hasattr(theta, "__len__"):
                    theta = float(np.mean(theta))
                p = theta / (theta + mu)
                y_pred[i] = np.random.negative_binomial(theta, p)
            else:
                # Default to Gaussian
                y_pred[i] = np.random.normal(mu, 1.0)

        return y_pred

    def waic(self) -> dict:
        """Compute Widely Applicable Information Criterion.

        Returns
        -------
        dict
            WAIC value and standard error
        """
        # Compute log-likelihood for each observation and sample
        ll = self._compute_pointwise_ll()

        # WAIC components
        lppd = np.sum(np.log(np.mean(np.exp(ll), axis=0)))
        p_waic = np.sum(np.var(ll, axis=0))

        waic = -2 * (lppd - p_waic)
        se = 2 * np.sqrt(self.n_obs_ * np.var(-2 * (np.log(np.mean(np.exp(ll), axis=0)) - np.var(ll, axis=0))))

        return {"waic": waic, "p_waic": p_waic, "se": se}

    def _compute_pointwise_ll(self) -> NDArray:
        """Compute pointwise log-likelihood for each sample."""
        coef_samples = self.coef_samples_
        n_samples = len(coef_samples)

        ll = np.zeros((n_samples, self.n_obs_))

        for i in range(n_samples):
            eta = self.X_ @ coef_samples[i]
            mu = self._apply_inverse_link(eta)

            if self.family == "gaussian":
                scale = self.scale_ if self.scale_ else 1.0
                ll[i] = -0.5 * ((self.y_ - mu) / scale) ** 2 - np.log(scale)
            elif self.family == "poisson":
                from scipy.special import gammaln

                mu = np.maximum(mu, 1e-10)
                ll[i] = self.y_ * np.log(mu) - mu - gammaln(self.y_ + 1)
            elif self.family == "binomial":
                mu = np.clip(mu, 1e-10, 1 - 1e-10)
                ll[i] = self.y_ * np.log(mu) + (1 - self.y_) * np.log(1 - mu)
            else:
                # Fallback to Gaussian
                ll[i] = -0.5 * (self.y_ - mu) ** 2

        return ll

    @classmethod
    def from_numpyro(
        cls,
        samples: dict,
        X: NDArray,
        y: NDArray,
        family: str,
        link: str,
        n_chains: int = 1,
    ) -> "BayesianGLMResult":
        """Construct result from NumPyro samples.

        Parameters
        ----------
        samples : dict
            Dictionary of posterior samples from NumPyro
        X : ndarray
            Design matrix
        y : ndarray
            Response variable
        family : str
            Distribution family
        link : str
            Link function
        n_chains : int
            Number of chains used

        Returns
        -------
        BayesianGLMResult
        """
        # Determine n_samples from first parameter
        first_key = next(iter(samples.keys()))
        total_samples = len(samples[first_key])
        n_samples = total_samples // n_chains if n_chains > 1 else total_samples

        return cls(
            posterior_samples_=samples,
            family=family,
            link=link,
            n_samples_=n_samples,
            n_chains_=n_chains,
            n_obs_=len(y),
            n_features_=X.shape[1],
            X_=X,
            y_=y,
            backend_="numpyro",
        )

    @classmethod
    def from_pymc(
        cls,
        trace,
        X: NDArray,
        y: NDArray,
        family: str,
        link: str,
    ) -> "BayesianGLMResult":
        """Construct result from PyMC InferenceData.

        Parameters
        ----------
        trace : arviz.InferenceData
            Trace from PyMC sampling
        X : ndarray
            Design matrix
        y : ndarray
            Response variable
        family : str
            Distribution family
        link : str
            Link function

        Returns
        -------
        BayesianGLMResult
        """
        # Extract samples from trace
        posterior = trace.posterior

        samples = {}
        for var in posterior.data_vars:
            # Convert to numpy, combining chains
            data = posterior[var].values
            samples[var] = data

        n_chains = data.shape[0]
        n_samples = data.shape[1]

        return cls(
            posterior_samples_=samples,
            family=family,
            link=link,
            n_samples_=n_samples,
            n_chains_=n_chains,
            n_obs_=len(y),
            n_features_=X.shape[1],
            X_=X,
            y_=y,
            backend_="pymc",
        )


@dataclass
class BayesianGAMResult(BayesianGLMResult):
    """Results from Bayesian GAM fitting.

    Extends BayesianGLMResult with smooth term-specific summaries.
    """

    smooth_names_: list[str] = field(default_factory=list)
    smooth_indices_: dict[str, tuple[int, int]] = field(default_factory=dict)

    def smooth_summary(self, smooth_name: str) -> dict:
        """Get summary for a specific smooth term."""
        if smooth_name not in self.smooth_indices_:
            raise ValueError(f"Unknown smooth term: {smooth_name}")

        start, end = self.smooth_indices_[smooth_name]
        coef_samples = self.coef_samples_[:, start:end]

        return {
            "name": smooth_name,
            "n_basis": end - start,
            "coef_mean": np.mean(coef_samples, axis=0),
            "coef_std": np.std(coef_samples, axis=0),
        }
