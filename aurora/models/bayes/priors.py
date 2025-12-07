# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Prior specification classes for Bayesian inference.

This module provides a flexible system for specifying priors on model
parameters in Bayesian GLM and GAM models.

Examples
--------
>>> from aurora.models.bayes import PriorSpec, Normal, HalfCauchy
>>>
>>> # Default weakly informative priors
>>> priors = PriorSpec()
>>>
>>> # Custom regularizing prior on coefficients
>>> priors = PriorSpec()
>>> priors.coef_prior = Normal(0, 1)
>>> priors.intercept_prior = Normal(0, 10)
>>> priors.scale_prior = HalfCauchy(2.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Prior",
    "Normal",
    "Cauchy",
    "HalfNormal",
    "HalfCauchy",
    "Exponential",
    "Gamma",
    "InverseGamma",
    "Uniform",
    "StudentT",
    "Laplace",
    "PriorSpec",
]


@dataclass
class Prior:
    """Base class for prior distributions.

    All prior classes inherit from this base and define their
    distribution-specific parameters.
    """

    def to_numpyro(self):
        """Convert to NumPyro distribution."""
        raise NotImplementedError("Subclasses must implement to_numpyro()")

    def to_pymc(self):
        """Convert to PyMC distribution."""
        raise NotImplementedError("Subclasses must implement to_pymc()")


@dataclass
class Normal(Prior):
    """Normal (Gaussian) prior N(mu, sigma^2).

    Parameters
    ----------
    mu : float
        Mean of the distribution
    sigma : float
        Standard deviation (must be positive)

    Examples
    --------
    >>> prior = Normal(0, 1)  # Standard normal
    >>> prior = Normal(0, 10)  # Weakly informative
    """

    mu: float = 0.0
    sigma: float = 10.0

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.Normal(self.mu, self.sigma)

    def to_pymc(self):
        import pymc as pm

        return pm.Normal.dist(mu=self.mu, sigma=self.sigma)


@dataclass
class Cauchy(Prior):
    """Cauchy prior for heavy-tailed uncertainty.

    The Cauchy distribution has heavier tails than the Normal,
    making it robust to outliers and useful for coefficients
    that may have large effects.

    Parameters
    ----------
    loc : float
        Location parameter (median)
    scale : float
        Scale parameter (must be positive)
    """

    loc: float = 0.0
    scale: float = 2.5

    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.Cauchy(self.loc, self.scale)

    def to_pymc(self):
        import pymc as pm

        return pm.Cauchy.dist(alpha=self.loc, beta=self.scale)


@dataclass
class HalfNormal(Prior):
    """Half-Normal prior for positive parameters.

    The Half-Normal is a Normal(0, sigma) truncated to positive values.
    Commonly used for scale parameters.

    Parameters
    ----------
    sigma : float
        Scale parameter of the underlying Normal
    """

    sigma: float = 1.0

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.HalfNormal(self.sigma)

    def to_pymc(self):
        import pymc as pm

        return pm.HalfNormal.dist(sigma=self.sigma)


@dataclass
class HalfCauchy(Prior):
    """Half-Cauchy prior for positive parameters with heavy tails.

    Recommended by Gelman (2006) as a weakly informative prior
    for scale parameters in hierarchical models.

    Parameters
    ----------
    scale : float
        Scale parameter (must be positive)

    References
    ----------
    .. [1] Gelman, A. (2006). Prior distributions for variance parameters
           in hierarchical models. Bayesian Analysis, 1(3), 515-534.
    """

    scale: float = 2.5

    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.HalfCauchy(self.scale)

    def to_pymc(self):
        import pymc as pm

        return pm.HalfCauchy.dist(beta=self.scale)


@dataclass
class Exponential(Prior):
    """Exponential prior for positive parameters.

    Parameters
    ----------
    rate : float
        Rate parameter (lambda). Mean is 1/rate.
    """

    rate: float = 1.0

    def __post_init__(self):
        if self.rate <= 0:
            raise ValueError("rate must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.Exponential(self.rate)

    def to_pymc(self):
        import pymc as pm

        return pm.Exponential.dist(lam=self.rate)


@dataclass
class Gamma(Prior):
    """Gamma prior for positive parameters.

    Parameterized by shape (alpha) and rate (beta).
    Mean = alpha/beta, Variance = alpha/beta^2.

    Parameters
    ----------
    alpha : float
        Shape parameter (must be positive)
    beta : float
        Rate parameter (must be positive)
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.Gamma(self.alpha, self.beta)

    def to_pymc(self):
        import pymc as pm

        return pm.Gamma.dist(alpha=self.alpha, beta=self.beta)


@dataclass
class InverseGamma(Prior):
    """Inverse-Gamma prior for variance parameters.

    If X ~ Gamma(alpha, beta), then 1/X ~ InverseGamma(alpha, beta).
    Commonly used as a conjugate prior for the variance of a Normal.

    Parameters
    ----------
    alpha : float
        Shape parameter (must be positive)
    beta : float
        Scale parameter (must be positive)

    Notes
    -----
    For a weakly informative prior on variance, use small alpha and beta
    (e.g., alpha=0.001, beta=0.001), though this can cause issues near zero.
    """

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.InverseGamma(self.alpha, self.beta)

    def to_pymc(self):
        import pymc as pm

        return pm.InverseGamma.dist(alpha=self.alpha, beta=self.beta)


@dataclass
class Uniform(Prior):
    """Uniform prior over an interval.

    Parameters
    ----------
    lower : float
        Lower bound of the interval
    upper : float
        Upper bound of the interval
    """

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError("lower must be less than upper")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.Uniform(self.lower, self.upper)

    def to_pymc(self):
        import pymc as pm

        return pm.Uniform.dist(lower=self.lower, upper=self.upper)


@dataclass
class StudentT(Prior):
    """Student-t prior for robust inference.

    Heavier tails than Normal, controlled by degrees of freedom.
    As df -> infinity, approaches Normal.

    Parameters
    ----------
    df : float
        Degrees of freedom (must be positive)
    loc : float
        Location parameter
    scale : float
        Scale parameter (must be positive)
    """

    df: float = 3.0
    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        if self.df <= 0 or self.scale <= 0:
            raise ValueError("df and scale must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.StudentT(self.df, self.loc, self.scale)

    def to_pymc(self):
        import pymc as pm

        return pm.StudentT.dist(nu=self.df, mu=self.loc, sigma=self.scale)


@dataclass
class Laplace(Prior):
    """Laplace (double exponential) prior.

    Induces sparsity similar to L1 regularization (Lasso).
    Useful when expecting many coefficients to be near zero.

    Parameters
    ----------
    loc : float
        Location parameter (mode)
    scale : float
        Scale parameter (must be positive)
    """

    loc: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def to_numpyro(self):
        import numpyro.distributions as dist

        return dist.Laplace(self.loc, self.scale)

    def to_pymc(self):
        import pymc as pm

        return pm.Laplace.dist(mu=self.loc, b=self.scale)


@dataclass
class PriorSpec:
    """Specification of priors for all model parameters.

    This class holds prior distributions for different parameter types
    in a GLM or GAM. Default priors are weakly informative.

    Attributes
    ----------
    intercept_prior : Prior
        Prior for the intercept term
    coef_prior : Prior
        Prior for regression coefficients
    scale_prior : Prior
        Prior for scale/dispersion parameters (e.g., sigma in Gaussian)
    shape_prior : Prior
        Prior for shape parameters (e.g., in Gamma, Negative Binomial)

    Examples
    --------
    >>> spec = PriorSpec()
    >>> spec.coef_prior = Normal(0, 1)  # Regularizing
    >>> spec.scale_prior = HalfCauchy(2.5)  # Weakly informative
    """

    intercept_prior: Prior = field(default_factory=lambda: Normal(0, 100))
    coef_prior: Prior = field(default_factory=lambda: Normal(0, 10))
    scale_prior: Prior = field(default_factory=lambda: HalfNormal(1.0))
    shape_prior: Prior = field(default_factory=lambda: Exponential(1.0))

    # For specific coefficient indices (optional)
    _coef_priors_by_index: dict[int, Prior] = field(default_factory=dict)

    def set_coef_prior(self, prior: Prior, indices: list[int] | None = None):
        """Set prior for specific coefficients.

        Parameters
        ----------
        prior : Prior
            Prior distribution to use
        indices : list of int, optional
            Coefficient indices. If None, sets default for all coefficients.
        """
        if indices is None:
            self.coef_prior = prior
        else:
            for idx in indices:
                self._coef_priors_by_index[idx] = prior

    def get_coef_prior(self, index: int) -> Prior:
        """Get prior for a specific coefficient index."""
        return self._coef_priors_by_index.get(index, self.coef_prior)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "intercept_prior": self.intercept_prior,
            "coef_prior": self.coef_prior,
            "scale_prior": self.scale_prior,
            "shape_prior": self.shape_prior,
        }
