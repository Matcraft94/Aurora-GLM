# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Bayesian GLM fitting.

This module provides the high-level API for fitting Bayesian
Generalized Linear Models using MCMC.

Examples
--------
>>> from aurora.models.bayes import fit_glm_bayes, PriorSpec, Normal
>>>
>>> # Fit with default priors
>>> result = fit_glm_bayes(X, y, family='poisson')
>>>
>>> # Custom priors
>>> priors = PriorSpec()
>>> priors.coef_prior = Normal(0, 1)  # Regularizing prior
>>> result = fit_glm_bayes(X, y, family='poisson', priors=priors)
>>>
>>> # Posterior summary
>>> print(result.summary())
>>> ci = result.credible_intervals(0.95)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .backends import HAS_NUMPYRO, HAS_PYMC, get_default_backend
from .priors import PriorSpec
from .result import BayesianGLMResult

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["fit_glm_bayes"]


def fit_glm_bayes(
    X: ArrayLike,
    y: ArrayLike,
    family: str = "gaussian",
    link: str | None = None,
    priors: PriorSpec | None = None,
    backend: str | None = None,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    seed: int | None = None,
    progress_bar: bool = True,
    **kwargs,
) -> BayesianGLMResult:
    """Fit Bayesian GLM using MCMC.

    This function fits a Generalized Linear Model using Bayesian inference
    with MCMC sampling. It supports multiple backends (NumPyro, PyMC) and
    distribution families.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix with n observations and p features
    y : array-like, shape (n,)
        Response variable
    family : str, default='gaussian'
        Distribution family. Options:
        - 'gaussian': Normal distribution
        - 'poisson': Poisson for count data
        - 'binomial': Bernoulli/Binomial for binary data
        - 'gamma': Gamma for positive continuous data
        - 'negative_binomial': For overdispersed counts
        - 'beta': Beta for (0, 1) bounded data
    link : str, optional
        Link function. If None, uses canonical link for the family.
        Options: 'identity', 'log', 'logit', 'probit', 'inverse', 'sqrt'
    priors : PriorSpec, optional
        Prior specifications for model parameters. If None, uses
        weakly informative defaults.
    backend : str, optional
        Probabilistic programming backend. Options: 'numpyro', 'pymc'.
        If None, uses best available (prefers numpyro).
    draws : int, default=2000
        Number of posterior samples per chain after warmup
    tune : int, default=1000
        Number of warmup/tuning samples per chain
    chains : int, default=4
        Number of MCMC chains (for convergence diagnostics)
    seed : int, optional
        Random seed for reproducibility
    progress_bar : bool, default=True
        Show progress bar during sampling
    **kwargs
        Additional arguments passed to the sampler

    Returns
    -------
    BayesianGLMResult
        Object containing posterior samples and methods for
        summarizing and predicting from the posterior

    Raises
    ------
    ImportError
        If no Bayesian backend is available
    ValueError
        If family or link is not supported

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.bayes import fit_glm_bayes
    >>>
    >>> # Generate data
    >>> np.random.seed(42)
    >>> n, p = 100, 3
    >>> X = np.random.randn(n, p)
    >>> beta_true = np.array([1.0, -0.5, 0.3])
    >>> y = np.random.poisson(np.exp(X @ beta_true))
    >>>
    >>> # Fit Bayesian Poisson regression
    >>> result = fit_glm_bayes(X, y, family='poisson')
    >>>
    >>> # Posterior summary
    >>> summary = result.summary()
    >>> print(f"Posterior mean: {result.coef_}")
    >>>
    >>> # 95% credible intervals
    >>> ci = result.credible_intervals(0.95)
    >>> print(f"95% CI: {ci['coef']}")

    Notes
    -----
    The MCMC sampler uses the No-U-Turn Sampler (NUTS), an adaptive
    variant of Hamiltonian Monte Carlo that automatically tunes
    step size and trajectory length.

    For convergence diagnostics, check:
    - R-hat values should be < 1.01
    - Effective sample size (ESS) should be > 100 per chain
    - No divergent transitions

    See Also
    --------
    fit_glm : Frequentist GLM fitting
    BayesianGLMResult : Result class with posterior summaries
    PriorSpec : Prior specification
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    n, p = X.shape
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)} elements")

    # Validate family
    valid_families = {
        "gaussian",
        "poisson",
        "binomial",
        "gamma",
        "negative_binomial",
        "beta",
    }
    if family not in valid_families:
        raise ValueError(f"Unknown family: {family}. Must be one of {valid_families}")

    # Default priors
    if priors is None:
        priors = PriorSpec()

    # Select backend
    if backend is None:
        backend = get_default_backend()

    if backend == "numpyro":
        if not HAS_NUMPYRO:
            raise ImportError(
                "NumPyro is not installed. Install with: pip install numpyro jax jaxlib"
            )
        return _fit_numpyro(
            X, y, family, link, priors, draws, tune, chains, seed, progress_bar, **kwargs
        )

    elif backend == "pymc":
        if not HAS_PYMC:
            raise ImportError(
                "PyMC is not installed. Install with: pip install pymc arviz"
            )
        return _fit_pymc(
            X, y, family, link, priors, draws, tune, chains, seed, progress_bar, **kwargs
        )

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'numpyro' or 'pymc'")


def _fit_numpyro(
    X: NDArray,
    y: NDArray,
    family: str,
    link: str | None,
    priors: PriorSpec,
    draws: int,
    tune: int,
    chains: int,
    seed: int | None,
    progress_bar: bool,
    **kwargs,
) -> BayesianGLMResult:
    """Fit using NumPyro backend."""
    from .backends.numpyro_backend import build_numpyro_model, sample_numpyro

    # Build model
    model = build_numpyro_model(X, y, family, link, priors)

    # Sample
    if seed is None:
        seed = np.random.randint(0, 2**31)

    samples = sample_numpyro(
        model,
        X,
        y,
        num_samples=draws,
        num_warmup=tune,
        num_chains=chains,
        seed=seed,
        progress_bar=progress_bar,
        **kwargs,
    )

    # Determine link if not specified
    if link is None:
        from .backends.numpyro_backend import _canonical_link

        link = _canonical_link(family)

    return BayesianGLMResult.from_numpyro(samples, X, y, family, link, n_chains=chains)


def _fit_pymc(
    X: NDArray,
    y: NDArray,
    family: str,
    link: str | None,
    priors: PriorSpec,
    draws: int,
    tune: int,
    chains: int,
    seed: int | None,
    progress_bar: bool,
    **kwargs,
) -> BayesianGLMResult:
    """Fit using PyMC backend."""
    from .backends.pymc_backend import build_pymc_model, sample_pymc

    # Build model
    model = build_pymc_model(X, y, family, link, priors)

    # Sample
    trace = sample_pymc(
        model,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=seed,
        progress_bar=progress_bar,
        **kwargs,
    )

    # Determine link if not specified
    if link is None:
        from .backends.pymc_backend import _canonical_link

        link = _canonical_link(family)

    return BayesianGLMResult.from_pymc(trace, X, y, family, link)
