# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""NumPyro backend for Bayesian inference.

This module provides functions to build and sample from NumPyro models
that correspond to Aurora GLM specifications.

NumPyro is a JAX-based probabilistic programming library that provides
fast MCMC sampling via NUTS (No-U-Turn Sampler).

References
----------
.. [1] Phan, D., Pradhan, N., & Jankowiak, M. (2019).
       Composable Effects for Flexible and Accelerated Probabilistic
       Programming in NumPyro. arXiv:1912.11554
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..priors import PriorSpec

__all__ = ["build_numpyro_model", "sample_numpyro"]


def build_numpyro_model(
    X: NDArray,
    y: NDArray,
    family: str,
    link: str | None,
    priors: "PriorSpec",
) -> Callable:
    """Build a NumPyro model from Aurora GLM specification.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix
    y : ndarray, shape (n,)
        Response variable
    family : str
        Distribution family ('gaussian', 'poisson', 'binomial', 'gamma',
        'negative_binomial')
    link : str or None
        Link function. If None, uses canonical link.
    priors : PriorSpec
        Prior specifications

    Returns
    -------
    callable
        NumPyro model function
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n, p = X.shape
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Determine link function
    if link is None:
        link = _canonical_link(family)

    def model(X=None, y=None):
        if X is None:
            X = X_jax
        if y is None:
            y = y_jax

        # Prior on coefficients
        coef_prior = priors.coef_prior.to_numpyro()
        beta = numpyro.sample("beta", coef_prior.expand([p]))

        # Linear predictor
        eta = jnp.dot(X, beta)

        # Apply inverse link
        mu = _apply_inverse_link_jax(eta, link)

        # Likelihood based on family
        if family == "gaussian":
            scale_prior = priors.scale_prior.to_numpyro()
            sigma = numpyro.sample("sigma", scale_prior)
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

        elif family == "poisson":
            # Ensure mu is positive
            mu = jnp.maximum(mu, 1e-10)
            numpyro.sample("y", dist.Poisson(mu), obs=y)

        elif family == "binomial":
            # Ensure mu is in (0, 1)
            mu = jnp.clip(mu, 1e-10, 1 - 1e-10)
            numpyro.sample("y", dist.Bernoulli(probs=mu), obs=y)

        elif family == "gamma":
            scale_prior = priors.scale_prior.to_numpyro()
            phi = numpyro.sample("phi", scale_prior)
            # Gamma parameterized by shape and rate
            # mean = shape/rate = mu, so shape = mu/phi, rate = 1/phi
            concentration = mu / phi
            rate = 1.0 / phi
            numpyro.sample("y", dist.Gamma(concentration, rate), obs=y)

        elif family == "negative_binomial":
            shape_prior = priors.shape_prior.to_numpyro()
            theta = numpyro.sample("theta", shape_prior)
            # NegBin2 parameterization: mean=mu, variance=mu + mu^2/theta
            mu = jnp.maximum(mu, 1e-10)
            numpyro.sample(
                "y", dist.NegativeBinomial2(mean=mu, concentration=theta), obs=y
            )

        elif family == "beta":
            # Beta regression with logit link
            scale_prior = priors.scale_prior.to_numpyro()
            phi = numpyro.sample("phi", scale_prior)
            # Beta parameterized by alpha=mu*phi, beta=(1-mu)*phi
            mu = jnp.clip(mu, 1e-6, 1 - 1e-6)
            alpha = mu * phi
            beta_param = (1 - mu) * phi
            numpyro.sample("y", dist.Beta(alpha, beta_param), obs=y)

        else:
            raise ValueError(f"Unsupported family: {family}")

    return model


def sample_numpyro(
    model: Callable,
    X: NDArray,
    y: NDArray,
    num_samples: int = 2000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    progress_bar: bool = True,
    **kwargs,
) -> dict:
    """Run NUTS sampling with NumPyro.

    Parameters
    ----------
    model : callable
        NumPyro model function
    X : ndarray
        Design matrix
    y : ndarray
        Response variable
    num_samples : int
        Number of posterior samples per chain
    num_warmup : int
        Number of warmup/tuning samples
    num_chains : int
        Number of MCMC chains
    seed : int
        Random seed
    progress_bar : bool
        Show progress bar during sampling
    **kwargs
        Additional arguments passed to MCMC

    Returns
    -------
    dict
        Dictionary of posterior samples
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Setup NUTS sampler
    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        **kwargs,
    )

    # Run sampling
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, X=X_jax, y=y_jax)

    # Get samples as numpy arrays
    samples = {k: np.array(v) for k, v in mcmc.get_samples().items()}

    return samples


def _canonical_link(family: str) -> str:
    """Return canonical link for a family."""
    canonical = {
        "gaussian": "identity",
        "poisson": "log",
        "binomial": "logit",
        "gamma": "inverse",
        "negative_binomial": "log",
        "beta": "logit",
        "inverse_gaussian": "inverse_squared",
    }
    return canonical.get(family, "identity")


def _apply_inverse_link_jax(eta, link: str):
    """Apply inverse link function using JAX operations."""
    import jax.numpy as jnp
    from jax.scipy.special import expit

    if link == "identity":
        return eta
    elif link == "log":
        return jnp.exp(eta)
    elif link == "logit":
        return expit(eta)
    elif link == "probit":
        from jax.scipy.stats import norm

        return norm.cdf(eta)
    elif link == "inverse":
        return 1.0 / eta
    elif link == "sqrt":
        return eta**2
    elif link == "inverse_squared":
        return 1.0 / jnp.sqrt(eta)
    else:
        raise ValueError(f"Unknown link function: {link}")
