# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""PyMC backend for Bayesian inference.

This module provides functions to build and sample from PyMC models
that correspond to Aurora GLM specifications.

PyMC is a probabilistic programming library that provides MCMC sampling
via NUTS and other algorithms.

References
----------
.. [1] Abril-Pla, O., et al. (2023). PyMC: A Modern and Comprehensive
       Probabilistic Programming Framework in Python. PeerJ Computer Science.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..priors import PriorSpec

__all__ = ["build_pymc_model", "sample_pymc"]


def build_pymc_model(
    X: NDArray,
    y: NDArray,
    family: str,
    link: str | None,
    priors: "PriorSpec",
) -> Any:
    """Build a PyMC model from Aurora GLM specification.

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
    pm.Model
        PyMC model object
    """
    import pymc as pm
    import pytensor.tensor as pt

    n, p = X.shape

    # Determine link function
    if link is None:
        link = _canonical_link(family)

    with pm.Model() as model:
        # Data
        X_data = pm.Data("X", X)
        y_data = pm.Data("y", y)

        # Prior on coefficients
        coef_prior = priors.coef_prior
        if hasattr(coef_prior, "mu"):
            beta = pm.Normal("beta", mu=coef_prior.mu, sigma=coef_prior.sigma, shape=p)
        elif hasattr(coef_prior, "loc") and hasattr(coef_prior, "scale"):
            # Cauchy or similar
            beta = pm.Cauchy(
                "beta", alpha=coef_prior.loc, beta=coef_prior.scale, shape=p
            )
        else:
            # Default to Normal
            beta = pm.Normal("beta", mu=0, sigma=10, shape=p)

        # Linear predictor
        eta = pm.math.dot(X_data, beta)

        # Apply inverse link
        mu = _apply_inverse_link_pymc(eta, link)

        # Likelihood based on family
        if family == "gaussian":
            scale_prior = priors.scale_prior
            if hasattr(scale_prior, "sigma"):
                sigma = pm.HalfNormal("sigma", sigma=scale_prior.sigma)
            elif hasattr(scale_prior, "scale"):
                sigma = pm.HalfCauchy("sigma", beta=scale_prior.scale)
            else:
                sigma = pm.HalfNormal("sigma", sigma=1.0)

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

        elif family == "poisson":
            # Ensure mu is positive
            mu_safe = pt.maximum(mu, 1e-10)
            pm.Poisson("y_obs", mu=mu_safe, observed=y_data)

        elif family == "binomial":
            # Ensure mu is in (0, 1)
            mu_safe = pt.clip(mu, 1e-10, 1 - 1e-10)
            pm.Bernoulli("y_obs", p=mu_safe, observed=y_data)

        elif family == "gamma":
            scale_prior = priors.scale_prior
            if hasattr(scale_prior, "sigma"):
                phi = pm.HalfNormal("phi", sigma=scale_prior.sigma)
            else:
                phi = pm.HalfNormal("phi", sigma=1.0)

            # Gamma with mean=mu, shape=mu/phi
            mu_safe = pt.maximum(mu, 1e-10)
            alpha = mu_safe / phi
            beta_param = 1.0 / phi
            pm.Gamma("y_obs", alpha=alpha, beta=beta_param, observed=y_data)

        elif family == "negative_binomial":
            shape_prior = priors.shape_prior
            if hasattr(shape_prior, "rate"):
                theta = pm.Exponential("theta", lam=shape_prior.rate)
            else:
                theta = pm.Exponential("theta", lam=1.0)

            mu_safe = pt.maximum(mu, 1e-10)
            pm.NegativeBinomial("y_obs", mu=mu_safe, alpha=theta, observed=y_data)

        elif family == "beta":
            scale_prior = priors.scale_prior
            if hasattr(scale_prior, "sigma"):
                phi = pm.HalfNormal("phi", sigma=scale_prior.sigma)
            else:
                phi = pm.HalfNormal("phi", sigma=1.0)

            mu_safe = pt.clip(mu, 1e-6, 1 - 1e-6)
            alpha = mu_safe * phi
            beta_param = (1 - mu_safe) * phi
            pm.Beta("y_obs", alpha=alpha, beta=beta_param, observed=y_data)

        else:
            raise ValueError(f"Unsupported family: {family}")

    return model


def sample_pymc(
    model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = None,
    progress_bar: bool = True,
    **kwargs,
):
    """Run MCMC sampling with PyMC.

    Parameters
    ----------
    model : pm.Model
        PyMC model
    draws : int
        Number of posterior samples per chain
    tune : int
        Number of tuning samples
    chains : int
        Number of MCMC chains
    target_accept : float
        Target acceptance probability for NUTS
    random_seed : int, optional
        Random seed for reproducibility
    progress_bar : bool
        Show progress bar during sampling
    **kwargs
        Additional arguments passed to pm.sample

    Returns
    -------
    arviz.InferenceData
        ArviZ InferenceData object with posterior samples
    """
    import pymc as pm

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progress_bar,
            return_inferencedata=True,
            **kwargs,
        )

    return trace


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


def _apply_inverse_link_pymc(eta, link: str):
    """Apply inverse link function using PyTensor operations."""
    import pytensor.tensor as pt

    if link == "identity":
        return eta
    elif link == "log":
        return pt.exp(eta)
    elif link == "logit":
        return pt.sigmoid(eta)
    elif link == "probit":
        # Probit = Normal CDF
        from pytensor.tensor.special import erfc

        return 0.5 * erfc(-eta / pt.sqrt(2.0))
    elif link == "inverse":
        return 1.0 / eta
    elif link == "sqrt":
        return eta**2
    elif link == "inverse_squared":
        return 1.0 / pt.sqrt(eta)
    else:
        raise ValueError(f"Unknown link function: {link}")
