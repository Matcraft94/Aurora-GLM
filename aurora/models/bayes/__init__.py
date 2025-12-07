# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Bayesian inference for GLM and GAM models.

This module provides Bayesian fitting capabilities using probabilistic
programming backends (NumPyro, PyMC).

Key Features
------------
- Flexible prior specification
- MCMC sampling via NUTS
- Posterior summaries and credible intervals
- Posterior predictive sampling
- Support for multiple distribution families

Examples
--------
>>> from aurora.models.bayes import fit_glm_bayes, PriorSpec, Normal
>>>
>>> # Fit with default weakly informative priors
>>> result = fit_glm_bayes(X, y, family='poisson')
>>> print(result.summary())
>>>
>>> # Custom regularizing priors
>>> priors = PriorSpec()
>>> priors.coef_prior = Normal(0, 1)
>>> result = fit_glm_bayes(X, y, family='poisson', priors=priors)
>>>
>>> # 95% credible intervals
>>> ci = result.credible_intervals(0.95)

See Also
--------
aurora.models.glm : Frequentist GLM fitting
aurora.models.gam : Generalized Additive Models
"""

from .priors import (
    Prior,
    Normal,
    Cauchy,
    HalfNormal,
    HalfCauchy,
    Exponential,
    Gamma,
    InverseGamma,
    Uniform,
    StudentT,
    Laplace,
    PriorSpec,
)
from .result import BayesianGLMResult, BayesianGAMResult
from .glm_bayes import fit_glm_bayes
from .backends import available_backends, HAS_NUMPYRO, HAS_PYMC

__all__ = [
    # Main API
    "fit_glm_bayes",
    # Results
    "BayesianGLMResult",
    "BayesianGAMResult",
    # Priors
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
    # Utilities
    "available_backends",
    "HAS_NUMPYRO",
    "HAS_PYMC",
]
