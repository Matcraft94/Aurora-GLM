# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Aurora-GLM: A modular framework for generalized linear modeling.

Aurora-GLM provides a unified interface for fitting:
- Generalized Linear Models (GLM)
- Generalized Additive Models (GAM)
- Generalized Additive Mixed Models (GAMM)

Quick Start
-----------
>>> import aurora
>>> from aurora import fit_glm, GaussianFamily
>>> result = fit_glm(X, y, family=GaussianFamily())

Multi-backend Support
---------------------
Aurora supports NumPy, PyTorch, and JAX backends:
>>> aurora.available_backends()
['numpy', 'pytorch', 'jax']
>>> aurora.get_backend('pytorch')
"""

from __future__ import annotations

# Core backend functionality
from .core.backends import available_backends, get_backend, register_backend

# Base classes
from .distributions.base import Family, LinkFunction

# Distribution families
from .distributions.families import (
    BetaFamily,
    BinomialFamily,
    CauchyFamily,
    GammaFamily,
    GaussianFamily,
    InverseGaussianFamily,
    NegativeBinomialFamily,
    PoissonFamily,
    StudentTFamily,
    TweedieFamily,
    WaldFamily,
)

# Link functions
from .distributions.links import (
    CLogLogLink,
    IdentityLink,
    InverseLink,
    InverseSquareLink,
    LogitLink,
    LogLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)

# High-level helper functions
from .helpers import compare, plot, summary

# Inference utilities
from .inference import (
    bootstrap_inference,
    confidence_intervals,
    glm_diagnostics,
    robust_covariance,
    wald_test,
)

# Model fitting functions
from .models import fit_gam, fit_gamm, fit_glm, predict_glm
from .models.gam import fit_additive_gam, fit_gam_formula

# Random effects and covariance structures
from .models.gamm import (
    AR1Covariance,
    CompoundSymmetryCovariance,
    ExponentialSpatialCovariance,
    MaternCovariance,
    RandomEffect,
    ToeplitzCovariance,
    fit_gamm_with_smooth,
    predict_from_gamm,
)
from .validation.cross_val import KFold, cross_val_score

# Validation and metrics
from .validation.metrics import accuracy_score, mean_squared_error

# Visualization (centralized)
from .visualization import (
    plot_all_smooths,
    plot_caterpillar,
    plot_diagnostics,
    plot_diagnostics_panel,
    plot_smooth,
)

# Convenience aliases (short names)
Gaussian = GaussianFamily
Binomial = BinomialFamily
Poisson = PoissonFamily
Gamma = GammaFamily
Beta = BetaFamily
InverseGaussian = InverseGaussianFamily
Wald = WaldFamily
StudentT = StudentTFamily
Cauchy = CauchyFamily
NegBin = NegativeBinomialFamily
Tweedie = TweedieFamily

__all__ = [
    # Version
    "__version__",
    # Backend functions
    "available_backends",
    "get_backend",
    "register_backend",
    # Model fitting (high-level)
    "fit_glm",
    "fit_gam",
    "fit_gamm",
    "predict_glm",
    "fit_additive_gam",
    "fit_gam_formula",
    "fit_gamm_with_smooth",
    "predict_from_gamm",
    # Distribution families
    "Family",
    "GaussianFamily",
    "BinomialFamily",
    "PoissonFamily",
    "GammaFamily",
    "BetaFamily",
    "InverseGaussianFamily",
    "WaldFamily",
    "StudentTFamily",
    "CauchyFamily",
    "NegativeBinomialFamily",
    "TweedieFamily",
    # Family aliases (convenience)
    "Gaussian",
    "Binomial",
    "Poisson",
    "Gamma",
    "Beta",
    "InverseGaussian",
    "Wald",
    "StudentT",
    "Cauchy",
    "NegBin",
    "Tweedie",
    # Link functions
    "LinkFunction",
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "InverseLink",
    "CLogLogLink",
    "ProbitLink",
    "SqrtLink",
    "PowerLink",
    "InverseSquareLink",
    # Random effects
    "RandomEffect",
    # Covariance structures
    "AR1Covariance",
    "CompoundSymmetryCovariance",
    "ExponentialSpatialCovariance",
    "MaternCovariance",
    "ToeplitzCovariance",
    # Inference
    "confidence_intervals",
    "glm_diagnostics",
    "wald_test",
    "robust_covariance",
    "bootstrap_inference",
    # Visualization
    "plot_smooth",
    "plot_all_smooths",
    "plot_caterpillar",
    "plot_diagnostics",
    "plot_diagnostics_panel",
    # Validation
    "mean_squared_error",
    "accuracy_score",
    "cross_val_score",
    "KFold",
    # High-level helpers
    "summary",
    "plot",
    "compare",
]

__version__ = "1.0.0"
