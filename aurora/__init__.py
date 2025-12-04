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

# Model fitting functions
from .models import fit_glm, fit_gam, fit_gamm, predict_glm
from .models.gam import fit_additive_gam, fit_gam_formula
from .models.gamm import fit_gamm_with_smooth, predict_from_gamm

# Distribution families
from .distributions.families import (
    GaussianFamily,
    BinomialFamily,
    PoissonFamily,
    GammaFamily,
    StudentTFamily,
    NegativeBinomialFamily,
    TweedieFamily,
)

# Link functions
from .distributions.links import (
    IdentityLink,
    LogLink,
    LogitLink,
    InverseLink,
    CLogLogLink,
    SqrtLink,
    PowerLink,
)

# Base classes
from .distributions.base import Family, LinkFunction

# Random effects and covariance structures
from .models.gamm import (
    RandomEffect,
    AR1Covariance,
    CompoundSymmetryCovariance,
    ExponentialSpatialCovariance,
    MaternCovariance,
)

# Inference utilities
from .inference import (
    confidence_intervals,
    glm_diagnostics,
    wald_test,
    robust_covariance,
    bootstrap_inference,
)

# Validation and metrics
from .validation.metrics import mean_squared_error, accuracy_score
from .validation.cross_val import cross_val_score, KFold

# Visualization (centralized)
from .visualization import (
    plot_smooth,
    plot_all_smooths,
    plot_caterpillar,
    plot_diagnostics,
    plot_diagnostics_panel,
)

# High-level helper functions
from .helpers import summary, plot, compare

# Convenience aliases (short names)
Gaussian = GaussianFamily
Binomial = BinomialFamily
Poisson = PoissonFamily
Gamma = GammaFamily
StudentT = StudentTFamily
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
    "StudentTFamily",
    "NegativeBinomialFamily",
    "TweedieFamily",
    # Family aliases (convenience)
    "Gaussian",
    "Binomial",
    "Poisson",
    "Gamma",
    "StudentT",
    "NegBin",
    "Tweedie",
    # Link functions
    "LinkFunction",
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "InverseLink",
    "CLogLogLink",
    "SqrtLink",
    "PowerLink",
    # Random effects
    "RandomEffect",
    # Covariance structures
    "AR1Covariance",
    "CompoundSymmetryCovariance",
    "ExponentialSpatialCovariance",
    "MaternCovariance",
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

__version__ = "0.6.1"
