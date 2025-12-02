"""Generalized Additive Mixed Model routines."""
from __future__ import annotations

from aurora.models.gamm.covariance import (
    CovarianceStructure,
    DiagonalCovariance,
    IdentityCovariance,
    UnstructuredCovariance,
    AR1Covariance,
    CompoundSymmetryCovariance,
    ExponentialSpatialCovariance,
    MaternCovariance,
    get_covariance_structure,
)
from aurora.models.gamm.design import construct_Z_matrix, extract_random_effects
from aurora.models.gamm.diagnostics import (
    compute_r2_conditional_marginal,
    interpret_variance_components,
    plot_diagnostics as plot_gamm_diagnostics,
    plot_random_effects as plot_gamm_random_effects,
)
from aurora.models.gamm.estimation import (
    REMLResult,
    compute_P_matrix,
    compute_V_matrix,
    estimate_fixed_effects,
    estimate_random_effects,
    estimate_variance_components,
    reml_log_likelihood,
)
from aurora.models.gamm.fitting import (
    GAMMResult,
    fit_gamm_gaussian,
    predict_gamm,
    solve_mixed_model_equations,
)
from aurora.models.gamm.interface import (
    fit_gamm,
    fit_gamm_with_smooth,
    predict_from_gamm,
)
from aurora.models.gamm.laplace import LaplaceResult, fit_laplace
from aurora.models.gamm.plotting import (
    plot_caterpillar,
    plot_diagnostics,
    plot_random_effects_density,
    plot_random_effects_qq,
    plot_random_effects_summary,
)
from aurora.models.gamm.pql import PQLResult, fit_pql
from aurora.models.gamm.random_effects import (
    RandomEffect,
    count_random_effects,
    get_group_indices,
    validate_random_effects,
)

__all__ = [
    # Random effects
    "RandomEffect",
    "validate_random_effects",
    "get_group_indices",
    "count_random_effects",
    # Covariance structures
    "CovarianceStructure",
    "UnstructuredCovariance",
    "DiagonalCovariance",
    "IdentityCovariance",
    "AR1Covariance",
    "CompoundSymmetryCovariance",
    "ExponentialSpatialCovariance",
    "MaternCovariance",
    "get_covariance_structure",
    # Design matrices
    "construct_Z_matrix",
    "extract_random_effects",
    # REML estimation
    "REMLResult",
    "estimate_variance_components",
    "estimate_fixed_effects",
    "estimate_random_effects",
    "compute_V_matrix",
    "compute_P_matrix",
    "reml_log_likelihood",
    # GAMM fitting (low-level)
    "GAMMResult",
    "fit_gamm_gaussian",
    "predict_gamm",
    "solve_mixed_model_equations",
    # GAMM interface (high-level)
    "fit_gamm",
    "fit_gamm_with_smooth",
    "predict_from_gamm",
    # PQL/Laplace for non-Gaussian GLMMs
    "PQLResult",
    "fit_pql",
    "LaplaceResult",
    "fit_laplace",
    # Visualization
    "plot_caterpillar",
    "plot_random_effects_qq",
    "plot_random_effects_density",
    "plot_diagnostics",
    "plot_random_effects_summary",
    # Diagnostics (new)
    "plot_gamm_diagnostics",
    "plot_gamm_random_effects",
    "interpret_variance_components",
    "compute_r2_conditional_marginal",
]
