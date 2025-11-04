"""Generalized Additive Mixed Model routines."""
from __future__ import annotations

from aurora.models.gamm.covariance import (
    CovarianceStructure,
    DiagonalCovariance,
    IdentityCovariance,
    UnstructuredCovariance,
    get_covariance_structure,
)
from aurora.models.gamm.design import construct_Z_matrix, extract_random_effects
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
    # GAMM fitting
    "GAMMResult",
    "fit_gamm_gaussian",
    "predict_gamm",
    "solve_mixed_model_equations",
]
