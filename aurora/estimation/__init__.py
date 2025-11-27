"""Estimation strategies for smoothing and mixed models.

This module provides various estimation methods for variance components
and smoothing parameters in mixed and additive models.

Submodules
----------
laplace
    Laplace approximation for non-Gaussian GLMMs.
    See: aurora.models.gamm.laplace for current implementation.

ml
    Maximum Likelihood estimation.
    (Planned for future release)

reml
    Restricted Maximum Likelihood estimation.
    See: aurora.models.gamm.estimation for current implementation.

Notes
-----
The main estimation functionality is currently implemented in:
- aurora.models.gamm.estimation (REML for GAMM)
- aurora.models.gamm.laplace (Laplace approximation)
- aurora.smoothing.selection (GCV and REML for smoothing parameters)

These will be consolidated here in a future refactoring.
"""