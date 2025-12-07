# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Visualization tools for GAM smooth terms.

This module provides plotting functions for visualizing fitted smooth
functions from GAMs, including confidence bands and partial residuals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aurora.models.gam.additive import AdditiveGAMResult
    from aurora.models.gam.result import GAMResult

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_smooth(
    result: GAMResult | AdditiveGAMResult,
    term: int | str | None = None,
    confidence_level: float = 0.95,
    n_points: int = 100,
    partial_residuals: bool = True,
    ax=None,
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Figure:
    """Plot a smooth term with confidence bands.

    Parameters
    ----------
    result : GAMResult or AdditiveGAMResult
        Fitted GAM result object.
    term : int, str, or None
        For AdditiveGAMResult: term name like "s(0)" or variable index.
        For GAMResult: should be None (univariate GAM).
    confidence_level : float, default=0.95
        Confidence level for bands (e.g., 0.95 for 95% CI).
    n_points : int, default=100
        Number of evaluation points for smooth curve.
    partial_residuals : bool, default=True
        Whether to show partial residuals.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple, default=(8, 6)
        Figure size if creating new figure.
    title : str, optional
        Plot title. Auto-generated if None.
    xlabel : str, optional
        X-axis label. Auto-generated if None.
    ylabel : str, optional
        Y-axis label. Auto-generated if None.

    Returns
    -------
    fig : matplotlib Figure
        The figure containing the plot.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If term specification is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gam import fit_gam, plot_smooth
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.1 * np.random.randn(100)
    >>> result = fit_gam(x, y, n_basis=15)
    >>> fig = plot_smooth(result)

    >>> # For additive GAM
    >>> from aurora.models.gam import fit_additive_gam, SmoothTerm
    >>> X = np.random.randn(200, 2)
    >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(200)
    >>> result = fit_additive_gam(
    ...     X, y,
    ...     smooth_terms=[SmoothTerm(variable=0), SmoothTerm(variable=1)]
    ... )
    >>> fig = plot_smooth(result, term="s(0)")

    Notes
    -----
    Confidence bands are computed using the Bayesian posterior covariance:
        Var(f(x)) = B(x) @ Cov(β) @ B(x).T

    where B(x) is the basis matrix and Cov(β) = σ² (X'WX + λS)^(-1) X'WX (X'WX + λS)^(-1).

    Partial residuals for term j are:
        r_j = y - f_{-j}(x)
    where f_{-j} excludes term j.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    # Handle univariate vs multivariate GAM
    from aurora.models.gam.additive import AdditiveGAMResult

    is_additive = isinstance(result, AdditiveGAMResult)

    if is_additive:
        if term is None:
            raise ValueError(
                "For AdditiveGAMResult, must specify which term to plot "
                "(e.g., term='s(0)' or term=0)"
            )
        # Convert variable index to term name
        if isinstance(term, int):
            term_name = f"s({term})"
        else:
            term_name = term

        if term_name not in result.smooth_coef:
            raise ValueError(
                f"Term '{term_name}' not found. "
                f"Available terms: {list(result.smooth_coef.keys())}"
            )

        # Find corresponding smooth term
        smooth_term = None
        for st in result.smooth_terms:
            if f"s({st.variable})" == term_name:
                smooth_term = st
                break

        if smooth_term is None:
            raise ValueError(f"Could not find smooth term for '{term_name}'")

        # Extract data for this term
        x_data = result.X_train[:, smooth_term.variable]
        basis = result.smooth_bases[term_name]
        coef = result.smooth_coef[term_name]
        lambda_ = result.lambda_values[term_name]

    else:  # Univariate GAMResult
        if term is not None:
            raise ValueError(
                "For univariate GAMResult, term should be None "
                "(only one smooth term exists)"
            )
        x_data = result.x
        basis = result.basis
        coef = result.coefficients
        lambda_ = result.lambda_
        term_name = "s(x)"

    # Create evaluation grid
    x_min, x_max = np.min(x_data), np.max(x_data)
    x_grid = np.linspace(x_min, x_max, n_points)

    # Evaluate smooth at grid points
    B_grid = basis.basis_matrix(x_grid)
    f_grid = B_grid @ coef

    # Compute confidence bands
    # For now, use simple approach: Var(f) ≈ σ² * B @ (X'X + λS)^(-1) @ B'
    # This is approximate; exact requires full covariance matrix

    residual_var = np.var(result.residuals)

    # Build X'X + λS for this term
    # Reconstruct design matrix and penalty matrix
    B_train = basis.basis_matrix(x_data)

    # Get penalty matrix from basis
    if hasattr(basis, "penalty_matrix"):
        # Check if penalty_matrix accepts order parameter
        import inspect

        sig = inspect.signature(basis.penalty_matrix)
        if "order" in sig.parameters:
            # B-spline basis - use second order penalty
            S = basis.penalty_matrix(order=2)
        else:
            # Cubic spline basis - no order parameter
            S = basis.penalty_matrix()
    else:
        # Fallback to identity if no penalty method
        S = np.eye(len(coef))

    # Penalized precision matrix
    XtX = B_train.T @ B_train
    precision = XtX + lambda_ * S

    # Compute variance at each grid point
    try:
        precision_inv = np.linalg.inv(precision)
        var_grid = np.array(
            [
                residual_var
                * (B_grid[i : i + 1] @ precision_inv @ B_grid[i : i + 1].T)[0, 0]
                for i in range(len(x_grid))
            ]
        )
        se_grid = np.sqrt(var_grid)

        # Confidence multiplier (approximate with normal)
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = f_grid - z_alpha * se_grid
        ci_upper = f_grid + z_alpha * se_grid

        has_ci = True
    except np.linalg.LinAlgError:
        # Singular matrix - skip confidence bands
        has_ci = False

    # Compute partial residuals if requested
    if partial_residuals and is_additive:
        # Partial residuals = y - (prediction without this term)
        # This requires re-predicting without the current term
        # Simplified: show actual residuals offset by this term's contribution
        B_data = basis.basis_matrix(x_data)
        f_data = B_data @ coef
        partial_resid = result.residuals + f_data
        x_resid = x_data
    elif partial_residuals and not is_additive:
        # For univariate GAM, partial residuals = residuals + fitted
        partial_resid = result.residuals + result.fitted_values
        x_resid = x_data
    else:
        partial_resid = None
        x_resid = None

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot confidence band
    if has_ci:
        ax.fill_between(
            x_grid,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color="steelblue",
            label=f"{int(confidence_level * 100)}% CI",
        )

    # Plot smooth curve
    ax.plot(x_grid, f_grid, color="steelblue", linewidth=2, label="Smooth")

    # Plot partial residuals
    if partial_resid is not None:
        ax.scatter(
            x_resid,
            partial_resid,
            alpha=0.3,
            s=10,
            color="gray",
            label="Partial residuals",
        )

    # Add zero reference line
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Labels and title
    if title is None:
        if is_additive:
            title = f"Smooth Term: {term_name}"
        else:
            title = "Smooth Function"
    ax.set_title(title)

    if xlabel is None:
        if is_additive:
            xlabel = f"x{smooth_term.variable}"
        else:
            xlabel = "x"
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = "f(x)"
    ax.set_ylabel(ylabel)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_all_smooths(
    result: AdditiveGAMResult,
    ncols: int = 2,
    confidence_level: float = 0.95,
    figsize_per_plot: tuple[int, int] = (6, 4),
    partial_residuals: bool = True,
) -> Figure:
    """Plot all smooth terms in a grid layout.

    Parameters
    ----------
    result : AdditiveGAMResult
        Fitted additive GAM result.
    ncols : int, default=2
        Number of columns in grid.
    confidence_level : float, default=0.95
        Confidence level for bands.
    figsize_per_plot : tuple, default=(6, 4)
        Size of each subplot.
    partial_residuals : bool, default=True
        Whether to show partial residuals.

    Returns
    -------
    fig : matplotlib Figure
        Figure with all smooth terms plotted.

    Examples
    --------
    >>> from aurora.models.gam import fit_additive_gam, SmoothTerm
    >>> import numpy as np
    >>> X = np.random.randn(200, 3)
    >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2 + 0.1 * np.random.randn(200)
    >>> result = fit_additive_gam(
    ...     X, y,
    ...     smooth_terms=[
    ...         SmoothTerm(variable=0),
    ...         SmoothTerm(variable=1),
    ...         SmoothTerm(variable=2)
    ...     ]
    ... )
    >>> fig = plot_all_smooths(result, ncols=3)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    from aurora.models.gam.additive import AdditiveGAMResult

    if not isinstance(result, AdditiveGAMResult):
        raise TypeError(
            "plot_all_smooths() requires AdditiveGAMResult. "
            "For univariate GAM, use plot_smooth() instead."
        )

    n_terms = result.n_smooth_terms_
    nrows = int(np.ceil(n_terms / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )

    # Flatten axes for easier indexing
    if n_terms == 1:
        if nrows * ncols == 1:
            # Single subplot - axes is a single Axes object
            axes = np.array([axes])
        else:
            # Multiple subplots but only 1 term - axes is array
            axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each smooth term
    for i, term in enumerate(result.smooth_terms):
        term_name = f"s({term.variable})"

        plot_smooth(
            result,
            term=term_name,
            confidence_level=confidence_level,
            partial_residuals=partial_residuals,
            ax=axes[i],
        )

    # Hide unused subplots
    for j in range(n_terms, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    return fig


__all__ = ["plot_smooth", "plot_all_smooths"]
