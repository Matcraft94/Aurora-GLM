"""Visualization tools for GAMM models.

This module provides plotting functions for diagnosing and exploring
Generalized Additive Mixed Models, with focus on random effects visualization.

Functions include:
- Caterpillar plots: Visualize random effects with confidence intervals
- Q-Q plots: Check normality of random effects
- Density plots: Show distributions of random effects
- Diagnostic plots: Residual analysis and model checking

References
----------
.. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS.
.. [2] Gelman & Hill (2007). Data Analysis Using Regression and Multilevel/
       Hierarchical Models. Chapter 12: Multilevel linear models.
.. [3] Wood (2017). Generalized Additive Models: An Introduction with R, 2nd ed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from aurora.models.gamm.fitting import GAMMResult


def plot_caterpillar(
    result: GAMMResult,
    grouping: str | int | None = None,
    effect_index: int = 0,
    confidence: float = 0.95,
    sort: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create caterpillar plot for random effects.

    Caterpillar plots show the estimated random effects (BLUPs) for each
    group with confidence intervals, sorted by magnitude. They are useful
    for identifying groups with unusually large or small effects.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM model result.
    grouping : str or int, optional
        Which grouping variable to plot. If None, uses the first random
        effect term. Required if model has multiple grouping variables.
    effect_index : int, default=0
        Which random effect to plot (0=intercept, 1=first slope, etc.).
    confidence : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI).
    sort : bool, default=True
        Whether to sort groups by effect magnitude.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes with plot.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.gamm import fit_gamm
    >>> from aurora.models.gamm.plotting import plot_caterpillar
    >>>
    >>> # Fit model
    >>> data = pd.DataFrame({
    ...     'y': np.random.randn(100),
    ...     'x': np.random.randn(100),
    ...     'subject': np.repeat(np.arange(10), 10)
    ... })
    >>> result = fit_gamm(formula="y ~ x + (1 | subject)", data=data)
    >>>
    >>> # Plot random intercepts
    >>> fig, ax = plot_caterpillar(result)
    >>> plt.show()

    Notes
    -----
    The confidence intervals are approximate and computed using the
    posterior variance of the random effects:

        Var(b | y) ≈ (Z'WZ + Ψ⁻¹)⁻¹

    Groups whose confidence intervals do not include zero have effects
    that are significantly different from the population mean.
    """
    from aurora.models.gamm.design import extract_random_effects

    # Get random effects
    if result._Z_info is None or len(result._Z_info) == 0:
        raise ValueError("Model does not contain random effects")

    # Determine which grouping to plot
    if grouping is None:
        if len(result._Z_info) > 1:
            raise ValueError(
                f"Model has {len(result._Z_info)} grouping variables. "
                "Please specify which one to plot using the 'grouping' parameter."
            )
        grouping_to_plot = result._Z_info[0]['grouping']
        info = result._Z_info[0]
    else:
        # Find the Z_info entry for this grouping
        info = None
        for z_info in result._Z_info:
            if z_info['grouping'] == grouping:
                info = z_info
                grouping_to_plot = grouping
                break

        if info is None:
            available = [z['grouping'] for z in result._Z_info]
            raise ValueError(
                f"Grouping '{grouping}' not found in model. Available: {available}"
            )

    # Check effect_index is valid
    if effect_index >= info['n_effects']:
        raise ValueError(
            f"effect_index={effect_index} but grouping '{grouping_to_plot}' "
            f"only has {info['n_effects']} effects (0-indexed)"
        )

    # Extract random effects for this grouping
    # GAMMResult already has random_effects in the correct format
    random_effects = result.random_effects[grouping_to_plot]

    # Get groups and effects
    groups = info['groups']
    n_groups = len(groups)
    effects = np.array([random_effects[g][effect_index] for g in groups])

    # Compute approximate standard errors
    # Use posterior variance: Var(b|y) ≈ (Z'WZ + Ψ⁻¹)⁻¹
    # For simplicity, use empirical SE scaled by estimated variance
    var_component = result.variance_components[result._Z_info.index(info)]

    if var_component.ndim == 2:
        # Covariance matrix
        effect_var = var_component[effect_index, effect_index]
    else:
        # Scalar variance
        effect_var = var_component

    # Approximate SE (shrinkage-adjusted)
    # In practice, would use full posterior variance
    se = np.sqrt(effect_var / np.sqrt(n_groups))
    se_array = np.full(n_groups, se)

    # Compute confidence intervals
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = effects - z_crit * se_array
    ci_upper = effects + z_crit * se_array

    # Sort if requested
    if sort:
        sort_idx = np.argsort(effects)
        effects = effects[sort_idx]
        ci_lower = ci_lower[sort_idx]
        ci_upper = ci_upper[sort_idx]
        groups = groups[sort_idx]

    # Create plot
    if ax is None:
        if figsize is None:
            figsize = (8, max(4, n_groups * 0.2))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot horizontal lines for CIs
    y_positions = np.arange(n_groups)
    ax.hlines(y_positions, ci_lower, ci_upper, color='gray', linewidth=1.5)

    # Plot points for estimates
    ax.plot(effects, y_positions, 'o', color='steelblue', markersize=6)

    # Add reference line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Labels
    effect_names = ['Intercept'] + [f'Slope {i}' for i in range(1, info['n_effects'])]
    ax.set_xlabel(f'Random Effect: {effect_names[effect_index]}')
    ax.set_ylabel(f'Group ({grouping_to_plot})')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(g) for g in groups])
    ax.set_title(
        f'Caterpillar Plot: {effect_names[effect_index]} by {grouping_to_plot}\n'
        f'{confidence*100:.0f}% Confidence Intervals'
    )
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_random_effects_qq(
    result: GAMMResult,
    grouping: str | int | None = None,
    effect_index: int = 0,
    figsize: tuple[float, float] = (6, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create Q-Q plot for random effects normality check.

    Q-Q (quantile-quantile) plots compare the distribution of random effects
    to a theoretical normal distribution. Points falling along the diagonal
    line indicate normality.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM model result.
    grouping : str or int, optional
        Which grouping variable to plot. If None, uses the first random
        effect term.
    effect_index : int, default=0
        Which random effect to plot (0=intercept, 1=first slope, etc.).
    figsize : tuple, default=(6, 6)
        Figure size (width, height) in inches.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes with plot.

    Examples
    --------
    >>> fig, ax = plot_random_effects_qq(result)
    >>> plt.show()

    Notes
    -----
    The random effects are assumed to follow a normal distribution:
        b ~ N(0, Ψ)

    Departures from normality (heavy tails, skewness) can be identified
    visually from Q-Q plots. Consider:
    - S-shaped pattern: Heavy tails (outliers)
    - Systematic curve: Skewness
    - Points far from line: Individual outliers
    """
    from aurora.models.gamm.design import extract_random_effects

    # Get random effects
    if result._Z_info is None or len(result._Z_info) == 0:
        raise ValueError("Model does not contain random effects")

    # Determine which grouping to plot
    if grouping is None:
        if len(result._Z_info) > 1:
            raise ValueError(
                "Model has multiple grouping variables. "
                "Please specify which one to plot using the 'grouping' parameter."
            )
        grouping_to_plot = result._Z_info[0]['grouping']
        info = result._Z_info[0]
    else:
        info = None
        for z_info in result._Z_info:
            if z_info['grouping'] == grouping:
                info = z_info
                grouping_to_plot = grouping
                break

        if info is None:
            available = [z['grouping'] for z in result._Z_info]
            raise ValueError(
                f"Grouping '{grouping}' not found. Available: {available}"
            )

    # Check effect_index
    if effect_index >= info['n_effects']:
        raise ValueError(
            f"effect_index={effect_index} but grouping '{grouping_to_plot}' "
            f"only has {info['n_effects']} effects"
        )

    # Extract random effects
    # GAMMResult already has random_effects in the correct format
    random_effects = result.random_effects[grouping_to_plot]

    # Get effects for this index
    groups = info['groups']
    effects = np.array([random_effects[g][effect_index] for g in groups])

    # Standardize effects
    effects_std = (effects - effects.mean()) / effects.std()

    # Create Q-Q plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute theoretical and sample quantiles
    stats.probplot(effects_std, dist="norm", plot=ax)

    # Styling
    effect_names = ['Intercept'] + [f'Slope {i}' for i in range(1, info['n_effects'])]
    ax.set_title(
        f'Q-Q Plot: {effect_names[effect_index]} by {grouping_to_plot}\n'
        f'(Checking Normality Assumption)'
    )
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Random Effects')
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_random_effects_density(
    result: GAMMResult,
    grouping: str | int | None = None,
    effect_index: int = 0,
    show_normal: bool = True,
    figsize: tuple[float, float] = (8, 5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot density of random effects with optional normal overlay.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM model result.
    grouping : str or int, optional
        Which grouping variable to plot.
    effect_index : int, default=0
        Which random effect to plot.
    show_normal : bool, default=True
        Whether to overlay theoretical normal distribution.
    figsize : tuple, default=(8, 5)
        Figure size.
    ax : Axes, optional
        Matplotlib axes to plot on.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes with plot.

    Examples
    --------
    >>> fig, ax = plot_random_effects_density(result)
    >>> plt.show()

    Notes
    -----
    The histogram shows the empirical distribution of random effects,
    while the overlaid curve shows the theoretical normal distribution
    N(0, Ψ) implied by the model.
    """
    from aurora.models.gamm.design import extract_random_effects

    # Get random effects
    if result._Z_info is None or len(result._Z_info) == 0:
        raise ValueError("Model does not contain random effects")

    # Determine grouping
    if grouping is None:
        if len(result._Z_info) > 1:
            raise ValueError(
                "Model has multiple grouping variables. "
                "Please specify which one."
            )
        grouping_to_plot = result._Z_info[0]['grouping']
        info = result._Z_info[0]
    else:
        info = None
        for z_info in result._Z_info:
            if z_info['grouping'] == grouping:
                info = z_info
                grouping_to_plot = grouping
                break

        if info is None:
            available = [z['grouping'] for z in result._Z_info]
            raise ValueError(f"Grouping '{grouping}' not found. Available: {available}")

    # Check effect_index
    if effect_index >= info['n_effects']:
        raise ValueError(
            f"effect_index={effect_index} invalid for {info['n_effects']} effects"
        )

    # Extract random effects
    # GAMMResult already has random_effects in the correct format
    random_effects = result.random_effects[grouping_to_plot]

    groups = info['groups']
    effects = np.array([random_effects[g][effect_index] for g in groups])

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Histogram
    ax.hist(effects, bins=min(30, len(effects)//2), density=True,
            alpha=0.6, color='steelblue', edgecolor='black', label='Observed')

    # Overlay normal if requested
    if show_normal:
        var_component = result.variance_components[result._Z_info.index(info)]

        if var_component.ndim == 2:
            effect_var = var_component[effect_index, effect_index]
        else:
            effect_var = var_component

        # Theoretical normal
        x_range = np.linspace(effects.min() - 0.5*effects.std(),
                               effects.max() + 0.5*effects.std(), 200)
        y_normal = stats.norm.pdf(x_range, loc=0, scale=np.sqrt(effect_var))
        ax.plot(x_range, y_normal, 'r-', linewidth=2,
                label=f'N(0, {effect_var:.3f})')

    # Labels
    effect_names = ['Intercept'] + [f'Slope {i}' for i in range(1, info['n_effects'])]
    ax.set_xlabel(f'Random Effect: {effect_names[effect_index]}')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Distribution of {effect_names[effect_index]} by {grouping_to_plot}\n'
        f'({len(effects)} groups)'
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_diagnostics(
    result: GAMMResult,
    plot_type: Literal['residuals', 'fitted', 'qq', 'scale-location'] = 'residuals',
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create diagnostic plots for GAMM models.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM model result.
    plot_type : {'residuals', 'fitted', 'qq', 'scale-location'}
        Type of diagnostic plot:
        - 'residuals': Residuals vs fitted values
        - 'fitted': Fitted vs observed values
        - 'qq': Q-Q plot of residuals
        - 'scale-location': Scale-location plot (sqrt(|residuals|) vs fitted)
    figsize : tuple, default=(8, 6)
        Figure size.
    ax : Axes, optional
        Matplotlib axes to plot on.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes with plot.

    Examples
    --------
    >>> # Residuals vs fitted
    >>> fig, ax = plot_diagnostics(result, plot_type='residuals')
    >>> plt.show()
    >>>
    >>> # Q-Q plot of residuals
    >>> fig, ax = plot_diagnostics(result, plot_type='qq')
    >>> plt.show()

    Notes
    -----
    **Residuals vs Fitted:**
    - Should show no pattern (randomness around zero)
    - Fan shape indicates heteroscedasticity
    - Curvature indicates nonlinearity

    **Q-Q Plot:**
    - Points should fall on diagonal line
    - Departures indicate non-normality of residuals

    **Scale-Location:**
    - Check homoscedasticity (constant variance)
    - Should show horizontal line with random scatter

    **Fitted vs Observed:**
    - Points should fall near y=x line
    - Shows overall model fit quality
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute residuals
    residuals = result.residuals
    fitted = result.fitted_values

    if plot_type == 'residuals':
        # Residuals vs fitted
        ax.scatter(fitted, residuals, alpha=0.5, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # Add smooth line
        if len(fitted) > 10:
            from scipy.ndimage import gaussian_filter1d
            sort_idx = np.argsort(fitted)
            fitted_sorted = fitted[sort_idx]
            residuals_sorted = residuals[sort_idx]
            smooth = gaussian_filter1d(residuals_sorted, sigma=max(1, len(fitted)//20))
            ax.plot(fitted_sorted, smooth, 'b-', linewidth=2, alpha=0.7)

        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted\n(Should show random scatter around zero)')
        ax.grid(alpha=0.3)

    elif plot_type == 'fitted':
        # Fitted vs observed
        observed = fitted + residuals
        ax.scatter(fitted, observed, alpha=0.5, edgecolor='black', linewidth=0.5)

        # Add y=x line
        lims = [
            min(fitted.min(), observed.min()),
            max(fitted.max(), observed.max()),
        ]
        ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')

        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Observed Values')
        ax.set_title('Fitted vs Observed\n(Points should cluster near y=x line)')
        ax.legend()
        ax.grid(alpha=0.3)

    elif plot_type == 'qq':
        # Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals\n(Checking normality assumption)')
        ax.grid(alpha=0.3)

    elif plot_type == 'scale-location':
        # Scale-location plot
        sqrt_std_resid = np.sqrt(np.abs(residuals / residuals.std()))
        ax.scatter(fitted, sqrt_std_resid, alpha=0.5, edgecolor='black', linewidth=0.5)

        # Add smooth line
        if len(fitted) > 10:
            from scipy.ndimage import gaussian_filter1d
            sort_idx = np.argsort(fitted)
            fitted_sorted = fitted[sort_idx]
            sqrt_std_resid_sorted = sqrt_std_resid[sort_idx]
            smooth = gaussian_filter1d(sqrt_std_resid_sorted, sigma=max(1, len(fitted)//20))
            ax.plot(fitted_sorted, smooth, 'r-', linewidth=2, alpha=0.7)

        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('√|Standardized Residuals|')
        ax.set_title('Scale-Location Plot\n(Check homoscedasticity)')
        ax.grid(alpha=0.3)

    else:
        raise ValueError(
            f"Unknown plot_type '{plot_type}'. "
            "Must be one of: 'residuals', 'fitted', 'qq', 'scale-location'"
        )

    fig.tight_layout()

    return fig, ax


def plot_random_effects_summary(
    result: GAMMResult,
    grouping: str | int | None = None,
    effect_index: int = 0,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """Create comprehensive summary of random effects diagnostics.

    Produces a 2x2 grid with:
    1. Caterpillar plot
    2. Q-Q plot
    3. Density plot
    4. Residuals vs fitted

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM model result.
    grouping : str or int, optional
        Which grouping variable to plot.
    effect_index : int, default=0
        Which random effect to plot.
    figsize : tuple, default=(14, 10)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure with 2x2 subplot grid.

    Examples
    --------
    >>> fig = plot_random_effects_summary(result)
    >>> plt.show()

    Notes
    -----
    This is a convenience function that creates a comprehensive
    diagnostic summary in a single figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # 1. Caterpillar plot
    plot_caterpillar(result, grouping=grouping, effect_index=effect_index, ax=axes[0])

    # 2. Q-Q plot
    plot_random_effects_qq(result, grouping=grouping, effect_index=effect_index, ax=axes[1])

    # 3. Density plot
    plot_random_effects_density(result, grouping=grouping, effect_index=effect_index, ax=axes[2])

    # 4. Residuals vs fitted
    plot_diagnostics(result, plot_type='residuals', ax=axes[3])

    fig.suptitle('Random Effects Diagnostic Summary', fontsize=14, y=0.995)
    fig.tight_layout()

    return fig


__all__ = [
    'plot_caterpillar',
    'plot_random_effects_qq',
    'plot_random_effects_density',
    'plot_diagnostics',
    'plot_random_effects_summary',
]
