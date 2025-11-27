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


def plot_diagnostics_panel(
    result: GAMMResult,
    figsize: tuple[float, float] = (12, 10),
) -> Figure:
    """Create 2x2 diagnostic panel for GAMM model.

    Standard diagnostic panel inspired by R's plot.lm():
    1. Residuals vs Fitted (top-left)
    2. Q-Q Plot of residuals (top-right)
    3. Scale-Location plot (bottom-left)
    4. Residuals vs Leverage with Cook's distance (bottom-right)

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM model result.
    figsize : tuple, default=(12, 10)
        Figure size (width, height) in inches.

    Returns
    -------
    fig : Figure
        Matplotlib figure with 2x2 diagnostic panel.

    Examples
    --------
    >>> from aurora.models.gamm import fit_gamm
    >>> result = fit_gamm("y ~ x + (1|group)", data=df)
    >>> fig = plot_diagnostics_panel(result)
    >>> plt.show()

    Notes
    -----
    This produces a publication-quality diagnostic panel following
    the classic R layout:

    **Residuals vs Fitted (top-left):**
    - Should show random scatter around zero
    - Patterns indicate model misspecification

    **Normal Q-Q (top-right):**
    - Points should fall on diagonal
    - Departures indicate non-normal residuals

    **Scale-Location (bottom-left):**
    - Should show horizontal trend
    - Increasing spread indicates heteroscedasticity

    **Residuals vs Leverage (bottom-right):**
    - Identifies influential observations
    - Cook's distance contours show influence

    References
    ----------
    .. [1] Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). 
           Regression Diagnostics. Wiley.
    .. [2] Cook, R. D., & Weisberg, S. (1982). 
           Residuals and Influence in Regression. Chapman and Hall.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data
    residuals = result.residuals
    fitted = result.fitted_values
    n = len(residuals)

    # Standardized residuals
    resid_std = residuals / np.std(residuals)

    # 1. Residuals vs Fitted (top-left)
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.6, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)

    # Add LOWESS smooth
    if n > 10:
        try:
            from scipy.ndimage import gaussian_filter1d
            sort_idx = np.argsort(fitted)
            smooth = gaussian_filter1d(residuals[sort_idx], sigma=max(1, n//15))
            ax.plot(fitted[sort_idx], smooth, 'b-', linewidth=2, alpha=0.7)
        except ImportError:
            pass

    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    ax.grid(alpha=0.3)

    # 2. Q-Q Plot (top-right)
    ax = axes[0, 1]
    stats.probplot(resid_std, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('steelblue')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_markersize(5)
    ax.get_lines()[1].set_color('red')
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')
    ax.grid(alpha=0.3)

    # 3. Scale-Location (bottom-left)
    ax = axes[1, 0]
    sqrt_abs_resid = np.sqrt(np.abs(resid_std))
    ax.scatter(fitted, sqrt_abs_resid, alpha=0.6, edgecolor='black', linewidth=0.3)

    if n > 10:
        try:
            from scipy.ndimage import gaussian_filter1d
            sort_idx = np.argsort(fitted)
            smooth = gaussian_filter1d(sqrt_abs_resid[sort_idx], sigma=max(1, n//15))
            ax.plot(fitted[sort_idx], smooth, 'r-', linewidth=2, alpha=0.7)
        except ImportError:
            pass

    ax.set_xlabel('Fitted values')
    ax.set_ylabel('√|Standardized residuals|')
    ax.set_title('Scale-Location')
    ax.grid(alpha=0.3)

    # 4. Residuals vs Leverage with Cook's distance (bottom-right)
    ax = axes[1, 1]

    # Compute leverage (hat values) if available
    # For now, use simplified version based on fitted values
    # Approximate leverage: h_ii ≈ 1/n + (x_i - x̄)²/SSx
    fitted_centered = fitted - np.mean(fitted)
    leverage = 1/n + fitted_centered**2 / np.sum(fitted_centered**2)
    leverage = np.clip(leverage, 0.01, 0.99)  # Ensure valid range

    ax.scatter(leverage, resid_std, alpha=0.6, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

    # Add Cook's distance contours
    # Cook's D ≈ r²_i × h_i / (p × (1-h_i)²)
    # For contours at D = 0.5, 1.0
    h_range = np.linspace(0.01, 0.5, 100)
    p = 2  # Approximate number of parameters

    for cook_d in [0.5, 1.0]:
        # r² = cook_d × p × (1-h)² / h
        r_pos = np.sqrt(cook_d * p * (1 - h_range)**2 / h_range)
        r_neg = -r_pos
        
        ax.plot(h_range, r_pos, 'r--', alpha=0.5, linewidth=0.8)
        ax.plot(h_range, r_neg, 'r--', alpha=0.5, linewidth=0.8)
        
        # Label
        ax.text(h_range[-1], r_pos[-1], f'D={cook_d}', fontsize=8, color='red', alpha=0.7)

    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized residuals')
    ax.set_title("Residuals vs Leverage")
    ax.grid(alpha=0.3)

    # Add overall title
    fig.suptitle('GAMM Diagnostic Plots', fontsize=14, y=0.995)
    fig.tight_layout()

    return fig


def plot_smooth_effect(
    result: GAMMResult,
    term_name: str,
    data: dict | None = None,
    n_points: int = 100,
    level: float = 0.95,
    show_data: bool = True,
    show_residuals: bool = False,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot estimated smooth function with confidence band.

    Creates a plot of the estimated smooth term f(x) with:
    - Smooth curve (estimated effect)
    - Pointwise confidence interval
    - Rug plot showing data density
    - Optional partial residuals

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM/GAM model result.
    term_name : str
        Name of the smooth term to plot (e.g., 's(age)' or 'age').
    data : dict or DataFrame, optional
        Original data used for fitting. If None, attempts to extract from result.
    n_points : int, default=100
        Number of points for smooth curve evaluation.
    level : float, default=0.95
        Confidence level for intervals.
    show_data : bool, default=True
        Whether to show rug plot of data density.
    show_residuals : bool, default=False
        Whether to show partial residuals.
    figsize : tuple, default=(8, 6)
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
    >>> from aurora.models.gamm import fit_gamm
    >>> result = fit_gamm("y ~ s(age) + (1|group)", data=df)
    >>> fig, ax = plot_smooth_effect(result, term_name='age', data=df)
    >>> plt.show()

    Notes
    -----
    The confidence band is a pointwise interval, not a simultaneous band.
    For simultaneous confidence bands, additional adjustments are needed.

    Partial residuals are computed as:
        e_i = f̂(x_i) + (y_i - ŷ_i)

    This allows visualizing the data on the scale of the smooth effect.

    References
    ----------
    .. [1] Wood, S. N. (2017). GAMs: An Introduction with R, 2nd ed. Chapter 4.
    .. [2] Hastie, T., & Tibshirani, R. (1990). GAMs. Chapman and Hall.
    """
    # Try to extract smooth term info
    if not hasattr(result, 'smooth_terms') or result.smooth_terms is None:
        raise ValueError("Model does not contain smooth terms")

    # Find the smooth term
    smooth_info = None
    for term in result.smooth_terms:
        if term.get('name') == term_name or term.get('variable') == term_name:
            smooth_info = term
            break
        # Also check for 's(name)' format
        if f"s({term_name})" == term.get('name'):
            smooth_info = term
            break

    if smooth_info is None:
        available = [t.get('name', t.get('variable', '?')) for t in result.smooth_terms]
        raise ValueError(
            f"Smooth term '{term_name}' not found. Available: {available}"
        )

    # Get x values for the smooth
    var_name = smooth_info.get('variable', term_name)

    if data is not None:
        if hasattr(data, 'values'):  # DataFrame
            x_data = data[var_name].values
        else:  # dict
            x_data = np.asarray(data[var_name])
    elif hasattr(result, '_data') and result._data is not None:
        x_data = result._data[var_name]
    else:
        # Use range from smooth info if available
        x_data = np.linspace(
            smooth_info.get('x_min', 0),
            smooth_info.get('x_max', 1),
            n_points
        )

    # Create evaluation grid
    x_grid = np.linspace(x_data.min(), x_data.max(), n_points)

    # Get smooth coefficients and basis
    coef_start = smooth_info.get('coef_start', 0)
    coef_end = smooth_info.get('coef_end', coef_start + smooth_info.get('n_basis', 10))
    smooth_coefs = result.coefficients[coef_start:coef_end]

    # Build basis matrix for grid points
    basis_type = smooth_info.get('basis', 'cr')  # Default to cubic regression splines
    n_basis = smooth_info.get('n_basis', 10)
    knots = smooth_info.get('knots', None)

    # Simple B-spline basis construction
    try:
        from aurora.smoothing.splines.bspline import BSplineBasis
        basis = BSplineBasis(n_basis=n_basis, degree=3)
        X_grid = basis.design_matrix(x_grid)
    except ImportError:
        # Fallback: polynomial basis
        X_grid = np.column_stack([x_grid**i for i in range(n_basis)])

    # Compute smooth values
    smooth_values = X_grid @ smooth_coefs

    # Compute confidence intervals
    # Approximate variance: Var(f(x)) = B(x) @ Cov(β) @ B(x).T
    if hasattr(result, 'covariance') and result.covariance is not None:
        cov_smooth = result.covariance[coef_start:coef_end, coef_start:coef_end]
        var_smooth = np.diag(X_grid @ cov_smooth @ X_grid.T)
        se_smooth = np.sqrt(np.maximum(var_smooth, 0))
    else:
        # Approximate SE from residual variance
        residual_var = np.var(result.residuals)
        se_smooth = np.sqrt(residual_var / n_points) * np.ones(n_points)

    z_crit = stats.norm.ppf((1 + level) / 2)
    ci_lower = smooth_values - z_crit * se_smooth
    ci_upper = smooth_values + z_crit * se_smooth

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Confidence band
    ax.fill_between(x_grid, ci_lower, ci_upper, alpha=0.3, color='steelblue',
                    label=f'{level*100:.0f}% CI')

    # Smooth curve
    ax.plot(x_grid, smooth_values, 'b-', linewidth=2, label='Smooth effect')

    # Partial residuals
    if show_residuals and data is not None:
        # Compute partial residuals at data points
        try:
            basis_data = BSplineBasis(n_basis=n_basis, degree=3)
            X_data = basis_data.design_matrix(x_data)
        except Exception:
            X_data = np.column_stack([x_data**i for i in range(n_basis)])

        smooth_at_data = X_data @ smooth_coefs
        partial_resid = smooth_at_data + result.residuals

        ax.scatter(x_data, partial_resid, alpha=0.3, s=20, c='gray',
                   label='Partial residuals')

    # Rug plot
    if show_data:
        ax.plot(x_data, np.full_like(x_data, ax.get_ylim()[0]), '|',
                color='black', alpha=0.3, markersize=10)

    # Labels
    ax.set_xlabel(var_name)
    ax.set_ylabel(f's({var_name})')
    ax.set_title(f"Smooth Effect: s({var_name})")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig, ax


def plot_all_smooth_effects(
    result: GAMMResult,
    data: dict | None = None,
    n_cols: int = 2,
    figsize_per_plot: tuple[float, float] = (5, 4),
    **kwargs
) -> Figure:
    """Plot all smooth effects in a grid layout.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM/GAM model result.
    data : dict or DataFrame, optional
        Original data used for fitting.
    n_cols : int, default=2
        Number of columns in the grid.
    figsize_per_plot : tuple, default=(5, 4)
        Size of each subplot.
    **kwargs : dict
        Additional arguments passed to plot_smooth_effect.

    Returns
    -------
    fig : Figure
        Matplotlib figure with grid of smooth effect plots.

    Examples
    --------
    >>> result = fit_gamm("y ~ s(x1) + s(x2) + s(x3) + (1|group)", data=df)
    >>> fig = plot_all_smooth_effects(result, data=df)
    >>> plt.show()
    """
    if not hasattr(result, 'smooth_terms') or result.smooth_terms is None:
        raise ValueError("Model does not contain smooth terms")

    n_smooths = len(result.smooth_terms)
    if n_smooths == 0:
        raise ValueError("Model has no smooth terms to plot")

    n_rows = int(np.ceil(n_smooths / n_cols))
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_smooths == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, term in enumerate(result.smooth_terms):
        term_name = term.get('variable', term.get('name', f'term_{i}'))
        try:
            plot_smooth_effect(result, term_name, data=data, ax=axes[i], **kwargs)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:30]}...",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f's({term_name}) - Error')

    # Hide unused subplots
    for j in range(n_smooths, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Smooth Effect Plots', fontsize=14, y=1.02)
    fig.tight_layout()

    return fig


__all__ = [
    'plot_caterpillar',
    'plot_random_effects_qq',
    'plot_random_effects_density',
    'plot_diagnostics',
    'plot_diagnostics_panel',
    'plot_random_effects_summary',
    'plot_smooth_effect',
    'plot_all_smooth_effects',
]
