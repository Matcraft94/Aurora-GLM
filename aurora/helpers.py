"""High-level convenience functions for Aurora-GLM.

This module provides unified helper functions that work with any Aurora result:

- summary(result): Formatted model summary
- plot(result): Diagnostic plots
- compare(*results): Model comparison

These are also available at the package level:
>>> import aurora
>>> aurora.summary(result)
>>> aurora.plot(result)
>>> aurora.compare(model1, model2, model3)

Examples
--------
>>> from aurora import fit_glm, summary, plot, compare
>>> 
>>> # Fit models
>>> result1 = fit_glm(X, y, family=Gaussian())
>>> result2 = fit_glm(X, y, family=Poisson())
>>> 
>>> # Get summaries
>>> print(summary(result1))
>>> 
>>> # Plot diagnostics
>>> plot(result1, kind='diagnostics')
>>> 
>>> # Compare models
>>> compare(result1, result2, criterion='aic')
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from .models.base.base_result import BaseResult, ResultProtocol
    import matplotlib.pyplot as plt


# =============================================================================
# Summary Function
# =============================================================================

def summary(
    result: Any,
    *,
    style: Literal['default', 'brief', 'detailed'] = 'default',
    print_output: bool = True,
) -> str:
    """Generate a formatted summary of a fitted model result.
    
    This is a unified interface for summarizing any Aurora model result.
    It automatically dispatches to the appropriate summary method based
    on the result type.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result from fit_glm, fit_gam, fit_gamm, etc.
    style : {'default', 'brief', 'detailed'}, default='default'
        Summary style:
        - 'default': Standard coefficient table with fit statistics
        - 'brief': One-line summary with key metrics
        - 'detailed': Full output including diagnostics
    print_output : bool, default=True
        If True, print the summary. If False, only return the string.
        
    Returns
    -------
    summary_str : str
        Formatted summary string.
        
    Examples
    --------
    >>> from aurora import fit_glm, summary, Gaussian
    >>> result = fit_glm(X, y, family=Gaussian())
    >>> print(summary(result))
    ======================================================================
                          Linear Model Results
    ======================================================================
    No. Observations:        100    R-squared:          0.8542
    ...
    
    >>> # Brief summary
    >>> summary(result, style='brief')
    'GLM(n=100, converged=True, R²=0.854)'
    
    >>> # Detailed with diagnostics
    >>> summary(result, style='detailed')
    
    Notes
    -----
    The function works with any result object that has a `.summary()` method.
    For custom result classes, implement `.summary()` to integrate with this.
    
    See Also
    --------
    plot : Diagnostic plots for model results
    compare : Compare multiple models
    """
    # Check for summary method
    if not hasattr(result, 'summary'):
        raise TypeError(
            f"Cannot summarize object of type {type(result).__name__}. "
            "Expected a model result with a .summary() method."
        )
    
    # Generate summary based on style
    if style == 'default':
        summary_str = result.summary()
    elif style == 'brief':
        summary_str = _brief_summary(result)
    elif style == 'detailed':
        summary_str = _detailed_summary(result)
    else:
        raise ValueError(f"Unknown style '{style}'. Use 'default', 'brief', or 'detailed'.")
    
    if print_output:
        print(summary_str)
    
    return summary_str


def _brief_summary(result: Any) -> str:
    """Generate a one-line brief summary."""
    parts = []
    
    # Class name
    class_name = type(result).__name__
    if class_name.endswith('Result'):
        class_name = class_name[:-6]  # Remove 'Result' suffix
    parts.append(class_name)
    
    # Observations
    if hasattr(result, 'n_obs_'):
        parts.append(f"n={result.n_obs_}")
    elif hasattr(result, 'n_observations'):
        parts.append(f"n={result.n_observations}")
    
    # Convergence
    if hasattr(result, 'converged_'):
        parts.append(f"converged={result.converged_}")
    elif hasattr(result, 'converged'):
        parts.append(f"converged={result.converged}")
    
    # R² for linear models
    if hasattr(result, 'r_squared'):
        r2 = result.r_squared
        if not np.isnan(r2):
            parts.append(f"R²={r2:.3f}")
    
    # Log-likelihood for mixed models
    if hasattr(result, 'log_likelihood_'):
        parts.append(f"LL={result.log_likelihood_:.2f}")
    
    # AIC if available
    if hasattr(result, 'aic'):
        aic = result.aic
        if not np.isnan(aic):
            parts.append(f"AIC={aic:.1f}")
    
    return f"{parts[0]}({', '.join(parts[1:])})"


def _detailed_summary(result: Any) -> str:
    """Generate detailed summary with diagnostics."""
    lines = []
    
    # Base summary
    lines.append(result.summary())
    lines.append("")
    
    # Diagnostics section
    lines.append("=" * 70)
    lines.append("Diagnostics")
    lines.append("=" * 70)
    
    # Residual statistics
    if hasattr(result, 'residuals'):
        try:
            resid = result.residuals
            lines.append(f"Residuals:")
            lines.append(f"  Min:    {np.min(resid):>12.4f}")
            lines.append(f"  1Q:     {np.percentile(resid, 25):>12.4f}")
            lines.append(f"  Median: {np.median(resid):>12.4f}")
            lines.append(f"  3Q:     {np.percentile(resid, 75):>12.4f}")
            lines.append(f"  Max:    {np.max(resid):>12.4f}")
            lines.append("")
        except Exception:
            pass
    
    # Fit statistics
    if hasattr(result, 'r_squared'):
        r2 = result.r_squared
        if not np.isnan(r2):
            lines.append(f"R-squared:          {r2:>12.4f}")
    
    if hasattr(result, 'adj_r_squared'):
        adj_r2 = result.adj_r_squared
        if not np.isnan(adj_r2):
            lines.append(f"Adjusted R-squared: {adj_r2:>12.4f}")
    
    if hasattr(result, 'aic'):
        aic = getattr(result, 'aic', np.nan)
        if not np.isnan(aic):
            lines.append(f"AIC:                {aic:>12.2f}")
    
    if hasattr(result, 'bic'):
        bic = getattr(result, 'bic', np.nan)
        if not np.isnan(bic):
            lines.append(f"BIC:                {bic:>12.2f}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Plot Function
# =============================================================================

def plot(
    result: Any,
    *,
    kind: Literal['diagnostics', 'residuals', 'qq', 'smooth', 'all'] = 'diagnostics',
    ax: Any = None,
    **kwargs
) -> Any:
    """Create diagnostic plots for a fitted model result.
    
    This is a unified interface for plotting any Aurora model result.
    It automatically creates appropriate plots based on the model type.
    
    Parameters
    ----------
    result : ModelResult
        A fitted model result from fit_glm, fit_gam, fit_gamm, etc.
    kind : {'diagnostics', 'residuals', 'qq', 'smooth', 'all'}, default='diagnostics'
        Type of plot:
        - 'diagnostics': Standard 4-panel diagnostic plot
        - 'residuals': Residuals vs fitted values
        - 'qq': Q-Q plot of residuals
        - 'smooth': Smooth function plots (GAM/GAMM only)
        - 'all': All available plots
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to the underlying plot function.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or list of Figures
        The created figure(s).
        
    Examples
    --------
    >>> from aurora import fit_glm, plot, Gaussian
    >>> result = fit_glm(X, y, family=Gaussian())
    >>> 
    >>> # Standard diagnostics
    >>> plot(result)
    >>> 
    >>> # Just residuals
    >>> plot(result, kind='residuals')
    >>> 
    >>> # For GAM, plot smooth terms
    >>> gam_result = fit_gam(X, y)
    >>> plot(gam_result, kind='smooth')
    
    See Also
    --------
    summary : Formatted model summary
    compare : Compare multiple models
    plot_diagnostics : Low-level diagnostic plots
    plot_smooth : Low-level smooth term plots
    """
    # Import visualization functions lazily
    from .visualization import (
        plot_diagnostics_panel,
        plot_smooth,
        plot_all_smooths,
    )
    
    if kind == 'diagnostics':
        return plot_diagnostics_panel(result, ax=ax, **kwargs)
    
    elif kind == 'residuals':
        return _plot_residuals(result, ax=ax, **kwargs)
    
    elif kind == 'qq':
        return _plot_qq(result, ax=ax, **kwargs)
    
    elif kind == 'smooth':
        # Check if GAM result
        if hasattr(result, 'smooth_terms') or hasattr(result, 'smooth_info'):
            return plot_all_smooths(result, **kwargs)
        else:
            raise ValueError(
                "kind='smooth' requires a GAM or GAMM result with smooth terms."
            )
    
    elif kind == 'all':
        figs = []
        figs.append(plot_diagnostics_panel(result, **kwargs))
        if hasattr(result, 'smooth_terms') or hasattr(result, 'smooth_info'):
            figs.append(plot_all_smooths(result, **kwargs))
        return figs
    
    else:
        raise ValueError(
            f"Unknown plot kind '{kind}'. "
            "Use 'diagnostics', 'residuals', 'qq', 'smooth', or 'all'."
        )


def _plot_residuals(result: Any, ax: Any = None, **kwargs) -> Any:
    """Plot residuals vs fitted values."""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    fitted = result.fitted_values
    residuals = result.residuals
    
    ax.scatter(fitted, residuals, alpha=0.6, **kwargs)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')
    
    return fig


def _plot_qq(result: Any, ax: Any = None, **kwargs) -> Any:
    """Create Q-Q plot of residuals."""
    import matplotlib.pyplot as plt
    from scipy import stats
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
    
    residuals = result.residuals
    
    # Standardize residuals
    std_resid = (residuals - np.mean(residuals)) / np.std(residuals)
    
    stats.probplot(std_resid, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot')
    
    return fig


# =============================================================================
# Compare Function
# =============================================================================

def compare(
    *results: Any,
    criterion: Literal['aic', 'bic', 'loglik', 'all'] = 'aic',
    names: Sequence[str] | None = None,
    print_output: bool = True,
) -> dict[str, Any]:
    """Compare multiple fitted models.
    
    This function provides a unified interface for comparing different
    model specifications using information criteria or likelihood ratio tests.
    
    Parameters
    ----------
    *results : ModelResult
        Two or more fitted model results to compare.
    criterion : {'aic', 'bic', 'loglik', 'all'}, default='aic'
        Criterion for comparison:
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion  
        - 'loglik': Log-likelihood
        - 'all': All available criteria
    names : sequence of str, optional
        Names for the models. If None, uses 'Model 1', 'Model 2', etc.
    print_output : bool, default=True
        If True, print the comparison table.
        
    Returns
    -------
    comparison : dict
        Dictionary with comparison results:
        - 'summary': Formatted comparison table (str)
        - 'criteria': DataFrame or dict with criterion values
        - 'best': Index of best model
        - 'delta': Differences from best model
        
    Examples
    --------
    >>> from aurora import fit_glm, compare, Gaussian, Poisson
    >>> 
    >>> # Fit different models
    >>> m1 = fit_glm(X, y, family=Gaussian())
    >>> m2 = fit_glm(X, y, family=Poisson())
    >>> m3 = fit_glm(X[:, :2], y, family=Gaussian())  # Fewer predictors
    >>> 
    >>> # Compare models
    >>> compare(m1, m2, m3, criterion='aic')
    ======================================================================
                            Model Comparison (AIC)
    ======================================================================
           Model        AIC      ΔAIC   Rank
    ----------------------------------------------------------------------
       Model 1     234.56      0.00      1   ← best
       Model 2     245.12     10.56      2
       Model 3     267.89     33.33      3
    ======================================================================
    
    >>> # With custom names
    >>> compare(m1, m2, names=['Linear', 'Poisson'], criterion='bic')
    
    Notes
    -----
    - Lower AIC/BIC indicates better fit (with parsimony)
    - For nested models, likelihood ratio tests can also be performed
    - Models should be fit on the same data for meaningful comparison
    
    See Also
    --------
    summary : Individual model summaries
    """
    if len(results) < 2:
        raise ValueError("compare() requires at least 2 models to compare.")
    
    # Generate names if not provided
    if names is None:
        names = [f"Model {i+1}" for i in range(len(results))]
    elif len(names) != len(results):
        raise ValueError(
            f"Number of names ({len(names)}) must match "
            f"number of models ({len(results)})."
        )
    
    # Extract criteria values
    metrics = _extract_comparison_metrics(results, criterion)
    
    # Create comparison summary
    summary_str = _format_comparison_table(names, metrics, criterion)
    
    # Find best model
    primary_criterion = criterion if criterion != 'all' else 'aic'
    if primary_criterion in metrics:
        values = metrics[primary_criterion]
        if primary_criterion == 'loglik':
            best_idx = int(np.argmax(values))  # Higher is better
        else:
            best_idx = int(np.argmin(values))  # Lower is better
        
        delta = np.array(values) - values[best_idx]
    else:
        best_idx = 0
        delta = np.zeros(len(results))
    
    if print_output:
        print(summary_str)
    
    return {
        'summary': summary_str,
        'criteria': metrics,
        'names': list(names),
        'best': best_idx,
        'best_model': names[best_idx],
        'delta': delta,
    }


def _extract_comparison_metrics(
    results: tuple[Any, ...],
    criterion: str
) -> dict[str, list[float]]:
    """Extract comparison metrics from results."""
    metrics: dict[str, list[float]] = {}
    
    # Always try to get n_obs and n_params
    n_obs_list = []
    n_params_list = []
    
    for result in results:
        # Number of observations
        if hasattr(result, 'n_obs_'):
            n_obs_list.append(result.n_obs_)
        elif hasattr(result, 'n_observations'):
            n_obs_list.append(result.n_observations)
        else:
            n_obs_list.append(np.nan)
        
        # Number of parameters
        if hasattr(result, 'df_model'):
            n_params_list.append(result.df_model + 1)
        elif hasattr(result, 'coefficients'):
            n_params_list.append(len(result.coefficients))
        else:
            n_params_list.append(np.nan)
    
    metrics['n_obs'] = n_obs_list
    metrics['n_params'] = n_params_list
    
    # Extract requested criteria
    criteria_to_extract = ['aic', 'bic', 'loglik'] if criterion == 'all' else [criterion]
    
    for crit in criteria_to_extract:
        values = []
        for result in results:
            if crit == 'aic':
                values.append(_get_aic(result))
            elif crit == 'bic':
                values.append(_get_bic(result))
            elif crit == 'loglik':
                values.append(_get_loglik(result))
            else:
                values.append(np.nan)
        metrics[crit] = values
    
    return metrics


def _get_aic(result: Any) -> float:
    """Extract or compute AIC."""
    if hasattr(result, 'aic'):
        return result.aic
    
    # Try to compute from log-likelihood
    ll = _get_loglik(result)
    if np.isnan(ll):
        return np.nan
    
    if hasattr(result, 'df_model'):
        k = result.df_model + 1
    elif hasattr(result, 'coefficients'):
        k = len(result.coefficients)
    else:
        return np.nan
    
    return -2 * ll + 2 * k


def _get_bic(result: Any) -> float:
    """Extract or compute BIC."""
    if hasattr(result, 'bic'):
        return result.bic
    
    # Try to compute from log-likelihood
    ll = _get_loglik(result)
    if np.isnan(ll):
        return np.nan
    
    if hasattr(result, 'n_obs_'):
        n = result.n_obs_
    elif hasattr(result, 'n_observations'):
        n = result.n_observations
    else:
        return np.nan
    
    if hasattr(result, 'df_model'):
        k = result.df_model + 1
    elif hasattr(result, 'coefficients'):
        k = len(result.coefficients)
    else:
        return np.nan
    
    return -2 * ll + k * np.log(n)


def _get_loglik(result: Any) -> float:
    """Extract log-likelihood."""
    if hasattr(result, 'log_likelihood_'):
        return result.log_likelihood_
    if hasattr(result, 'loglik'):
        return result.loglik
    if hasattr(result, 'llf'):
        return result.llf
    return np.nan


def _format_comparison_table(
    names: Sequence[str],
    metrics: dict[str, list[float]],
    criterion: str
) -> str:
    """Format comparison table as string."""
    lines = []
    sep = "=" * 70
    
    lines.append(sep)
    lines.append(f"{'Model Comparison':^70}")
    lines.append(sep)
    
    # Determine which criteria to show
    criteria_to_show = ['aic', 'bic', 'loglik'] if criterion == 'all' else [criterion]
    
    # Header
    header_parts = [f"{'Model':>15}"]
    for crit in criteria_to_show:
        header_parts.append(f"{crit.upper():>12}")
    header_parts.append(f"{'Rank':>8}")
    lines.append("  ".join(header_parts))
    lines.append("-" * 70)
    
    # Determine ranking (by first criterion)
    primary = criteria_to_show[0]
    if primary in metrics:
        values = np.array(metrics[primary])
        if primary == 'loglik':
            order = np.argsort(-values)  # Higher is better
        else:
            order = np.argsort(values)   # Lower is better
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
    else:
        ranks = np.arange(1, len(names) + 1)
    
    # Data rows
    for i, name in enumerate(names):
        row_parts = [f"{name:>15}"]
        for crit in criteria_to_show:
            if crit in metrics:
                val = metrics[crit][i]
                if np.isnan(val):
                    row_parts.append(f"{'N/A':>12}")
                else:
                    row_parts.append(f"{val:>12.2f}")
            else:
                row_parts.append(f"{'N/A':>12}")
        
        rank_str = str(int(ranks[i]))
        if ranks[i] == 1:
            rank_str += " ← best"
        row_parts.append(f"{rank_str:>8}")
        
        lines.append("  ".join(row_parts))
    
    lines.append(sep)
    
    return "\n".join(lines)


__all__ = [
    "summary",
    "plot",
    "compare",
]
