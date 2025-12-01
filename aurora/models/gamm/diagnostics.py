"""Diagnostic and interpretation utilities for GAMM models."""

import numpy as np
from numpy.typing import NDArray


def interpret_variance_components(
    variance_components: list[NDArray[np.floating]],
    group_names: list[str] | None = None,
) -> str:
    """Interpret variance-covariance components of random effects.

    Parameters
    ----------
    variance_components : list of ndarray
        List of variance-covariance matrices for each grouping variable.
    group_names : list of str, optional
        Names of grouping variables. If None, uses generic names.

    Returns
    -------
    interpretation : str
        Human-readable interpretation of variance components.

    Examples
    --------
    >>> from aurora.models.gamm import fit_gamm
    >>> result = fit_gamm(formula='y ~ x + (1 + x | subject)', data=df)
    >>> print(interpret_variance_components(
    ...     result.variance_components,
    ...     list(result.random_effects.keys())
    ... ))
    """
    if group_names is None:
        group_names = [f"Group_{i+1}" for i in range(len(variance_components))]

    lines = []
    lines.append("=" * 75)
    lines.append("Interpretación de Componentes de Varianza")
    lines.append("=" * 75)
    lines.append("")

    for i, (vc, group_name) in enumerate(zip(variance_components, group_names)):
        lines.append(f"Grupo: {group_name}")
        lines.append("-" * 75)

        if vc.shape[0] == 1:
            # Single variance component (random intercept only)
            variance = vc[0, 0]
            sd = np.sqrt(variance)
            lines.append(f"  Tipo: Intercepto aleatorio únicamente")
            lines.append(f"  Varianza: {variance:.4f}")
            lines.append(f"  Desviación estándar: {sd:.4f}")
            lines.append("")
            lines.append(f"  Interpretación:")
            lines.append(
                f"    - Hay una variabilidad de ±{sd:.2f} unidades entre {group_name}s"
            )
            lines.append(
                f"    - Aproximadamente 95% de {group_name}s tienen interceptos"
            )
            lines.append(f"      aleatorios dentro de ±{1.96*sd:.2f} unidades del promedio")

        elif vc.shape[0] == 2:
            # Random intercept + slope
            var_intercept = vc[0, 0]
            var_slope = vc[1, 1]
            cov = vc[0, 1]

            sd_intercept = np.sqrt(var_intercept)
            sd_slope = np.sqrt(var_slope)
            correlation = cov / (sd_intercept * sd_slope)

            lines.append(f"  Tipo: Intercepto y pendiente aleatoria")
            lines.append("")
            lines.append(f"  Componente 1 (Intercepto):")
            lines.append(f"    Varianza: {var_intercept:.4f}")
            lines.append(f"    Desviación estándar: {sd_intercept:.4f}")
            lines.append("")
            lines.append(f"  Componente 2 (Pendiente):")
            lines.append(f"    Varianza: {var_slope:.4f}")
            lines.append(f"    Desviación estándar: {sd_slope:.4f}")
            lines.append("")
            lines.append(f"  Covarianza: {cov:.4f}")
            lines.append(f"  Correlación: {correlation:.4f}")
            lines.append("")
            lines.append(f"  Interpretación:")
            lines.append(
                f"    - Variabilidad en niveles basales: ±{sd_intercept:.2f} unidades"
            )
            lines.append(
                f"    - Variabilidad en tasas de cambio: ±{sd_slope:.2f} unidades/unidad de X"
            )

            if abs(correlation) < 0.3:
                lines.append(f"    - Correlación débil ({correlation:.3f}):")
                lines.append(
                    f"      Los {group_name}s con mayores valores basales no necesariamente"
                )
                lines.append(f"      tienen mayores (o menores) tasas de cambio")
            elif correlation > 0.3:
                lines.append(f"    - Correlación positiva moderada/fuerte ({correlation:.3f}):")
                lines.append(
                    f"      Los {group_name}s con mayores valores basales tienden a"
                )
                lines.append(f"      tener mayores tasas de cambio")
            elif correlation < -0.3:
                lines.append(f"    - Correlación negativa moderada/fuerte ({correlation:.3f}):")
                lines.append(
                    f"      Los {group_name}s con mayores valores basales tienden a"
                )
                lines.append(f"      tener menores tasas de cambio (efecto compensatorio)")

        else:
            # Multiple components
            lines.append(f"  Tipo: {vc.shape[0]} componentes aleatorios")
            lines.append(f"  Matriz de varianza-covarianza:")
            for row in vc:
                row_str = "    " + "  ".join([f"{val:>10.4f}" for val in row])
                lines.append(row_str)

            lines.append("")
            lines.append(f"  Desviaciones estándar:")
            for j in range(vc.shape[0]):
                lines.append(f"    Componente {j+1}: {np.sqrt(vc[j, j]):.4f}")

        lines.append("")

    lines.append("=" * 75)
    return "\n".join(lines)


def compute_r2_conditional_marginal(
    result,
) -> tuple[float, float]:
    """Compute marginal and conditional R² for GAMM.

    Based on Nakagawa & Schielzeth (2013) method for mixed models.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM result.

    Returns
    -------
    r2_marginal : float
        R² marginal - variance explained by fixed effects only.
    r2_conditional : float
        R² conditional - variance explained by fixed + random effects.

    References
    ----------
    Nakagawa, S., & Schielzeth, H. (2013). A general and simple method for
    obtaining R2 from generalized linear mixed-effects models. Methods in
    Ecology and Evolution, 4(2), 133-142.

    Examples
    --------
    >>> from aurora.models.gamm import fit_gamm
    >>> result = fit_gamm(formula='y ~ x + (1|subject)', data=df)
    >>> r2_m, r2_c = compute_r2_conditional_marginal(result)
    >>> print(f"R² marginal: {r2_m:.3f}, R² conditional: {r2_c:.3f}")
    """
    # Variance of fixed effects
    if result._X_parametric is not None:
        fixed_pred = result._X_parametric @ result.beta_parametric
        var_fixed = np.var(fixed_pred)
    else:
        # Approximate from fitted values and random effects
        var_fixed = np.var(result.fitted_values) - np.var(result.residuals)

    # Variance of random effects
    var_random = sum([np.trace(vc) / vc.shape[0] for vc in result.variance_components])

    # Residual variance
    var_residual = result.residual_variance

    # Total variance
    var_total = var_fixed + var_random + var_residual

    # R² calculations
    r2_marginal = var_fixed / var_total
    r2_conditional = (var_fixed + var_random) / var_total

    return r2_marginal, r2_conditional


def plot_diagnostics(result, figsize=(12, 10)):
    """Create diagnostic plots for GAMM model.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM result.
    figsize : tuple, default=(12, 10)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with diagnostic plots.
    axes : array of matplotlib.axes.Axes
        Array of axes objects.

    Examples
    --------
    >>> from aurora.models.gamm import fit_gamm
    >>> result = fit_gamm(formula='y ~ x + (1|subject)', data=df)
    >>> fig, axes = plot_diagnostics(result)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(result.fitted_values, result.residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Valores Ajustados")
    axes[0, 0].set_ylabel("Residuos")
    axes[0, 0].set_title("Residuos vs Valores Ajustados")
    axes[0, 0].grid(True, alpha=0.3)

    # Add smoother to check for patterns
    try:
        from scipy.interpolate import make_interp_spline

        sorted_idx = np.argsort(result.fitted_values)
        x_smooth = result.fitted_values[sorted_idx]
        y_smooth = result.residuals[sorted_idx]
        # Use every 10th point for smoother
        step = max(1, len(x_smooth) // 50)
        if len(x_smooth[::step]) > 3:
            spline = make_interp_spline(x_smooth[::step], y_smooth[::step], k=3)
            x_plot = np.linspace(x_smooth.min(), x_smooth.max(), 100)
            axes[0, 0].plot(x_plot, spline(x_plot), "b-", linewidth=2, alpha=0.7)
    except:
        pass

    # 2. Q-Q Plot
    stats.probplot(result.residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot Normal")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scale-Location (√|residuals| vs fitted)
    sqrt_abs_resid = np.sqrt(np.abs(result.residuals))
    axes[1, 0].scatter(result.fitted_values, sqrt_abs_resid, alpha=0.5)
    axes[1, 0].set_xlabel("Valores Ajustados")
    axes[1, 0].set_ylabel("√|Residuos|")
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].grid(True, alpha=0.3)

    # Add smoother
    try:
        from scipy.interpolate import make_interp_spline

        sorted_idx = np.argsort(result.fitted_values)
        x_smooth = result.fitted_values[sorted_idx]
        y_smooth = sqrt_abs_resid[sorted_idx]
        step = max(1, len(x_smooth) // 50)
        if len(x_smooth[::step]) > 3:
            spline = make_interp_spline(x_smooth[::step], y_smooth[::step], k=3)
            x_plot = np.linspace(x_smooth.min(), x_smooth.max(), 100)
            axes[1, 0].plot(x_plot, spline(x_plot), "r-", linewidth=2, alpha=0.7)
    except:
        pass

    # 4. Histogram of residuals
    axes[1, 1].hist(result.residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("Residuos")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].set_title("Distribución de Residuos")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add normal curve overlay
    mu, sigma = result.residuals.mean(), result.residuals.std()
    x = np.linspace(result.residuals.min(), result.residuals.max(), 100)
    # Scale to histogram
    n, bins, _ = axes[1, 1].hist(result.residuals, bins=30, alpha=0)
    bin_width = bins[1] - bins[0]
    scale = len(result.residuals) * bin_width
    axes[1, 1].plot(
        x,
        scale * stats.norm.pdf(x, mu, sigma),
        "r-",
        linewidth=2,
        label="Normal teórica",
    )
    axes[1, 1].legend()

    plt.tight_layout()
    return fig, axes


def plot_random_effects(result, group_name=None, figsize=(15, 5)):
    """Visualize random effects distribution.

    Parameters
    ----------
    result : GAMMResult
        Fitted GAMM result.
    group_name : str, optional
        Name of grouping variable to plot. If None, uses first group.
    figsize : tuple, default=(15, 5)
        Figure size (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with random effects plots.
    axes : array of matplotlib.axes.Axes
        Array of axes objects.

    Examples
    --------
    >>> from aurora.models.gamm import fit_gamm
    >>> result = fit_gamm(formula='y ~ x + (1 + x | subject)', data=df)
    >>> fig, axes = plot_random_effects(result, 'subject')
    >>> plt.show()
    """
    import matplotlib.pyplot as plt

    if group_name is None:
        group_name = list(result.random_effects.keys())[0]

    # Extract random effects for this group
    # result.random_effects[group_name] is a dict: {group_id: array([intercept, slope, ...])}
    random_effects_dict = result.random_effects[group_name]
    
    # Convert dict to array: stack all group effects
    # Sort by group_id to maintain consistent ordering
    group_ids = sorted(random_effects_dict.keys())
    random_effects_list = [random_effects_dict[gid] for gid in group_ids]
    
    # Stack into array where each row is one group's effects
    # Shape: (n_groups, n_components)
    random_effects_matrix = np.vstack(random_effects_list)
    
    # Determine structure
    vc = result.variance_components[
        list(result.random_effects.keys()).index(group_name)
    ]
    n_components = vc.shape[0]
    n_groups = len(group_ids)

    if n_components == 1:
        # Only intercepts
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        re_intercepts = random_effects_matrix[:, 0]

        sorted_idx = np.argsort(re_intercepts)
        ax.scatter(range(n_groups), re_intercepts[sorted_idx], s=50)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel(f"{group_name} (ordenado)")
        ax.set_ylabel("Efecto Aleatorio")
        ax.set_title(f"Interceptos Aleatorios por {group_name}")
        ax.grid(True, alpha=0.3)

        return fig, ax

    elif n_components == 2:
        # Intercepts + slopes
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        re_intercepts = random_effects_matrix[:, 0]
        re_slopes = random_effects_matrix[:, 1]

        # 1. Caterpillar plot - intercepts
        sorted_idx = np.argsort(re_intercepts)
        axes[0].scatter(range(n_groups), re_intercepts[sorted_idx], s=50)
        axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0].set_xlabel(f"{group_name} (ordenado)")
        axes[0].set_ylabel("Efecto Aleatorio (Intercepto)")
        axes[0].set_title(f"Interceptos Aleatorios por {group_name}")
        axes[0].grid(True, alpha=0.3)

        # 2. Caterpillar plot - slopes
        sorted_idx = np.argsort(re_slopes)
        axes[1].scatter(range(n_groups), re_slopes[sorted_idx], s=50)
        axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[1].set_xlabel(f"{group_name} (ordenado)")
        axes[1].set_ylabel("Efecto Aleatorio (Pendiente)")
        axes[1].set_title(f"Pendientes Aleatorias por {group_name}")
        axes[1].grid(True, alpha=0.3)

        # 3. Scatter: intercepts vs slopes
        axes[2].scatter(re_intercepts, re_slopes, s=50, alpha=0.6)
        axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[2].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Intercepto Aleatorio")
        axes[2].set_ylabel("Pendiente Aleatoria")
        axes[2].set_title("Correlación Intercepto-Pendiente")
        axes[2].grid(True, alpha=0.3)

        # Add correlation value
        correlation = vc[0, 1] / (np.sqrt(vc[0, 0]) * np.sqrt(vc[1, 1]))
        axes[2].text(
            0.05,
            0.95,
            f"r = {correlation:.3f}",
            transform=axes[2].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig, axes

    else:
        raise NotImplementedError(
            f"Plotting for {n_components} components not yet implemented"
        )
