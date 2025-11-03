"""Dataclasses encapsulating fitted model state."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np

from ...core.types import Array, ArrayLike
from ...distributions._utils import as_namespace_array, namespace
from ...distributions.base import Family, LinkFunction

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from ...inference.diagnostics import GLMDiagnosticResult


@dataclass(frozen=True)
class ModelResult:
    """Immutable container storing the outcome of a model fitting routine."""

    params: ArrayLike
    fitted_values: ArrayLike
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary representation."""
        return {
            "params": self.params,
            "converged": self.converged,
            "diagnostics": self.diagnostics,
        }

    def predict(self, design_matrix: ArrayLike, *, backend: str = "jax") -> ArrayLike:
        """Dispatch to the selected backend to generate predictions."""
        from ...core.backends import get_backend

        backend_impl = get_backend(backend)
        return backend_impl.array(design_matrix) @ backend_impl.array(self.params)


__all__ = ["ModelResult"]


@dataclass
class GLMResult:
    """Structured container describing the outcome of a GLM fit."""

    coef_: Array
    intercept_: float | None
    family: Family
    link: LinkFunction
    mu_: Array
    eta_: Array
    deviance_: float
    null_deviance_: float
    aic_: float
    bic_: float
    n_iter_: int
    converged_: bool
    _coef_cov: Array | None = None
    _std_errors: Array | None = None
    _p_values: Array | None = None
    _X: Array | None = None
    _y: Array | None = None
    _weights: Array | None = None
    _fit_intercept: bool = True
    _intercept_std_error: float | None = None
    _intercept_p_value: float | None = None
    _diagnostics_cache: "GLMDiagnosticResult" | None = None

    @property
    def std_errors_(self) -> Array:
        """Return standard errors for the fitted coefficients."""

        if self._std_errors is None:
            self._compute_inference()
        return self._std_errors

    @property
    def p_values_(self) -> Array:
        """Return Wald p-values for the fitted coefficients."""

        if self._p_values is None:
            self._compute_inference()
        return self._p_values

    @property
    def coef_cov_(self) -> Array:
        """Return the covariance matrix of the fitted coefficients."""

        if self._coef_cov is None:
            self._compute_inference()
        return self._coef_cov

    def _compute_inference(self) -> None:
        """Compute covariance matrix, standard errors, and Wald p-values."""

        if self._X is None or self._y is None:
            raise RuntimeError("Design matrix and response are required for inference.")

        X_np = _to_numpy(self._X)
        xp = namespace(self.mu_)
        mu_xp = as_namespace_array(self.mu_, xp, like=self.mu_)
        deriv_xp = as_namespace_array(self.link.derivative(mu_xp), xp, like=mu_xp)
        variance_xp = as_namespace_array(self.family.variance(mu_xp), xp, like=mu_xp)
        deriv_np = _to_numpy(deriv_xp)
        variance_np = _to_numpy(variance_xp)

        weights_np = _to_numpy(self._weights) if self._weights is not None else None

        fisher = _fisher_information_numpy(
            X=X_np,
            deriv=deriv_np,
            variance=variance_np,
            weights=weights_np,
            fit_intercept=self._fit_intercept,
        )

        cov_full = _invert_information_numpy(fisher)
        self._coef_cov = cov_full

        diag = np.clip(np.diag(cov_full), 1e-12, None)
        std_full = np.sqrt(diag)

        coef_np = _to_numpy(self.coef_)
        coef_full = _combine_coefficients_numpy(self.intercept_, coef_np, self._fit_intercept)

        z_scores = np.divide(coef_full, std_full, out=np.zeros_like(std_full), where=std_full > 0)
        p_full = 2.0 * (1.0 - _standard_normal_cdf_numpy(np.abs(z_scores)))

        if self._fit_intercept and self.intercept_ is not None:
            self._intercept_std_error = float(std_full[0])
            self._intercept_p_value = float(p_full[0])
            self._std_errors = std_full[1:]
            self._p_values = p_full[1:]
        else:
            self._intercept_std_error = None
            self._intercept_p_value = None
            self._std_errors = std_full
            self._p_values = p_full

    @property
    def intercept_std_error_(self) -> float | None:
        """Standard error of the intercept parameter, if fitted."""

        if self.intercept_ is None:
            return None
        if self._intercept_std_error is None:
            self._compute_inference()
        return self._intercept_std_error

    @property
    def intercept_p_value_(self) -> float | None:
        """Wald p-value for the intercept parameter, if fitted."""

        if self.intercept_ is None:
            return None
        if self._intercept_p_value is None:
            self._compute_inference()
        return self._intercept_p_value

    @property
    def diagnostics_(self) -> "GLMDiagnosticResult":
        """Return (and cache) residual and influence diagnostics for this fit."""

        if self._diagnostics_cache is None:
            from ...inference.diagnostics import glm_diagnostics

            self._diagnostics_cache = glm_diagnostics(self)
        return self._diagnostics_cache

    def predict(
        self,
        X_new: Array,
        *,
        type: str = "response",
        backend: str | None = None,
        interval: str | None = None,
        level: float = 0.95,
    ) -> Array | tuple[Array, Array, Array]:
        """Generate predictions for new design matrices.

        Returns the mean prediction when ``interval`` is ``None``; otherwise a
        tuple ``(mean, lower, upper)`` with Wald confidence bounds.
        """

        del backend  # Reserved for future multi-backend dispatching

        interval_kind = interval.lower() if interval is not None else None
        if interval_kind not in (None, "confidence"):
            raise NotImplementedError(f"Unsupported interval type: {interval!r}")
        if interval_kind is not None and not (0.0 < level < 1.0):
            raise ValueError("level must be in the open interval (0, 1)")

        xp = namespace(X_new, self._X, self._y)
        like = self._X if self._X is not None else self.mu_
        X_arr = as_namespace_array(X_new, xp, like=like)

        if xp is np:
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
        else:
            if getattr(X_arr, "ndim", 1) == 1:
                X_arr = X_arr.unsqueeze(0)

        coef_arr = as_namespace_array(self.coef_, xp, like=X_arr)
        if coef_arr.ndim == 1:
            coef_arr = coef_arr.reshape(-1, 1)

        eta = X_arr @ coef_arr
        eta = eta.squeeze(-1)

        if self.intercept_ is not None:
            eta = eta + self.intercept_

        if type == "link":
            predictions = eta
        elif type == "response":
            predictions = self.link.inverse(eta)
        else:
            raise ValueError(f"Unknown prediction type: {type!r}")

        if interval_kind is None:
            return predictions

        if self._coef_cov is None or self._std_errors is None or self._p_values is None:
            self._compute_inference()

        design_np = _design_with_intercept_numpy(X_arr, fit_intercept=self._fit_intercept)
        cov = np.asarray(self._coef_cov, dtype=np.float64)
        se_eta = _prediction_standard_errors(design_np, cov)

        quantile = NormalDist().inv_cdf(0.5 + level / 2.0)
        eta_np = _to_numpy(eta)
        lower_eta = eta_np - quantile * se_eta
        upper_eta = eta_np + quantile * se_eta

        if type == "link":
            lower_vals = as_namespace_array(lower_eta, xp, like=eta)
            upper_vals = as_namespace_array(upper_eta, xp, like=eta)
            return predictions, lower_vals, upper_vals

        # Delta method for response-scale intervals
        mu = predictions
        derivative = self.link.derivative(mu)
        deriv_np = np.clip(np.abs(_to_numpy(derivative)), 1e-12, None)
        se_mu = se_eta / deriv_np
        mu_np = _to_numpy(mu)
        lower_mu = mu_np - quantile * se_mu
        upper_mu = mu_np + quantile * se_mu

        lower_vals = as_namespace_array(lower_mu, xp, like=mu)
        upper_vals = as_namespace_array(upper_mu, xp, like=mu)
        return predictions, lower_vals, upper_vals

    def summary(self, *, detailed: bool = True) -> str:
        """
        Generate a formatted summary table of the GLM fit.

        Parameters
        ----------
        detailed : bool, default=True
            If True, includes full coefficient table with statistics. If False,
            returns a condensed summary with key model metrics only.

        Returns
        -------
        str
            Multi-line formatted string containing:

            - Model information (family, link, observations, parameters)
            - Convergence status and iterations
            - Coefficient table with estimates, std errors, z-values, p-values
            - Goodness-of-fit statistics (deviance, AIC, BIC, pseudo R²)

        Examples
        --------
        >>> import numpy as np
        >>> from aurora.models.glm import fit_glm
        >>> X = np.random.randn(100, 2)
        >>> y = np.random.poisson(np.exp(X[:, 0] * 0.5))
        >>> result = fit_glm(X, y, family='poisson', link='log')
        >>> print(result.summary())  # doctest: +SKIP
        """
        lines = []
        sep = "=" * 78
        lines.append(sep)
        lines.append("Generalized Linear Model Results".center(78))
        lines.append(sep)

        # Model information
        family_name = type(self.family).__name__.replace("Family", "")
        link_name = type(self.link).__name__.replace("Link", "")

        n_obs = int(self.mu_.shape[0]) if hasattr(self.mu_, "shape") else 0
        n_params = int(self.coef_.shape[0]) if hasattr(self.coef_, "shape") else 0
        if self._fit_intercept and self.intercept_ is not None:
            n_params += 1
        df_resid = max(n_obs - n_params, 0)
        df_model = n_params - (1 if self._fit_intercept else 0)

        # Two-column layout for header
        lines.append(f"{'Family:':<25} {family_name:<26} {'Link function:':<15} {link_name}")
        lines.append(f"{'No. Observations:':<25} {n_obs:<26} {'Df Residuals:':<15} {df_resid}")
        lines.append(f"{'Df Model:':<25} {df_model:<26}")

        converged_str = "Yes" if self.converged_ else "No"
        lines.append(f"{'Converged:':<25} {converged_str:<26} {'No. Iterations:':<15} {self.n_iter_}")
        lines.append(sep)

        if detailed:
            # Coefficient table
            lines.append(f"{'':>12} {'coef':>10} {'std err':>10} {'z':>10} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}")
            lines.append("-" * 78)

            # Trigger inference computation if needed
            if self._std_errors is None or self._p_values is None:
                try:
                    _ = self.std_errors_  # Triggers _compute_inference()
                except RuntimeError:
                    # If inference fails (e.g., missing design matrix), show coefficients only
                    coef_np = _to_numpy(self.coef_)
                    if self._fit_intercept and self.intercept_ is not None:
                        lines.append(f"{'intercept':>12} {self.intercept_:>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                    for i, coef_val in enumerate(coef_np):
                        lines.append(f"{'X' + str(i):>12} {float(coef_val):>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                    lines.append(sep)
                    lines.append("(Inference statistics unavailable: design matrix not stored)")
                    return "\n".join(lines)

            # Confidence interval quantile (95%)
            from statistics import NormalDist
            quantile = NormalDist().inv_cdf(0.975)

            # Add intercept row if present
            if self._fit_intercept and self.intercept_ is not None:
                intercept_se = self.intercept_std_error_ or 0.0
                intercept_pval = self.intercept_p_value_ or 1.0
                z_val = self.intercept_ / intercept_se if intercept_se > 0 else 0.0

                ci_lower = self.intercept_ - quantile * intercept_se
                ci_upper = self.intercept_ + quantile * intercept_se

                sig = _significance_stars(intercept_pval)
                lines.append(
                    f"{'intercept':>12} {self.intercept_:>10.4f} {intercept_se:>10.4f} "
                    f"{z_val:>10.3f} {intercept_pval:>10.3f} {ci_lower:>10.4f} {ci_upper:>10.4f} {sig}"
                )

            # Add coefficient rows
            coef_np = _to_numpy(self.coef_)
            std_errors_np = _to_numpy(self.std_errors_)
            p_values_np = _to_numpy(self.p_values_)

            for i in range(len(coef_np)):
                coef_val = float(coef_np[i])
                se_val = float(std_errors_np[i]) if i < len(std_errors_np) else 0.0
                p_val = float(p_values_np[i]) if i < len(p_values_np) else 1.0
                z_val = coef_val / se_val if se_val > 0 else 0.0

                # Confidence interval (95%)
                ci_lower = coef_val - quantile * se_val
                ci_upper = coef_val + quantile * se_val

                sig = _significance_stars(p_val)
                var_name = f"X{i}"
                lines.append(
                    f"{var_name:>12} {coef_val:>10.4f} {se_val:>10.4f} "
                    f"{z_val:>10.3f} {p_val:>10.3f} {ci_lower:>10.4f} {ci_upper:>10.4f} {sig}"
                )

            lines.append(sep)
            lines.append("Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
            lines.append(sep)

        # Goodness of fit
        pseudo_r2 = 1.0 - (self.deviance_ / self.null_deviance_) if self.null_deviance_ > 0 else 0.0

        lines.append(f"{'Deviance:':<30} {self.deviance_:>15.2f} {'Null Deviance:':<20} {self.null_deviance_:>10.2f}")
        lines.append(f"{'AIC:':<30} {self.aic_:>15.2f} {'BIC:':<20} {self.bic_:>10.2f}")
        lines.append(f"{'Pseudo R-squared:':<30} {pseudo_r2:>15.4f}")
        lines.append(sep)

        return "\n".join(lines)

    def plot_diagnostics(self, *, figsize: tuple[float, float] = (12, 10)) -> Any:
        """
        Generate standard diagnostic plots for the fitted GLM.

        Creates a 2x2 grid of diagnostic plots:
        1. Residuals vs Fitted: response residuals against fitted values
        2. Q-Q Plot: theoretical normal quantiles vs studentized residuals
        3. Scale-Location: sqrt(|studentized residuals|) vs fitted values
        4. Residuals vs Leverage: studentized residuals vs leverage, with Cook's distance contours

        Parameters
        ----------
        figsize : tuple of float, default=(12, 10)
            Figure size in inches (width, height).

        Returns
        -------
        matplotlib.figure.Figure
            The created figure object containing the 4 diagnostic plots.

        Raises
        ------
        ImportError
            If matplotlib is not installed.

        Examples
        --------
        >>> import numpy as np
        >>> from aurora.models.glm import fit_glm
        >>> X = np.random.randn(100, 2)
        >>> y = np.random.poisson(np.exp(X[:, 0] * 0.5))
        >>> result = fit_glm(X, y, family='poisson')
        >>> fig = result.plot_diagnostics()  # doctest: +SKIP
        >>> fig.savefig('diagnostics.png')  # doctest: +SKIP
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_diagnostics(). "
                "Install it with: pip install matplotlib"
            ) from exc

        # Get diagnostics (uses cached version if available)
        diag = self.diagnostics_

        # Get fitted values
        mu = _to_numpy(self.mu_)

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("GLM Diagnostic Plots", fontsize=14, fontweight="bold")

        # Plot 1: Residuals vs Fitted
        ax1 = axes[0, 0]
        ax1.scatter(mu, diag.response_residuals, alpha=0.6, s=20, edgecolors="k", linewidths=0.5)
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")
        ax1.grid(True, alpha=0.3)

        # Add lowess smooth line if scipy available
        try:
            from scipy.signal import savgol_filter

            # Sort by fitted values for smooth line
            sorted_idx = np.argsort(mu)
            mu_sorted = mu[sorted_idx]
            resid_sorted = diag.response_residuals[sorted_idx]

            # Apply Savitzky-Golay filter for smooth trend
            window = min(51, len(mu) // 3)
            if window % 2 == 0:
                window += 1  # Must be odd
            if window >= 5:
                smooth = savgol_filter(resid_sorted, window_length=window, polyorder=3)
                ax1.plot(mu_sorted, smooth, color="blue", linewidth=2, alpha=0.8)
        except (ImportError, ValueError):
            pass  # Skip smooth line if scipy unavailable or data too small

        # Plot 2: Q-Q Plot
        ax2 = axes[0, 1]
        studentized = diag.studentized_residuals
        studentized_sorted = np.sort(studentized)
        n = len(studentized)
        theoretical_quantiles = np.array([NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)])

        ax2.scatter(theoretical_quantiles, studentized_sorted, alpha=0.6, s=20, edgecolors="k", linewidths=0.5)
        # Add reference line
        min_val = min(theoretical_quantiles.min(), studentized_sorted.min())
        max_val = max(theoretical_quantiles.max(), studentized_sorted.max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.7)
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Studentized Residuals")
        ax2.set_title("Normal Q-Q Plot")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scale-Location
        ax3 = axes[1, 0]
        sqrt_abs_studentized = np.sqrt(np.abs(studentized))
        ax3.scatter(mu, sqrt_abs_studentized, alpha=0.6, s=20, edgecolors="k", linewidths=0.5)
        ax3.set_xlabel("Fitted values")
        ax3.set_ylabel(r"$\sqrt{|Studentized\ Residuals|}$")
        ax3.set_title("Scale-Location")
        ax3.grid(True, alpha=0.3)

        # Add smooth trend line
        try:
            from scipy.signal import savgol_filter

            sorted_idx = np.argsort(mu)
            mu_sorted = mu[sorted_idx]
            sqrt_resid_sorted = sqrt_abs_studentized[sorted_idx]

            window = min(51, len(mu) // 3)
            if window % 2 == 0:
                window += 1
            if window >= 5:
                smooth = savgol_filter(sqrt_resid_sorted, window_length=window, polyorder=3)
                ax3.plot(mu_sorted, smooth, color="red", linewidth=2, alpha=0.8)
        except (ImportError, ValueError):
            pass

        # Plot 4: Residuals vs Leverage
        ax4 = axes[1, 1]
        leverage = diag.leverage
        cooks_d = diag.cooks_distance

        # Color points by Cook's distance
        scatter = ax4.scatter(
            leverage,
            studentized,
            c=cooks_d,
            cmap="YlOrRd",
            alpha=0.6,
            s=20,
            edgecolors="k",
            linewidths=0.5,
        )
        ax4.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax4.set_xlabel("Leverage")
        ax4.set_ylabel("Studentized Residuals")
        ax4.set_title("Residuals vs Leverage")
        ax4.grid(True, alpha=0.3)

        # Add colorbar for Cook's distance
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("Cook's Distance", rotation=270, labelpad=15)

        # Highlight high leverage or influential points
        high_cooks = cooks_d > 4.0 / len(mu)  # Common threshold
        if np.any(high_cooks):
            ax4.scatter(
                leverage[high_cooks],
                studentized[high_cooks],
                s=100,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                label="High Cook's D",
            )
            ax4.legend()

        plt.tight_layout()
        return fig


def _to_numpy(value: Any | None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _fisher_information_numpy(
    *,
    X: np.ndarray,
    deriv: np.ndarray,
    variance: np.ndarray,
    weights: np.ndarray | None,
    fit_intercept: bool,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    deriv = np.asarray(deriv, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)

    denom = np.clip(deriv * deriv * variance, 1e-12, None)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64) / denom
    else:
        w = 1.0 / denom
    w = np.clip(w, 1e-12, None)

    if fit_intercept:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_design = np.concatenate((ones, X), axis=1)
    else:
        X_design = X

    return _weighted_gram_numpy(X_design, w)


def _invert_information_numpy(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    eye = np.eye(n, dtype=matrix.dtype)
    jitter = 1e-8
    for _ in range(6):
        try:
            return np.linalg.solve(matrix + jitter * eye, eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.pinv(matrix + jitter * eye)


def _combine_coefficients_numpy(
    intercept: float | None,
    coef: np.ndarray,
    fit_intercept: bool,
) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float64)
    if fit_intercept and intercept is not None:
        return np.concatenate(([float(intercept)], coef))
    return coef


def _standard_normal_cdf_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def _weighted_gram_numpy(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_samples, n_features = X.shape
    weights = np.asarray(weights, dtype=np.float64)
    gram = np.zeros((n_features, n_features), dtype=np.float64)
    for idx in range(n_samples):
        xi = X[idx]
        wi = weights[idx]
        for j in range(n_features):
            gram[j, j] += wi * xi[j] * xi[j]
            for k in range(j + 1, n_features):
                val = wi * xi[j] * xi[k]
                gram[j, k] += val
                gram[k, j] += val
    return gram


def _design_with_intercept_numpy(X: Array, *, fit_intercept: bool) -> np.ndarray:
    matrix = _to_numpy(X)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if not fit_intercept:
        return matrix
    ones = np.ones((matrix.shape[0], 1), dtype=matrix.dtype)
    return np.concatenate((ones, matrix), axis=1)


def _prediction_standard_errors(design: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape[0] != design.shape[1]:
        raise ValueError("Covariance matrix and design matrix dimensions are incompatible")
    projection = design @ covariance
    variances = np.einsum("ij,ij->i", projection, design)
    variances = np.clip(variances, 1e-12, None)
    return np.sqrt(variances)


def _significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value thresholds."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""


__all__.append("GLMResult")
