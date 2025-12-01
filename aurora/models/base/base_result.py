"""Base result classes for Aurora-GLM models.

This module defines the unified inheritance hierarchy for all model results:

    BaseResult (ABC)
    ├── LinearModelResult
    │   └── GLMResult  
    │       └── GAMResult (planned)
    └── MixedModelResultBase
        └── GAMMResult, PQLResult, LaplaceResult

Mathematical Framework
----------------------
All results share a common structure based on the general linear predictor:

.. math::
    g(\\mu) = \\eta = X\\beta + \\text{(smooth terms)} + \\text{(random effects)}

where:
- g(·) is the link function
- μ = E[Y] is the conditional mean
- η is the linear predictor
- β are the fixed effect coefficients

References
----------
.. [1] McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models.
.. [2] Wood, S. N. (2017). Generalized Additive Models: An Introduction with R.
.. [3] Pinheiro, J. C., & Bates, D. M. (2000). Mixed-Effects Models in S and S-PLUS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ...distributions.base import Family, LinkFunction


# =============================================================================
# Protocols for duck typing
# =============================================================================

@runtime_checkable
class ResultProtocol(Protocol):
    """Protocol defining the minimal interface for all results."""
    
    @property
    def converged(self) -> bool:
        """Whether the fitting algorithm converged."""
        ...
    
    @property
    def fitted_values(self) -> np.ndarray:
        """Fitted values on the response scale."""
        ...
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        ...
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions for new data."""
        ...


@runtime_checkable  
class MixedModelProtocol(Protocol):
    """Protocol for mixed model results with random effects."""
    
    @property
    def fixed_effects(self) -> np.ndarray:
        """Fixed effect coefficients."""
        ...
    
    @property
    def random_effects(self) -> np.ndarray:
        """Random effect coefficients (BLUPs)."""
        ...
    
    @property
    def variance_components(self) -> dict[str, float]:
        """Variance components for random effects."""
        ...


# =============================================================================
# Base Result Class (not a dataclass to allow flexible inheritance)
# =============================================================================

class BaseResult(ABC):
    """Abstract base class for all Aurora-GLM model results.
    
    This class defines the common interface that all fitted models must implement.
    It provides a consistent API for accessing model coefficients, fitted values,
    predictions, and summaries.
    
    Attributes
    ----------
    converged_ : bool
        Whether the fitting algorithm converged successfully.
    n_iter_ : int
        Number of iterations performed during fitting.
    n_obs_ : int
        Number of observations used in fitting.
    
    Notes
    -----
    Subclasses must implement:
    - `coefficients` property
    - `fitted_values` property  
    - `residuals` property
    - `predict()` method
    - `summary()` method
    """
    
    def __init__(
        self,
        converged: bool,
        n_iter: int,
        n_obs: int,
        **kwargs
    ):
        self.converged_ = converged
        self.n_iter_ = n_iter
        self.n_obs_ = n_obs
        self._diagnostics: dict[str, Any] = {}
    
    @property
    @abstractmethod
    def coefficients(self) -> np.ndarray:
        """All model coefficients as a single array."""
        ...
    
    @property
    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """Fitted values μ̂ on the response scale."""
        ...
    
    @property
    @abstractmethod
    def residuals(self) -> np.ndarray:
        """Response residuals (y - μ̂)."""
        ...
    
    @property
    def converged(self) -> bool:
        """Whether fitting converged (alias for converged_)."""
        return self.converged_
    
    @property
    def n_observations(self) -> int:
        """Number of observations (alias for n_obs_)."""
        return self.n_obs_
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions for new data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict on.
        **kwargs : dict
            Additional prediction options (type, interval, etc.)
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Model predictions.
        """
        ...
    
    @abstractmethod
    def summary(self) -> str:
        """Return a formatted summary of the fitted model.
        
        Returns
        -------
        summary : str
            Multi-line formatted string with model information,
            coefficient table, and goodness-of-fit statistics.
        """
        ...
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to a dictionary representation.
        
        Returns
        -------
        result_dict : dict
            Dictionary containing all model components.
        """
        return {
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "n_obs": self.n_obs_,
            "coefficients": self.coefficients,
            "fitted_values": self.fitted_values,
            "residuals": self.residuals,
        }
    
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"n_obs={self.n_obs_}, "
            f"converged={self.converged_}, "
            f"n_iter={self.n_iter_})"
        )


# =============================================================================
# Linear Model Result
# =============================================================================

class LinearModelResult(BaseResult):
    """Base class for linear model results (OLS, GLM).
    
    Extends BaseResult with coefficient-specific attributes including
    intercept, standard errors, and inference statistics.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (excluding intercept).
    intercept_ : float or None
        Estimated intercept term, if fitted.
    residual_variance_ : float
        Estimated residual variance σ².
    
    Properties
    ----------
    std_errors_ : ndarray
        Standard errors of coefficients.
    p_values_ : ndarray
        Two-sided p-values from Wald tests.
    coef_cov_ : ndarray
        Covariance matrix of coefficient estimates.
    """
    
    def __init__(
        self,
        coef: np.ndarray,
        intercept: float | None,
        fitted_values: np.ndarray,
        residuals: np.ndarray,
        residual_variance: float,
        converged: bool,
        n_iter: int,
        n_obs: int,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs
    ):
        super().__init__(converged=converged, n_iter=n_iter, n_obs=n_obs)
        self.coef_ = coef
        self.intercept_ = intercept
        self.residual_variance_ = residual_variance
        self._fitted_values = fitted_values
        self._residuals = residuals
        self._X = X
        self._y = y
        
        # Lazy computation storage
        self._coef_cov: np.ndarray | None = None
        self._std_errors: np.ndarray | None = None
        self._p_values: np.ndarray | None = None
    
    @property
    def coefficients(self) -> np.ndarray:
        """All coefficients including intercept if present."""
        if self.intercept_ is not None:
            return np.concatenate([[self.intercept_], self.coef_])
        return self.coef_
    
    @property
    def fitted_values(self) -> np.ndarray:
        """Fitted values μ̂."""
        return self._fitted_values
    
    @property
    def residuals(self) -> np.ndarray:
        """Response residuals (y - μ̂)."""
        return self._residuals
    
    @property
    def n_features(self) -> int:
        """Number of features (excluding intercept)."""
        return len(self.coef_)
    
    @property
    def df_model(self) -> int:
        """Model degrees of freedom (number of parameters - 1)."""
        n_params = len(self.coef_)
        if self.intercept_ is not None:
            n_params += 1
        return n_params - 1
    
    @property
    def df_residual(self) -> int:
        """Residual degrees of freedom."""
        n_params = len(self.coef_)
        if self.intercept_ is not None:
            n_params += 1
        return self.n_obs_ - n_params
    
    @property
    def r_squared(self) -> float:
        """Coefficient of determination R².
        
        .. math::
            R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}
        """
        if self._y is None:
            return np.nan
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self._y - np.mean(self._y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot
    
    @property
    def adj_r_squared(self) -> float:
        """Adjusted R² accounting for number of predictors.
        
        .. math::
            R^2_{adj} = 1 - (1 - R^2) \\frac{n - 1}{n - p - 1}
        """
        r2 = self.r_squared
        n = self.n_obs_
        p = self.n_features
        if n - p - 1 <= 0:
            return np.nan
        return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions for new data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New design matrix.
        
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Linear predictions Xβ + intercept.
        """
        X = np.atleast_2d(X)
        eta = X @ self.coef_
        if self.intercept_ is not None:
            eta = eta + self.intercept_
        return eta
    
    def summary(self) -> str:
        """Return formatted summary table."""
        lines = []
        sep = "=" * 70
        
        lines.append(sep)
        lines.append(f"{'Linear Model Results':^70}")
        lines.append(sep)
        lines.append(f"No. Observations: {self.n_obs_:>10}    R-squared:     {self.r_squared:>10.4f}")
        lines.append(f"Df Model:         {self.df_model:>10}    Adj. R-squared:{self.adj_r_squared:>10.4f}")
        lines.append(f"Df Residuals:     {self.df_residual:>10}    Residual Var:  {self.residual_variance_:>10.4f}")
        lines.append(f"Converged:        {'Yes' if self.converged_ else 'No':>10}    Iterations:    {self.n_iter_:>10}")
        lines.append(sep)
        
        # Coefficient table
        lines.append(f"{'':>12} {'coef':>12} {'std err':>12}")
        lines.append("-" * 70)
        
        if self.intercept_ is not None:
            lines.append(f"{'intercept':>12} {self.intercept_:>12.4f}")
        
        for i, coef in enumerate(self.coef_):
            lines.append(f"{'X' + str(i):>12} {coef:>12.4f}")
        
        lines.append(sep)
        return "\n".join(lines)


# =============================================================================
# Mixed Model Result Base
# =============================================================================

class MixedModelResultBase(BaseResult):
    """Base class for mixed model results (GAMM, PQL, Laplace).
    
    Provides common structure for models with both fixed and random effects.
    
    Attributes
    ----------
    fixed_effects_ : ndarray
        Fixed effect coefficients β.
    random_effects_ : ndarray
        Random effect coefficients b (BLUPs).
    variance_components_ : dict
        Variance components for random effects.
    log_likelihood_ : float
        (Marginal) log-likelihood.
        
    Mathematical Framework
    ----------------------
    Mixed models have the general form:
    
    .. math::
        y = X\\beta + Zb + \\epsilon
    
    where:
    - X is the fixed effects design matrix
    - β are the fixed effects
    - Z is the random effects design matrix  
    - b ~ N(0, Ψ) are the random effects
    - ε ~ N(0, σ²I) is the error term
    
    The marginal distribution is:
    
    .. math::
        y \\sim N(X\\beta, ZΨZ' + σ²I)
    """
    
    def __init__(
        self,
        fixed_effects: np.ndarray,
        random_effects: np.ndarray,
        variance_components: dict[str, Any],
        log_likelihood: float,
        fitted_values: np.ndarray,
        linear_predictor: np.ndarray,
        converged: bool,
        n_iter: int,
        n_obs: int,
        X: np.ndarray | None = None,
        Z: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs
    ):
        super().__init__(converged=converged, n_iter=n_iter, n_obs=n_obs)
        self.fixed_effects_ = fixed_effects
        self.random_effects_ = random_effects
        self.variance_components_ = variance_components
        self.log_likelihood_ = log_likelihood
        self._fitted_values = fitted_values
        self._linear_predictor = linear_predictor
        self._X = X
        self._Z = Z
        self._y = y
    
    @property
    def coefficients(self) -> np.ndarray:
        """Combined coefficients [β, b]."""
        return np.concatenate([self.fixed_effects_, self.random_effects_])
    
    @property
    def beta(self) -> np.ndarray:
        """Fixed effects (alias for fixed_effects_)."""
        return self.fixed_effects_
    
    @property
    def b(self) -> np.ndarray:
        """Random effects BLUPs (alias for random_effects_)."""
        return self.random_effects_
    
    @property
    def fitted_values(self) -> np.ndarray:
        """Fitted values on response scale."""
        return self._fitted_values
    
    @property
    def linear_predictor(self) -> np.ndarray:
        """Linear predictor η = Xβ + Zb."""
        return self._linear_predictor
    
    @property
    def residuals(self) -> np.ndarray:
        """Response residuals."""
        if self._y is None:
            raise ValueError("Response y not stored; cannot compute residuals")
        return self._y - self._fitted_values
    
    @property
    def n_random_effects(self) -> int:
        """Number of random effect coefficients."""
        return len(self.random_effects_)
    
    @property
    def n_fixed_effects(self) -> int:
        """Number of fixed effect coefficients."""
        return len(self.fixed_effects_)
    
    def predict(
        self, 
        X: np.ndarray, 
        Z: np.ndarray | None = None,
        include_random: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Generate predictions for new data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_fixed)
            Fixed effects design matrix.
        Z : ndarray of shape (n_samples, n_random), optional
            Random effects design matrix. Required if include_random=True.
        include_random : bool, default=True
            Whether to include random effects in predictions.
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Model predictions.
        """
        X = np.atleast_2d(X)
        eta = X @ self.fixed_effects_
        
        if include_random and Z is not None:
            Z = np.atleast_2d(Z)
            eta = eta + Z @ self.random_effects_
        
        return eta
    
    def summary(self) -> str:
        """Return formatted summary table."""
        lines = []
        sep = "=" * 70
        
        lines.append(sep)
        lines.append(f"{'Mixed Model Results':^70}")
        lines.append(sep)
        lines.append(f"No. Observations:   {self.n_obs_:>10}")
        lines.append(f"No. Fixed Effects:  {self.n_fixed_effects:>10}")
        lines.append(f"No. Random Effects: {self.n_random_effects:>10}")
        lines.append(f"Log-Likelihood:     {self.log_likelihood_:>10.4f}")
        lines.append(f"Converged:          {'Yes' if self.converged_ else 'No':>10}")
        lines.append(sep)
        
        # Fixed effects
        lines.append("Fixed Effects:")
        lines.append("-" * 70)
        for i, coef in enumerate(self.fixed_effects_):
            lines.append(f"  β{i}: {coef:>12.4f}")
        
        lines.append("")
        lines.append("Variance Components:")
        lines.append("-" * 70)
        for name, value in self.variance_components_.items():
            if isinstance(value, np.ndarray):
                lines.append(f"  {name}: {value.shape} matrix")
            else:
                lines.append(f"  {name}: {value:>12.4f}")
        
        lines.append(sep)
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "fixed_effects": self.fixed_effects_,
            "random_effects": self.random_effects_,
            "variance_components": self.variance_components_,
            "log_likelihood": self.log_likelihood_,
            "linear_predictor": self.linear_predictor,
        })
        return base_dict


__all__ = [
    "ResultProtocol",
    "MixedModelProtocol", 
    "BaseResult",
    "LinearModelResult",
    "MixedModelResultBase",
]
