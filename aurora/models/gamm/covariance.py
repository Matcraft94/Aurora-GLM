"""Covariance structures for random effects.

This module provides classes for different covariance parameterizations
used in random effects modeling.

References
----------
.. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CovarianceStructure(ABC):
    """Base class for random effect covariance structures.

    A covariance structure defines how to parameterize the variance-covariance
    matrix Ψ for random effects. Different structures impose different
    constraints (e.g., diagonal, identity) to reduce the number of parameters.

    Methods
    -------
    n_parameters(n_effects)
        Number of free parameters for given number of random effects
    construct_psi(params, n_effects)
        Build covariance matrix from parameter vector
    extract_params(psi)
        Extract parameter vector from covariance matrix
    """

    @abstractmethod
    def n_parameters(self, n_effects: int) -> int:
        """Number of free parameters.

        Parameters
        ----------
        n_effects : int
            Number of random effects (dimension of Ψ)

        Returns
        -------
        int
            Number of free parameters to specify Ψ
        """
        pass

    @abstractmethod
    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct covariance matrix from parameters.

        Parameters
        ----------
        params : ndarray
            Parameter vector (length = n_parameters(n_effects))
        n_effects : int
            Number of random effects

        Returns
        -------
        psi : ndarray, shape (n_effects, n_effects)
            Variance-covariance matrix (symmetric, positive definite)
        """
        pass

    @abstractmethod
    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from covariance matrix.

        Parameters
        ----------
        psi : ndarray, shape (q, q)
            Variance-covariance matrix

        Returns
        -------
        params : ndarray
            Parameter vector
        """
        pass


class UnstructuredCovariance(CovarianceStructure):
    """Unstructured covariance (full matrix).

    Allows all variances and covariances to vary freely.
    For q random effects, requires q(q+1)/2 parameters.

    Parameterization uses Cholesky decomposition: Ψ = LL'
    This ensures Ψ is positive definite.

    Parameters
    ----------
    None

    Examples
    --------
    >>> cov = UnstructuredCovariance()
    >>> cov.n_parameters(2)  # 2 random effects
    3

    >>> # Parameters: [L11, L21, L22]
    >>> params = np.array([1.0, 0.5, 0.8])
    >>> psi = cov.construct_psi(params, n_effects=2)
    >>> psi
    array([[1.  , 0.5 ],
           [0.5 , 0.89]])

    >>> # Extract back
    >>> extracted = cov.extract_params(psi)
    >>> np.allclose(extracted, params)
    True

    Notes
    -----
    The Cholesky parameterization:
    - Ensures positive definiteness
    - Provides unconstrained optimization (except L_ii > 0)
    - Standard approach in lme4 and other mixed model software
    """

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = q(q+1)/2."""
        return n_effects * (n_effects + 1) // 2

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct Ψ = LL' from Cholesky parameters.

        Parameters are ordered as lower triangle of L:
        [L11, L21, L22, L31, L32, L33, ...]
        """
        expected_n = self.n_parameters(n_effects)
        if len(params) != expected_n:
            raise ValueError(
                f"Expected {expected_n} parameters for {n_effects} effects, "
                f"got {len(params)}"
            )

        # Build lower triangular matrix L
        L = np.zeros((n_effects, n_effects))
        idx = 0
        for i in range(n_effects):
            for j in range(i + 1):
                L[i, j] = params[idx]
                idx += 1

        # Ψ = LL'
        psi = L @ L.T
        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract Cholesky parameters from Ψ.

        Computes L such that Ψ = LL', then extracts lower triangle.
        """
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(psi)
        except np.linalg.LinAlgError:
            raise ValueError("psi must be positive definite")

        # Extract lower triangle
        n = psi.shape[0]
        params = []
        for i in range(n):
            for j in range(i + 1):
                params.append(L[i, j])

        return np.array(params)


class DiagonalCovariance(CovarianceStructure):
    """Diagonal covariance (independent random effects).

    Assumes random effects are independent (zero correlations).
    For q random effects, requires q parameters (variances only).

    Parameters are log-transformed variances to ensure positivity:
    σ²_i = exp(params[i])

    Examples
    --------
    >>> cov = DiagonalCovariance()
    >>> cov.n_parameters(3)
    3

    >>> # Parameters: log-variances
    >>> params = np.array([0.0, 0.5, 1.0])  # log(σ²)
    >>> psi = cov.construct_psi(params, n_effects=3)
    >>> psi
    array([[1.        , 0.        , 0.        ],
           [0.        , 1.64872127, 0.        ],
           [0.        , 0.        , 2.71828183]])

    >>> np.diag(psi)  # Variances
    array([1.        , 1.64872127, 2.71828183])

    Notes
    -----
    Log transformation:
    - Ensures variances are positive
    - Provides unconstrained optimization
    - Standard in many mixed model packages
    """

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = q (one variance per effect)."""
        return n_effects

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct diagonal Ψ from log-variances."""
        if len(params) != n_effects:
            raise ValueError(
                f"Expected {n_effects} parameters, got {len(params)}"
            )

        # Transform from log scale to ensure positivity
        variances = np.exp(params)
        psi = np.diag(variances)
        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract log-variances from diagonal Ψ."""
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        # Check if truly diagonal
        off_diag = psi - np.diag(np.diag(psi))
        if not np.allclose(off_diag, 0):
            raise ValueError("psi must be diagonal for DiagonalCovariance")

        # Extract variances and log-transform
        variances = np.diag(psi)
        if np.any(variances <= 0):
            raise ValueError("Diagonal elements must be positive")

        params = np.log(variances)
        return params


class IdentityCovariance(CovarianceStructure):
    """Identity covariance (equal variances, no correlation).

    All random effects have the same variance and are uncorrelated.
    For q random effects, requires only 1 parameter: σ²

    Ψ = σ² I_q

    Parameter is log(σ²) to ensure positivity.

    Examples
    --------
    >>> cov = IdentityCovariance()
    >>> cov.n_parameters(3)
    1

    >>> # Single parameter: log(σ²)
    >>> params = np.array([0.5])  # log(σ²)
    >>> psi = cov.construct_psi(params, n_effects=3)
    >>> psi
    array([[1.64872127, 0.        , 0.        ],
           [0.        , 1.64872127, 0.        ],
           [0.        , 0.        , 1.64872127]])

    >>> # All variances are equal
    >>> np.allclose(np.diag(psi), np.exp(0.5))
    True

    Notes
    -----
    This is the most constrained structure, useful when:
    - Sample size is small
    - Random effects are believed to be exchangeable
    - Computational efficiency is important
    """

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = 1 (single variance)."""
        return 1

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct σ²I from log-variance."""
        if len(params) != 1:
            raise ValueError(
                f"Expected 1 parameter, got {len(params)}"
            )

        # Transform from log scale
        variance = np.exp(params[0])
        psi = variance * np.eye(n_effects)
        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract log(σ²) from Ψ = σ²I."""
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        # Check if truly σ²I
        n = psi.shape[0]
        expected = psi[0, 0] * np.eye(n)
        if not np.allclose(psi, expected):
            raise ValueError("psi must be proportional to identity for IdentityCovariance")

        # Extract variance and log-transform
        variance = psi[0, 0]
        if variance <= 0:
            raise ValueError("Variance must be positive")

        params = np.array([np.log(variance)])
        return params


class AR1Covariance(CovarianceStructure):
    """Autoregressive order 1 (AR(1)) covariance structure.

    Models temporal correlation where observations closer in time are more
    correlated: Cov(b_t, b_s) = σ² ρ^|t-s|

    For q random effects (time points), requires 2 parameters:
    - params[0]: log(σ²) - log-variance
    - params[1]: arctanh(ρ) - transformed correlation (ensures ρ ∈ (-1, 1))

    References
    ----------
    .. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS. Ch. 5.
    .. [2] Diggle et al. (2002). Analysis of Longitudinal Data, 2nd ed.

    Examples
    --------
    >>> cov = AR1Covariance()
    >>> cov.n_parameters(4)
    2

    >>> # Parameters: [log(σ²), arctanh(ρ)]
    >>> params = np.array([0.0, 0.5])  # σ²=1, ρ≈0.46
    >>> psi = cov.construct_psi(params, n_effects=4)
    >>> # Creates correlation matrix with AR(1) structure

    Notes
    -----
    The AR(1) structure is ideal for:
    - Equally-spaced longitudinal data
    - Time series with decaying correlations
    - Repeated measures where adjacent observations are most correlated

    The inverse of AR(1) covariance matrix is tridiagonal, enabling
    efficient computation for large matrices.
    """

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = 2 (variance and correlation)."""
        return 2

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct AR(1) covariance matrix.

        Parameters
        ----------
        params : ndarray
            [log(σ²), arctanh(ρ)] - transformed variance and correlation
        n_effects : int
            Number of time points/random effects

        Returns
        -------
        psi : ndarray (n_effects, n_effects)
            AR(1) covariance matrix
        """
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(params)}")

        # Transform parameters
        sigma2 = np.exp(params[0])
        rho = np.tanh(params[1])  # Maps R to (-1, 1)

        # Build AR(1) structure: Cov(i,j) = σ² ρ^|i-j|
        i, j = np.ogrid[:n_effects, :n_effects]
        psi = sigma2 * (rho ** np.abs(i - j))

        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from AR(1) covariance matrix.

        Estimates σ² from diagonal and ρ from first off-diagonal.
        """
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        n = psi.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 effects to estimate AR(1)")

        # Estimate variance from diagonal (should all be equal to σ²)
        sigma2 = np.mean(np.diag(psi))

        # Estimate ρ from first off-diagonal
        if n > 1:
            off_diag = np.mean([psi[i, i+1] for i in range(n-1)])
            rho = off_diag / sigma2
            rho = np.clip(rho, -0.999, 0.999)  # Ensure valid range
        else:
            rho = 0.0

        params = np.array([np.log(sigma2), np.arctanh(rho)])
        return params

    def inverse(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Efficient tridiagonal inverse for AR(1) structure.

        The inverse of an AR(1) matrix is tridiagonal, which allows
        for O(n) computation instead of O(n³).
        """
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(params)}")

        sigma2 = np.exp(params[0])
        rho = np.tanh(params[1])

        n = n_effects
        if n == 1:
            return np.array([[1.0 / sigma2]])

        # Tridiagonal inverse structure
        # Psi^{-1}[i,i] = (1 + rho²) / (σ²(1-rho²)) for interior
        # Psi^{-1}[i,i] = 1 / (σ²(1-rho²)) for boundaries
        # Psi^{-1}[i,i±1] = -rho / (σ²(1-rho²))

        denom = sigma2 * (1 - rho**2)
        psi_inv = np.zeros((n, n))

        # Main diagonal
        for i in range(n):
            if i == 0 or i == n - 1:
                psi_inv[i, i] = 1.0 / denom
            else:
                psi_inv[i, i] = (1 + rho**2) / denom

        # Off-diagonals
        for i in range(n - 1):
            psi_inv[i, i+1] = -rho / denom
            psi_inv[i+1, i] = -rho / denom

        return psi_inv


class CompoundSymmetryCovariance(CovarianceStructure):
    """Compound symmetry (exchangeable) covariance structure.

    Models constant correlation between all pairs of observations:
    - Diagonal: Var(b_i) = σ²
    - Off-diagonal: Cov(b_i, b_j) = σ²ρ for i ≠ j

    For q random effects, requires 2 parameters:
    - params[0]: log(σ²) - log-variance
    - params[1]: logit(ρ_scaled) - transformed correlation

    References
    ----------
    .. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS.
    .. [2] Diggle et al. (2002). Analysis of Longitudinal Data, 2nd ed.

    Examples
    --------
    >>> cov = CompoundSymmetryCovariance()
    >>> cov.n_parameters(5)
    2

    >>> params = np.array([0.0, 0.0])  # σ²=1, ρ=0.5
    >>> psi = cov.construct_psi(params, n_effects=3)
    >>> # Creates matrix with equal variances and equal correlations

    Notes
    -----
    Compound symmetry is equivalent to:
    - Random intercept model with iid residuals
    - Exchangeable correlation (ICC model)

    Constraint: ρ > -1/(q-1) for positive definiteness, where q is
    the number of effects.
    """

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = 2 (variance and correlation)."""
        return 2

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct compound symmetry covariance matrix.

        Parameters
        ----------
        params : ndarray
            [log(σ²), logit((ρ + 1/(q-1))/(1 + 1/(q-1)))]
        n_effects : int
            Number of random effects

        Returns
        -------
        psi : ndarray (n_effects, n_effects)
            Compound symmetry covariance matrix
        """
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(params)}")

        sigma2 = np.exp(params[0])

        # Transform to get ρ in valid range (-1/(q-1), 1)
        # Using shifted logit transformation
        q = n_effects
        rho_min = -1.0 / (q - 1) if q > 1 else -0.99
        rho_range = 1.0 - rho_min

        rho_scaled = 1.0 / (1.0 + np.exp(-params[1]))  # Sigmoid to (0, 1)
        rho = rho_min + rho_range * rho_scaled

        # Build compound symmetry structure
        psi = sigma2 * (rho * np.ones((n_effects, n_effects)) +
                        (1 - rho) * np.eye(n_effects))

        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from compound symmetry matrix."""
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        n = psi.shape[0]

        # Estimate σ² from diagonal
        sigma2 = np.mean(np.diag(psi))

        # Estimate ρ from off-diagonals
        if n > 1:
            off_diag_sum = np.sum(psi) - np.trace(psi)
            n_off = n * (n - 1)
            avg_cov = off_diag_sum / n_off
            rho = avg_cov / sigma2
            rho = np.clip(rho, -1/(n-1) + 0.001, 0.999)
        else:
            rho = 0.0

        # Inverse transform
        rho_min = -1.0 / (n - 1) if n > 1 else -0.99
        rho_range = 1.0 - rho_min
        rho_scaled = (rho - rho_min) / rho_range
        rho_scaled = np.clip(rho_scaled, 0.001, 0.999)

        params = np.array([np.log(sigma2), np.log(rho_scaled / (1 - rho_scaled))])
        return params


class ExponentialSpatialCovariance(CovarianceStructure):
    """Exponential spatial covariance structure.

    Models spatial correlation that decays exponentially with distance:
    Cov(b_i, b_j) = σ² exp(-d_ij / φ)

    where d_ij is the Euclidean distance between locations i and j,
    and φ is the range parameter.

    For geostatistical data, requires coordinates at construction time.
    Parameters:
    - params[0]: log(σ²) - log-variance (sill)
    - params[1]: log(φ) - log-range parameter

    References
    ----------
    .. [1] Diggle & Ribeiro (2007). Model-based Geostatistics.
    .. [2] Cressie (1993). Statistics for Spatial Data.
    .. [3] Rue & Held (2005). Gaussian Markov Random Fields.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> cov = ExponentialSpatialCovariance(coordinates=coords)
    >>> params = np.array([0.0, 0.5])  # σ²=1, φ≈1.65
    >>> psi = cov.construct_psi(params, n_effects=4)

    Notes
    -----
    The exponential covariance is a member of the Matérn family with
    smoothness ν = 0.5. It produces relatively rough spatial surfaces.

    The range parameter φ approximately equals the distance at which
    the correlation drops to ~37% (= 1/e).

    Practical range (correlation ≈ 5%) is approximately 3φ.
    """

    def __init__(self, coordinates: np.ndarray | None = None):
        """Initialize with spatial coordinates.

        Parameters
        ----------
        coordinates : ndarray (n, d), optional
            Spatial coordinates for n locations in d dimensions.
            If not provided, assumes 1D equally-spaced locations.
        """
        self.coordinates = coordinates
        self._distance_matrix = None

    def _compute_distances(self, n_effects: int) -> np.ndarray:
        """Compute pairwise distance matrix."""
        if self._distance_matrix is not None:
            if self._distance_matrix.shape[0] == n_effects:
                return self._distance_matrix

        if self.coordinates is not None:
            if len(self.coordinates) != n_effects:
                raise ValueError(
                    f"Coordinates have {len(self.coordinates)} locations, "
                    f"but n_effects is {n_effects}"
                )
            # Compute Euclidean distances
            from scipy.spatial.distance import pdist, squareform
            self._distance_matrix = squareform(pdist(self.coordinates))
        else:
            # Assume 1D equally-spaced
            i, j = np.ogrid[:n_effects, :n_effects]
            self._distance_matrix = np.abs(i - j).astype(float)

        return self._distance_matrix

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = 2 (variance and range)."""
        return 2

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct exponential spatial covariance matrix.

        Parameters
        ----------
        params : ndarray
            [log(σ²), log(φ)] - log-variance and log-range
        n_effects : int
            Number of spatial locations

        Returns
        -------
        psi : ndarray (n_effects, n_effects)
            Spatial covariance matrix
        """
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(params)}")

        sigma2 = np.exp(params[0])
        phi = np.exp(params[1])

        # Get distance matrix
        D = self._compute_distances(n_effects)

        # Exponential covariance: σ² exp(-d/φ)
        psi = sigma2 * np.exp(-D / phi)

        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from spatial covariance matrix.

        Uses method of moments estimation based on variogram fitting.
        """
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        n = psi.shape[0]

        # Estimate σ² from diagonal
        sigma2 = np.mean(np.diag(psi))

        # Get distances
        D = self._compute_distances(n)

        # Estimate φ from log-linear regression on off-diagonals
        # log(cov / σ²) = -d / φ
        mask = ~np.eye(n, dtype=bool)
        distances = D[mask]
        covs = psi[mask]

        if np.any(covs > 0):
            log_ratio = np.log(np.maximum(covs / sigma2, 1e-10))
            # Weighted regression (closer pairs have more info)
            weights = 1.0 / (distances + 1)
            phi_est = -np.sum(weights * distances) / np.sum(weights * log_ratio)
            phi = max(phi_est, 0.1)  # Ensure positive
        else:
            phi = 1.0

        params = np.array([np.log(sigma2), np.log(phi)])
        return params


class MaternCovariance(CovarianceStructure):
    """Matérn covariance structure for spatial data.

    The Matérn family is the most commonly used in geostatistics due
    to its flexibility in modeling different degrees of smoothness.

    Cov(d) = σ² × (2^(1-ν) / Γ(ν)) × (√(2ν) d/φ)^ν × K_ν(√(2ν) d/φ)

    where K_ν is the modified Bessel function of the second kind.

    Special cases:
    - ν = 0.5: Exponential covariance
    - ν = 1.5: Once differentiable
    - ν = 2.5: Twice differentiable
    - ν → ∞: Gaussian (squared exponential)

    Parameters (for fixed ν):
    - params[0]: log(σ²) - log-variance (sill)
    - params[1]: log(φ) - log-range parameter

    References
    ----------
    .. [1] Matérn (1960). Spatial Variation.
    .. [2] Stein (1999). Interpolation of Spatial Data.
    .. [3] Rasmussen & Williams (2006). Gaussian Processes for ML.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 0], [2, 0]])
    >>> cov = MaternCovariance(coordinates=coords, nu=1.5)
    >>> params = np.array([0.0, 0.0])  # σ²=1, φ=1
    >>> psi = cov.construct_psi(params, n_effects=3)
    """

    def __init__(
        self,
        coordinates: np.ndarray | None = None,
        nu: float = 1.5
    ):
        """Initialize Matérn covariance.

        Parameters
        ----------
        coordinates : ndarray (n, d), optional
            Spatial coordinates. If None, assumes 1D equally-spaced.
        nu : float, default=1.5
            Smoothness parameter. Common choices: 0.5, 1.5, 2.5
        """
        self.coordinates = coordinates
        self.nu = nu
        self._distance_matrix = None

    def _compute_distances(self, n_effects: int) -> np.ndarray:
        """Compute pairwise distance matrix."""
        if self._distance_matrix is not None:
            if self._distance_matrix.shape[0] == n_effects:
                return self._distance_matrix

        if self.coordinates is not None:
            if len(self.coordinates) != n_effects:
                raise ValueError(
                    f"Coordinates have {len(self.coordinates)} locations, "
                    f"but n_effects is {n_effects}"
                )
            from scipy.spatial.distance import pdist, squareform
            self._distance_matrix = squareform(pdist(self.coordinates))
        else:
            i, j = np.ogrid[:n_effects, :n_effects]
            self._distance_matrix = np.abs(i - j).astype(float)

        return self._distance_matrix

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = 2 (variance and range)."""
        return 2

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct Matérn covariance matrix.

        Parameters
        ----------
        params : ndarray
            [log(σ²), log(φ)]
        n_effects : int
            Number of spatial locations

        Returns
        -------
        psi : ndarray (n_effects, n_effects)
            Matérn covariance matrix
        """
        if len(params) != 2:
            raise ValueError(f"Expected 2 parameters, got {len(params)}")

        from scipy.special import gamma, kv

        sigma2 = np.exp(params[0])
        phi = np.exp(params[1])
        nu = self.nu

        D = self._compute_distances(n_effects)

        # Matérn formula
        # Handle d=0 separately (limit is σ²)
        scaled_d = np.sqrt(2 * nu) * D / phi

        # Compute Matérn covariance
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = (2**(1 - nu)) / gamma(nu)
            psi = sigma2 * factor * (scaled_d ** nu) * kv(nu, scaled_d)

        # Fix diagonal (d=0 case: cov = σ²)
        np.fill_diagonal(psi, sigma2)

        # Handle numerical issues
        psi = np.nan_to_num(psi, nan=0.0)

        # Ensure symmetry
        psi = (psi + psi.T) / 2

        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from Matérn covariance matrix."""
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        n = psi.shape[0]

        # Estimate σ² from diagonal
        sigma2 = np.mean(np.diag(psi))

        # Estimate φ using method of moments (simplified)
        D = self._compute_distances(n)
        mask = ~np.eye(n, dtype=bool)
        distances = D[mask]
        covs = psi[mask]

        # Find correlation at various distances
        if np.any(covs > 0) and np.any(distances > 0):
            # Use effective range: distance where corr ≈ 0.05
            corrs = covs / sigma2
            # Simple estimation based on decay
            valid = (corrs > 0.01) & (distances > 0)
            if np.any(valid):
                # φ ≈ -median_dist / log(median_corr)
                med_d = np.median(distances[valid])
                med_c = np.median(corrs[valid])
                phi = -med_d / np.log(med_c + 0.01)
                phi = np.clip(phi, 0.1, 100)
            else:
                phi = 1.0
        else:
            phi = 1.0

        params = np.array([np.log(sigma2), np.log(phi)])
        return params


class ToeplitzCovariance(CovarianceStructure):
    """Toeplitz (banded) covariance structure for temporal data.

    Toeplitz matrices have constant values along diagonals, meaning the
    covariance between observations depends only on their lag |t-s|:

    Cov(b_t, b_s) = r_{|t-s|}

    This is a generalization of AR(1) that allows arbitrary values at
    each lag, rather than assuming exponential decay.

    For q random effects (time points), requires (band+1) parameters where
    band is the number of non-zero off-diagonal bands:
    - params[0]: log(r_0) - log-variance
    - params[k]: arctanh(r_k / r_0) for k = 1, ..., band (transformed correlations)

    Parameters
    ----------
    band : int, default=2
        Number of off-diagonal bands to include. Higher values allow
        more complex temporal patterns but require more parameters.
        - band=1: First-order correlations only (like AR(1) but with free parameter)
        - band=2: First and second-order correlations

    References
    ----------
    .. [1] Pinheiro & Bates (2000). Mixed-Effects Models in S and S-PLUS. Ch. 5.
    .. [2] Pourahmadi (1999). "Joint mean-covariance models with applications
           to longitudinal data." Biometrika, 86(3), 677-690.

    Examples
    --------
    >>> cov = ToeplitzCovariance(band=2)
    >>> cov.n_parameters(5)
    3

    >>> params = np.array([0.0, 0.3, 0.1])  # log(σ²)=0, r_1/σ²≈0.29, r_2/σ²≈0.10
    >>> psi = cov.construct_psi(params, n_effects=4)

    Notes
    -----
    Toeplitz covariance is appropriate for:
    - Equally-spaced longitudinal data
    - Time series where correlations don't follow AR(p) exactly
    - Exploratory analysis before fitting structured models

    The Toeplitz structure is a special case of a block structure and
    allows efficient computation using banded matrix algorithms.
    """

    def __init__(self, band: int = 2):
        """Initialize Toeplitz covariance.

        Parameters
        ----------
        band : int, default=2
            Number of off-diagonal bands (lags) to model.
        """
        if band < 1:
            raise ValueError("band must be >= 1")
        self.band = band

    def n_parameters(self, n_effects: int) -> int:
        """Number of parameters = min(band, n_effects-1) + 1."""
        # Can't have more bands than n_effects - 1
        effective_band = min(self.band, n_effects - 1)
        return effective_band + 1  # variance + correlations

    def construct_psi(self, params: np.ndarray, n_effects: int) -> np.ndarray:
        """Construct Toeplitz covariance matrix.

        Parameters
        ----------
        params : ndarray
            [log(σ²), arctanh(ρ_1), arctanh(ρ_2), ...] - transformed parameters
        n_effects : int
            Number of time points/random effects

        Returns
        -------
        psi : ndarray (n_effects, n_effects)
            Toeplitz covariance matrix
        """
        n_params = self.n_parameters(n_effects)
        if len(params) != n_params:
            raise ValueError(f"Expected {n_params} parameters, got {len(params)}")

        # Extract variance and correlations
        sigma2 = np.exp(params[0])

        # Build correlation vector
        # Use tanh to ensure correlations in (-1, 1)
        correlations = np.tanh(params[1:])

        # Build first row of Toeplitz matrix (defines the structure)
        first_row = np.zeros(n_effects)
        first_row[0] = sigma2

        effective_band = len(correlations)
        for k in range(effective_band):
            first_row[k + 1] = sigma2 * correlations[k]

        # Construct symmetric Toeplitz matrix from first row
        from scipy.linalg import toeplitz
        psi = toeplitz(first_row)

        # Ensure positive definiteness by checking eigenvalues
        # If not positive definite, add small regularization
        eigvals = np.linalg.eigvalsh(psi)
        if np.min(eigvals) < 1e-10:
            # Add regularization to make positive definite
            psi += (1e-8 - np.min(eigvals) + 1e-10) * np.eye(n_effects)

        return psi

    def extract_params(self, psi: np.ndarray) -> np.ndarray:
        """Extract parameters from Toeplitz covariance matrix.

        Estimates variance from diagonal and correlations from off-diagonals.
        """
        if psi.shape[0] != psi.shape[1]:
            raise ValueError("psi must be square")

        n = psi.shape[0]
        n_params = self.n_parameters(n)

        # Extract variance from diagonal
        sigma2 = np.mean(np.diag(psi))

        # Extract correlations from off-diagonals
        params = [np.log(sigma2)]
        effective_band = n_params - 1

        for k in range(1, effective_band + 1):
            if k < n:
                # Average correlation at lag k
                diag_k = np.diag(psi, k)
                corr_k = np.mean(diag_k) / sigma2
                corr_k = np.clip(corr_k, -0.999, 0.999)
                params.append(np.arctanh(corr_k))
            else:
                params.append(0.0)

        return np.array(params)


def get_covariance_structure(
    structure: str,
    **kwargs
) -> CovarianceStructure:
    """Get covariance structure instance by name.

    Parameters
    ----------
    structure : {'unstructured', 'diagonal', 'identity', 'ar1', 
                 'compound_symmetry', 'cs', 'exponential', 'matern'}
        Covariance structure name
    **kwargs : dict
        Additional arguments for specific structures:
        - coordinates: ndarray for spatial structures
        - nu: smoothness for Matérn

    Returns
    -------
    CovarianceStructure
        Covariance structure instance

    Raises
    ------
    ValueError
        If structure name is not recognized

    Examples
    --------
    >>> cov = get_covariance_structure('unstructured')
    >>> isinstance(cov, UnstructuredCovariance)
    True

    >>> cov = get_covariance_structure('ar1')
    >>> isinstance(cov, AR1Covariance)
    True

    >>> coords = np.array([[0, 0], [1, 0], [0, 1]])
    >>> cov = get_covariance_structure('exponential', coordinates=coords)
    >>> isinstance(cov, ExponentialSpatialCovariance)
    True

    >>> cov = get_covariance_structure('matern', coordinates=coords, nu=2.5)
    >>> cov.nu
    2.5
    """
    # Structures without extra arguments
    simple_structures = {
        'unstructured': UnstructuredCovariance,
        'diagonal': DiagonalCovariance,
        'identity': IdentityCovariance,
        'ar1': AR1Covariance,
        'compound_symmetry': CompoundSymmetryCovariance,
        'cs': CompoundSymmetryCovariance,  # Alias
    }

    if structure in simple_structures:
        return simple_structures[structure]()

    # Toeplitz with optional band parameter
    if structure == 'toeplitz':
        band = kwargs.get('band', 2)
        return ToeplitzCovariance(band=band)

    # Structures with coordinates
    if structure == 'exponential':
        return ExponentialSpatialCovariance(
            coordinates=kwargs.get('coordinates')
        )

    if structure == 'matern':
        return MaternCovariance(
            coordinates=kwargs.get('coordinates'),
            nu=kwargs.get('nu', 1.5)
        )

    # Unknown structure
    all_structures = list(simple_structures.keys()) + ['toeplitz', 'exponential', 'matern']
    raise ValueError(
        f"Unknown covariance structure: '{structure}'. "
        f"Must be one of {all_structures}"
    )


__all__ = [
    'CovarianceStructure',
    'UnstructuredCovariance',
    'DiagonalCovariance',
    'IdentityCovariance',
    'AR1Covariance',
    'CompoundSymmetryCovariance',
    'ToeplitzCovariance',
    'ExponentialSpatialCovariance',
    'MaternCovariance',
    'get_covariance_structure',
]
