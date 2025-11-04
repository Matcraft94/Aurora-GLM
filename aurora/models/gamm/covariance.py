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


def get_covariance_structure(
    structure: str,
) -> CovarianceStructure:
    """Get covariance structure instance by name.

    Parameters
    ----------
    structure : {'unstructured', 'diagonal', 'identity'}
        Covariance structure name

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

    >>> cov = get_covariance_structure('diagonal')
    >>> isinstance(cov, DiagonalCovariance)
    True
    """
    structures = {
        'unstructured': UnstructuredCovariance,
        'diagonal': DiagonalCovariance,
        'identity': IdentityCovariance,
    }

    if structure not in structures:
        raise ValueError(
            f"Unknown covariance structure: '{structure}'. "
            f"Must be one of {list(structures.keys())}"
        )

    return structures[structure]()


__all__ = [
    'CovarianceStructure',
    'UnstructuredCovariance',
    'DiagonalCovariance',
    'IdentityCovariance',
    'get_covariance_structure',
]
