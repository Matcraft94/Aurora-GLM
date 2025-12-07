# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Data-parallel IRLS for distributed GLM fitting.

This module provides data-parallel implementations of IRLS that
distribute computation across workers while aggregating sufficient
statistics centrally.

The key insight is that IRLS only requires X'WX and X'Wz, which
can be computed locally and summed across workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["fit_glm_parallel", "DataParallelIRLS", "ParallelResult"]


@dataclass
class ParallelResult:
    """Results from data-parallel GLM fitting.

    Attributes
    ----------
    coef_ : ndarray
        Fitted coefficients
    converged_ : bool
        Whether IRLS converged
    n_iter_ : int
        Number of iterations
    n_obs_ : int
        Total number of observations across all chunks
    n_chunks_ : int
        Number of data chunks processed
    """

    coef_: NDArray
    converged_: bool
    n_iter_: int
    n_obs_: int
    n_chunks_: int
    family: str
    link: str

    def predict(self, X: ArrayLike, type: str = "response") -> NDArray:
        """Generate predictions."""
        X = np.asarray(X)
        eta = X @ self.coef_

        if type == "link":
            return eta
        else:
            return _apply_inverse_link(eta, self.link)

    def summary(self) -> dict:
        """Return model summary."""
        return {
            "n_obs": self.n_obs_,
            "n_features": len(self.coef_),
            "n_chunks": self.n_chunks_,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "family": self.family,
            "link": self.link,
            "coef": self.coef_,
        }


class DataParallelIRLS:
    """Data-parallel IRLS algorithm.

    Distributes IRLS computation by computing sufficient statistics
    (X'WX and X'Wz) on each data chunk and aggregating.

    At each iteration:
    1. Broadcast current beta to all chunks
    2. Each chunk computes local X'WX and X'Wz
    3. Sum across chunks: X'WX_total, X'Wz_total
    4. Solve: beta_new = (X'WX_total)^-1 X'Wz_total

    Memory requirement: O(p^2) per chunk + O(n_chunk) data.

    Parameters
    ----------
    family : str
        Distribution family
    link : str, optional
        Link function (default: canonical)
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
    """

    def __init__(
        self,
        family: str = "gaussian",
        link: str | None = None,
        max_iter: int = 25,
        tol: float = 1e-6,
    ):
        self.family = family
        self.link = link if link else _canonical_link(family)
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X_chunks: list[NDArray] | Iterator[NDArray],
        y_chunks: list[NDArray] | Iterator[NDArray],
    ) -> ParallelResult:
        """Fit GLM using data-parallel IRLS.

        Parameters
        ----------
        X_chunks : list or iterator of ndarray
            List of design matrix chunks
        y_chunks : list or iterator of ndarray
            List of response vector chunks

        Returns
        -------
        ParallelResult
            Fitted model result
        """
        # Convert to lists if iterators
        if not isinstance(X_chunks, list):
            X_chunks = list(X_chunks)
        if not isinstance(y_chunks, list):
            y_chunks = list(y_chunks)

        n_chunks = len(X_chunks)
        n_total = sum(len(y) for y in y_chunks)
        p = X_chunks[0].shape[1]

        # Initialize beta
        beta = np.zeros(p)

        converged = False
        for iteration in range(self.max_iter):
            # Aggregate sufficient statistics across chunks
            XtWX_total = np.zeros((p, p))
            XtWz_total = np.zeros(p)

            for X_chunk, y_chunk in zip(X_chunks, y_chunks):
                XtWX, XtWz = self._compute_sufficient_stats(
                    X_chunk, y_chunk, beta
                )
                XtWX_total += XtWX
                XtWz_total += XtWz

            # Solve for new beta
            # Add small ridge for numerical stability
            XtWX_total += 1e-8 * np.eye(p)

            try:
                beta_new = np.linalg.solve(XtWX_total, XtWz_total)
            except np.linalg.LinAlgError:
                # Fall back to pseudoinverse
                beta_new = np.linalg.lstsq(XtWX_total, XtWz_total, rcond=None)[0]

            # Check convergence
            if np.max(np.abs(beta_new - beta)) < self.tol:
                converged = True
                beta = beta_new
                break

            beta = beta_new

        return ParallelResult(
            coef_=beta,
            converged_=converged,
            n_iter_=iteration + 1,
            n_obs_=n_total,
            n_chunks_=n_chunks,
            family=self.family,
            link=self.link,
        )

    def _compute_sufficient_stats(
        self, X: NDArray, y: NDArray, beta: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Compute X'WX and X'Wz for one data chunk.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Design matrix chunk
        y : ndarray, shape (n,)
            Response chunk
        beta : ndarray, shape (p,)
            Current coefficient estimates

        Returns
        -------
        XtWX : ndarray, shape (p, p)
            X'WX for this chunk
        XtWz : ndarray, shape (p,)
            X'Wz for this chunk
        """
        n = len(y)
        eta = X @ beta
        mu = _apply_inverse_link(eta, self.link)

        # Working weights and response
        W, z = self._compute_working_quantities(y, mu, eta)

        # Sufficient statistics
        XtW = X.T * W  # Broadcasting: (p, n) * (n,) -> (p, n)
        XtWX = XtW @ X  # (p, p)
        XtWz = XtW @ z  # (p,)

        return XtWX, XtWz

    def _compute_working_quantities(
        self, y: NDArray, mu: NDArray, eta: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Compute IRLS working weights and response.

        Returns
        -------
        W : ndarray
            Working weights
        z : ndarray
            Working response
        """
        # Variance function
        V = _variance_function(mu, self.family)

        # Link derivative g'(mu)
        deta_dmu = _link_derivative(mu, self.link)

        # Working weights: W = 1 / (V * (g'(mu))^2)
        W = 1 / (V * deta_dmu**2 + 1e-10)
        W = np.clip(W, 1e-10, 1e10)

        # Working response: z = eta + (y - mu) * g'(mu)
        z = eta + (y - mu) * deta_dmu

        return W, z


def fit_glm_parallel(
    X_chunks: list[ArrayLike] | Iterator[ArrayLike],
    y_chunks: list[ArrayLike] | Iterator[ArrayLike],
    family: str = "gaussian",
    link: str | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> ParallelResult:
    """Fit GLM using data-parallel IRLS.

    This is a convenience function that creates a DataParallelIRLS
    instance and fits the model.

    Parameters
    ----------
    X_chunks : list or iterator
        Design matrix chunks
    y_chunks : list or iterator
        Response vector chunks
    family : str
        Distribution family
    link : str, optional
        Link function
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    ParallelResult
        Fitted model

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.distributed import fit_glm_parallel
    >>>
    >>> # Split data into chunks
    >>> X = np.random.randn(1000, 5)
    >>> y = X @ [1, -0.5, 0.3, 0, 0.2] + np.random.randn(1000) * 0.5
    >>>
    >>> X_chunks = np.array_split(X, 10)
    >>> y_chunks = np.array_split(y, 10)
    >>>
    >>> result = fit_glm_parallel(X_chunks, y_chunks, family='gaussian')
    >>> print(f"Coefficients: {result.coef_}")
    """
    # Convert to numpy arrays
    X_chunks = [np.asarray(X, dtype=np.float64) for X in X_chunks]
    y_chunks = [np.asarray(y, dtype=np.float64).ravel() for y in y_chunks]

    irls = DataParallelIRLS(family=family, link=link, max_iter=max_iter, tol=tol)
    return irls.fit(X_chunks, y_chunks)


def _canonical_link(family: str) -> str:
    """Return canonical link for family."""
    canonical = {
        "gaussian": "identity",
        "poisson": "log",
        "binomial": "logit",
        "gamma": "inverse",
    }
    return canonical.get(family, "identity")


def _apply_inverse_link(eta: NDArray, link: str) -> NDArray:
    """Apply inverse link function."""
    if link == "identity":
        return eta
    elif link == "log":
        return np.exp(np.clip(eta, -20, 20))
    elif link == "logit":
        return 1 / (1 + np.exp(-np.clip(eta, -20, 20)))
    elif link == "inverse":
        return 1 / np.maximum(np.abs(eta), 1e-10)
    else:
        return eta


def _variance_function(mu: NDArray, family: str) -> NDArray:
    """Compute variance function V(mu)."""
    if family == "gaussian":
        return np.ones_like(mu)
    elif family == "poisson":
        return np.maximum(mu, 1e-10)
    elif family == "binomial":
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return mu * (1 - mu)
    elif family == "gamma":
        return np.maximum(mu**2, 1e-10)
    else:
        return np.ones_like(mu)


def _link_derivative(mu: NDArray, link: str) -> NDArray:
    """Compute derivative deta/dmu = g'(mu)."""
    if link == "identity":
        return np.ones_like(mu)
    elif link == "log":
        return 1 / np.maximum(mu, 1e-10)
    elif link == "logit":
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return 1 / (mu * (1 - mu))
    elif link == "inverse":
        return -1 / np.maximum(mu**2, 1e-10)
    else:
        return np.ones_like(mu)
