# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Mini-batch stochastic gradient descent for GLM fitting.

This module provides functions for fitting GLMs using mini-batch SGD,
which is suitable for large datasets that don't fit in memory.

Examples
--------
>>> from aurora.models.distributed import fit_glm_sgd
>>>
>>> # Streaming from data iterator
>>> def data_iterator():
...     for i in range(100):
...         X_batch = load_batch_X(i)
...         y_batch = load_batch_y(i)
...         yield X_batch, y_batch
>>>
>>> result = fit_glm_sgd(data_iterator(), family='poisson')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

from .optimizers import AdamOptimizer, Optimizer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["fit_glm_sgd", "SGDResult"]


def fit_glm_sgd(
    data_iterator: Iterator[tuple[ArrayLike, ArrayLike]],
    family: str = "gaussian",
    link: str | None = None,
    optimizer: Optimizer | str = "adam",
    learning_rate: float = 0.01,
    max_epochs: int = 10,
    l2_penalty: float = 0.0,
    batch_size: int | None = None,
    n_features: int | None = None,
    verbose: bool = False,
    seed: int | None = None,
) -> "SGDResult":
    """Fit GLM using mini-batch stochastic gradient descent.

    This function fits a GLM by iterating through data in batches,
    computing gradients, and updating parameters. Suitable for:
    - Streaming data
    - Datasets too large for memory
    - Online learning scenarios

    Parameters
    ----------
    data_iterator : Iterator
        Iterator yielding (X_batch, y_batch) tuples
    family : str, default='gaussian'
        Distribution family ('gaussian', 'poisson', 'binomial', 'gamma')
    link : str, optional
        Link function. If None, uses canonical link.
    optimizer : Optimizer or str, default='adam'
        Optimizer for parameter updates. String options: 'adam', 'sgd', 'adagrad'
    learning_rate : float, default=0.01
        Learning rate for optimizer
    max_epochs : int, default=10
        Maximum number of passes through data
    l2_penalty : float, default=0.0
        L2 regularization strength
    batch_size : int, optional
        Batch size (used for reporting only if data iterator provides batches)
    n_features : int, optional
        Number of features. If None, inferred from first batch.
    verbose : bool, default=False
        Print progress information
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    SGDResult
        Fitted model with coefficients and training history

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.models.distributed import fit_glm_sgd
    >>>
    >>> # Create sample data iterator
    >>> def make_iterator():
    ...     np.random.seed(42)
    ...     for _ in range(50):  # 50 batches
    ...         X = np.random.randn(100, 3)  # 100 samples, 3 features
    ...         y = np.random.poisson(np.exp(X @ [1, -0.5, 0.3]))
    ...         yield X, y
    >>>
    >>> result = fit_glm_sgd(make_iterator(), family='poisson', max_epochs=5)
    >>> print(f"Coefficients: {result.coef_}")

    Notes
    -----
    The gradient for GLM with canonical link is:
        grad = X.T @ (mu - y) / n

    For non-canonical links, we use:
        grad = X.T @ ((mu - y) * dmu_deta) / n

    where dmu_deta is the derivative of the inverse link.
    """
    if seed is not None:
        np.random.seed(seed)

    # Setup optimizer
    if isinstance(optimizer, str):
        optimizer = _create_optimizer(optimizer, learning_rate)

    # Determine link function
    if link is None:
        link = _canonical_link(family)

    # Training state
    beta = None
    n_batches = 0
    n_samples_total = 0
    loss_history = []

    # Collect first pass data for re-iteration
    first_pass_data = []

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        # Use stored data for subsequent epochs, or iterate fresh for first epoch
        if epoch == 0:
            data_source = data_iterator
        else:
            data_source = iter(first_pass_data)

        for X_batch, y_batch in data_source:
            X_batch = np.asarray(X_batch, dtype=np.float64)
            y_batch = np.asarray(y_batch, dtype=np.float64).ravel()

            # Store for future epochs (first epoch only)
            if epoch == 0:
                first_pass_data.append((X_batch.copy(), y_batch.copy()))

            n_batch = len(y_batch)
            p = X_batch.shape[1]

            # Initialize beta on first batch
            if beta is None:
                if n_features is not None and p != n_features:
                    raise ValueError(
                        f"Expected {n_features} features but got {p}"
                    )
                beta = np.zeros(p)

            # Forward pass
            eta = X_batch @ beta
            mu = _apply_inverse_link(eta, link)

            # Compute gradient
            grad = _compute_gradient(X_batch, y_batch, mu, eta, family, link)

            # L2 regularization gradient
            if l2_penalty > 0:
                grad = grad + l2_penalty * beta

            # Update parameters
            beta = optimizer.step(beta, grad)

            # Compute batch loss for monitoring
            batch_loss = _compute_loss(y_batch, mu, family)
            epoch_loss += batch_loss * n_batch
            epoch_samples += n_batch

            if epoch == 0:
                n_batches += 1
                n_samples_total += n_batch

        # Record epoch loss
        avg_loss = epoch_loss / max(epoch_samples, 1)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{max_epochs}: loss = {avg_loss:.6f}")

    # Final predictions on all data
    all_mu = []
    for X_batch, _ in first_pass_data:
        eta = X_batch @ beta
        mu = _apply_inverse_link(eta, link)
        all_mu.append(mu)

    mu_all = np.concatenate(all_mu)

    return SGDResult(
        coef_=beta,
        mu_=mu_all,
        family=family,
        link=link,
        n_obs_=n_samples_total,
        n_batches_=n_batches,
        n_epochs_=max_epochs,
        loss_history_=np.array(loss_history),
    )


@dataclass
class SGDResult:
    """Results from SGD-based GLM fitting.

    Attributes
    ----------
    coef_ : ndarray
        Fitted coefficients
    mu_ : ndarray
        Fitted mean values
    family : str
        Distribution family used
    link : str
        Link function used
    n_obs_ : int
        Total number of observations
    n_batches_ : int
        Number of batches processed per epoch
    n_epochs_ : int
        Number of epochs completed
    loss_history_ : ndarray
        Loss at end of each epoch
    """

    coef_: NDArray
    mu_: NDArray
    family: str
    link: str
    n_obs_: int
    n_batches_: int
    n_epochs_: int
    loss_history_: NDArray

    def predict(self, X: ArrayLike, type: str = "response") -> NDArray:
        """Generate predictions.

        Parameters
        ----------
        X : array-like
            Design matrix for prediction
        type : str
            'response' for mean (mu), 'link' for linear predictor (eta)

        Returns
        -------
        ndarray
            Predictions
        """
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
            "n_batches": self.n_batches_,
            "n_epochs": self.n_epochs_,
            "final_loss": float(self.loss_history_[-1]),
            "family": self.family,
            "link": self.link,
            "coef": self.coef_,
        }


def _create_optimizer(name: str, learning_rate: float) -> Optimizer:
    """Create optimizer by name."""
    from .optimizers import SGDOptimizer, AdamOptimizer, AdaGradOptimizer

    if name == "adam":
        return AdamOptimizer(learning_rate=learning_rate)
    elif name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate)
    elif name == "adagrad":
        return AdaGradOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Use 'adam', 'sgd', or 'adagrad'")


def _canonical_link(family: str) -> str:
    """Return canonical link for family."""
    canonical = {
        "gaussian": "identity",
        "poisson": "log",
        "binomial": "logit",
        "gamma": "inverse",
        "negative_binomial": "log",
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
    elif link == "probit":
        from scipy.stats import norm

        return norm.cdf(eta)
    elif link == "inverse":
        return 1 / np.maximum(eta, 1e-10)
    elif link == "sqrt":
        return np.maximum(eta, 0) ** 2
    else:
        raise ValueError(f"Unknown link: {link}")


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
    elif link == "sqrt":
        return 0.5 / np.maximum(np.sqrt(mu), 1e-10)
    else:
        raise ValueError(f"Unknown link: {link}")


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
    elif family == "negative_binomial":
        return np.maximum(mu, 1e-10)  # Simplified
    else:
        return np.ones_like(mu)


def _compute_gradient(
    X: NDArray, y: NDArray, mu: NDArray, eta: NDArray, family: str, link: str
) -> NDArray:
    """Compute gradient of negative log-likelihood.

    For canonical link:
        grad = -X.T @ (y - mu) / n

    For non-canonical link:
        grad = -X.T @ ((y - mu) / V(mu) * dmu/deta) / n
    """
    n = len(y)
    residual = y - mu

    # For canonical links, gradient simplifies
    canonical = {
        "gaussian": "identity",
        "poisson": "log",
        "binomial": "logit",
    }

    if canonical.get(family) == link:
        # Canonical link: grad = -X.T @ (y - mu) / n
        grad = -X.T @ residual / n
    else:
        # Non-canonical: need variance function and link derivative
        V = _variance_function(mu, family)
        # dmu/deta = 1 / (deta/dmu)
        deta_dmu = _link_derivative(mu, link)
        dmu_deta = 1 / np.maximum(deta_dmu, 1e-10)

        # Gradient: -X.T @ (residual / V * dmu_deta) / n
        working_residual = residual / np.maximum(V, 1e-10) * dmu_deta
        grad = -X.T @ working_residual / n

    return grad


def _compute_loss(y: NDArray, mu: NDArray, family: str) -> float:
    """Compute negative log-likelihood loss."""
    if family == "gaussian":
        return float(0.5 * np.mean((y - mu) ** 2))
    elif family == "poisson":
        mu = np.maximum(mu, 1e-10)
        return float(-np.mean(y * np.log(mu) - mu))
    elif family == "binomial":
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        return float(-np.mean(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    elif family == "gamma":
        mu = np.maximum(mu, 1e-10)
        return float(np.mean(y / mu + np.log(mu)))
    else:
        return float(0.5 * np.mean((y - mu) ** 2))
