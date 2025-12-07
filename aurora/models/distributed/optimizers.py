# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Optimizers for mini-batch stochastic gradient descent.

This module provides optimizer implementations for training GLMs
using stochastic gradient descent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Optimizer", "SGDOptimizer", "AdamOptimizer", "AdaGradOptimizer"]


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def step(self, params: NDArray, grad: NDArray) -> NDArray:
        """Perform one optimization step.

        Parameters
        ----------
        params : ndarray
            Current parameter values
        grad : ndarray
            Gradient of the loss w.r.t. parameters

        Returns
        -------
        ndarray
            Updated parameter values
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass


@dataclass
class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent with optional momentum.

    Parameters
    ----------
    learning_rate : float
        Step size for parameter updates
    momentum : float
        Momentum coefficient (0 for vanilla SGD)
    nesterov : bool
        Use Nesterov momentum
    weight_decay : float
        L2 regularization coefficient

    Examples
    --------
    >>> opt = SGDOptimizer(learning_rate=0.01, momentum=0.9)
    >>> for batch in data_iterator:
    ...     grad = compute_gradient(params, batch)
    ...     params = opt.step(params, grad)
    """

    learning_rate: float = 0.01
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0

    _velocity: NDArray | None = field(default=None, repr=False)

    def step(self, params: NDArray, grad: NDArray) -> NDArray:
        # Apply weight decay (L2 regularization)
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * params

        if self.momentum > 0:
            if self._velocity is None:
                self._velocity = np.zeros_like(params)

            self._velocity = self.momentum * self._velocity + grad

            if self.nesterov:
                update = self.momentum * self._velocity + grad
            else:
                update = self._velocity
        else:
            update = grad

        return params - self.learning_rate * update

    def reset(self):
        self._velocity = None


@dataclass
class AdamOptimizer(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation).

    Combines momentum with adaptive learning rates per parameter.

    Parameters
    ----------
    learning_rate : float
        Base learning rate
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    eps : float
        Small constant for numerical stability
    weight_decay : float
        L2 regularization coefficient

    References
    ----------
    .. [1] Kingma, D. P., & Ba, J. (2015). Adam: A Method for
           Stochastic Optimization. ICLR.

    Examples
    --------
    >>> opt = AdamOptimizer(learning_rate=0.001)
    >>> for batch in data_iterator:
    ...     grad = compute_gradient(params, batch)
    ...     params = opt.step(params, grad)
    """

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    _m: NDArray | None = field(default=None, repr=False)
    _v: NDArray | None = field(default=None, repr=False)
    _t: int = field(default=0, repr=False)

    def step(self, params: NDArray, grad: NDArray) -> NDArray:
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)

        # Apply weight decay
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * params

        self._t += 1

        # Update biased first moment estimate
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad**2

        # Compute bias-corrected estimates
        m_hat = self._m / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)

        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self._m = None
        self._v = None
        self._t = 0


@dataclass
class AdaGradOptimizer(Optimizer):
    """AdaGrad optimizer with adaptive learning rates.

    AdaGrad adapts learning rates based on historical gradient magnitudes,
    performing larger updates for infrequent parameters.

    Parameters
    ----------
    learning_rate : float
        Base learning rate
    eps : float
        Small constant for numerical stability

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient
           Methods for Online Learning and Stochastic Optimization. JMLR.
    """

    learning_rate: float = 0.01
    eps: float = 1e-8

    _g_sum: NDArray | None = field(default=None, repr=False)

    def step(self, params: NDArray, grad: NDArray) -> NDArray:
        if self._g_sum is None:
            self._g_sum = np.zeros_like(params)

        # Accumulate squared gradients
        self._g_sum += grad**2

        # Update with adaptive learning rate
        return params - self.learning_rate * grad / (np.sqrt(self._g_sum) + self.eps)

    def reset(self):
        self._g_sum = None
