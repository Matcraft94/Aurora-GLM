"""Core type definitions and protocols for Aurora-GLM."""
from __future__ import annotations

from typing import Any, Callable, Protocol, TypeAlias, Union

import numpy as np

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp  # type: ignore

    JAXArray = jnp.ndarray  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    JAXArray = Any

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore

    TorchTensor = torch.Tensor  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    TorchTensor = Any

# Type aliases for arrays
Array: TypeAlias = Union[np.ndarray, JAXArray, TorchTensor]
Scalar: TypeAlias = Union[int, float, complex]

# Shape and dtype types
Shape: TypeAlias = tuple[int, ...]
DType: TypeAlias = Any  # numpy.dtype, jax dtype, or torch dtype

# Optimizer callback types
OptimizationCallback: TypeAlias = Callable[[int, Array, float], None]

# Loss function type
LossFunction: TypeAlias = Callable[..., Scalar]


class ArrayLike(Protocol):
    """Protocol for array-like objects."""

    def __array__(self) -> np.ndarray:  # pragma: no cover - structural typing hook
        ...

    @property
    def shape(self) -> Shape:
        ...

    @property
    def dtype(self) -> DType:
        ...


class Distribution(Protocol):
    """Protocol for probability distributions."""

    def log_likelihood(self, y: Array, mu: Array, **params: Any) -> Scalar:
        """Compute log-likelihood."""

    def deviance(self, y: Array, mu: Array, **params: Any) -> Scalar:
        """Compute deviance."""

    def variance(self, mu: Array, **params: Any) -> Array:
        """Compute variance function."""

    def initialize(self, y: Array) -> Array:
        """Get starting values for mu."""


class Link(Protocol):
    """Protocol for link functions."""

    def link(self, mu: Array) -> Array:
        """Apply link function g(mu)."""

    def inverse(self, eta: Array) -> Array:
        """Apply inverse link g^{-1}(eta)."""

    def derivative(self, mu: Array) -> Array:
        """Compute derivative dg/dmu."""


class OptimizationResult(Protocol):
    """Protocol for optimization results."""

    @property
    def x(self) -> Array:
        """Optimal parameters."""

    @property
    def fun(self) -> Scalar:
        """Function value at optimum."""

    @property
    def success(self) -> bool:
        """Whether optimization succeeded."""

    @property
    def message(self) -> str:
        """Status message."""

    @property
    def nit(self) -> int:
        """Number of iterations."""


class Optimizer(Protocol):
    """Protocol for optimization algorithms."""

    def minimize(
        self,
        loss_fn: LossFunction,
        init_params: Array,
        *,
        callback: OptimizationCallback | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Minimize loss function."""


__all__ = [
    "Array",
    "ArrayLike",
    "Scalar",
    "Shape",
    "DType",
    "OptimizationCallback",
    "LossFunction",
    "Distribution",
    "Link",
    "Optimizer",
    "OptimizationResult",
    "JAXArray",
    "TorchTensor",
]
