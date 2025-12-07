# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Binomial distribution family implementation."""

from __future__ import annotations

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, clip_probability, namespace
from ..links import LogitLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


def _safe_log(value, xp, eps: float = 1e-12):
    """Compute log with numeric stability across backends.

    Parameters
    ----------
    value : array
        Input array
    xp : module
        Array namespace (np, torch, or jnp)
    eps : float, default=1e-12
        Minimum value for numerical stability

    Returns
    -------
    array
        log(max(value, eps))

    Notes
    -----
    Uses consistent epsilon (1e-12) across all backends for numerical stability.
    Creates tensor with correct dtype and device for PyTorch/JAX compatibility.
    """
    if xp is torch:  # type: ignore[comparison-overlap]
        eps_tensor = torch.tensor(eps, dtype=value.dtype, device=value.device)
        return torch.log(torch.clamp(value, min=eps_tensor))
    elif xp is jnp:  # type: ignore[comparison-overlap]
        return jnp.log(jnp.clip(value, eps, None))
    return np.log(np.clip(value, eps, None))


class BinomialFamily(Family):
    """Binomial family with optional trials parameter ``n`` and link."""

    def __init__(self, n: float = 1.0, link: LinkFunction | None = None) -> None:
        self._n = n
        self._link = link or LogitLink()

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)
        probability = clip_probability(mu_arr / n_arr, xp)
        term1 = y_arr * _safe_log(probability, xp)
        term2 = (n_arr - y_arr) * _safe_log(1.0 - probability, xp)
        return (term1 + term2).sum()

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        """Compute binomial deviance with consistent epsilon handling.

        Uses _safe_log() for consistency with log_likelihood().
        """
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)

        eps = 1e-12

        # Clamp y and mu to valid range [eps, n-eps]
        if xp is torch:
            eps_tensor = torch.tensor(eps, dtype=mu_arr.dtype, device=mu_arr.device)
            y_safe = torch.clamp(y_arr, min=eps_tensor, max=n_arr - eps_tensor)
            mu_safe = torch.clamp(mu_arr, min=eps_tensor, max=n_arr - eps_tensor)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            y_safe = jnp.clip(y_arr, eps, n_arr - eps)
            mu_safe = jnp.clip(mu_arr, eps, n_arr - eps)
        else:
            y_safe = np.clip(y_arr, eps, n_arr - eps)
            mu_safe = np.clip(mu_arr, eps, n_arr - eps)

        # Use _safe_log for consistency with log_likelihood
        term1 = y_arr * _safe_log(y_safe / mu_safe, xp, eps=eps)
        term2 = (n_arr - y_arr) * _safe_log(
            (n_arr - y_safe) / (n_arr - mu_safe), xp, eps=eps
        )

        return (2.0 * (term1 + term2)).sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = as_namespace_array(mu, xp, like=mu)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)
        probability = clip_probability(mu_arr / n_arr, xp)
        return n_arr * probability * (1.0 - probability)

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)
        n_arr = as_namespace_array(self._n, xp, like=y_arr)
        p_init = clip_probability((y_arr + 0.5) / (n_arr + 1.0), xp)
        return n_arr * p_init

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["BinomialFamily"]
