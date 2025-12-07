# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Poisson distribution family implementation."""

from __future__ import annotations

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, ensure_positive, namespace
from ..links import LogLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


class PoissonFamily(Family):
    """Poisson distribution family."""

    def __init__(self, link: LinkFunction | None = None) -> None:
        self._link = link or LogLink()

    def log_likelihood(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)
        return (y_arr * xp.log(mu_arr) - mu_arr).sum()

    def deviance(self, y, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=y_arr), xp)
        if xp is torch:  # type: ignore[comparison-overlap]
            ones = torch.ones_like(mu_arr)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            ones = jnp.ones_like(mu_arr)
        else:
            ones = np.ones_like(mu_arr)
        ratio = xp.where(y_arr == 0, ones, y_arr / mu_arr)
        log_term = xp.log(ratio)
        return (2.0 * (y_arr * log_term - (y_arr - mu_arr))).sum()

    def variance(self, mu, **params):  # noqa: ANN001 - match Family signature
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return mu_arr

    def initialize(self, y):  # noqa: ANN001 - match Family signature
        xp = namespace(y)
        y_arr = as_namespace_array(y, xp, like=y)
        if xp is torch:  # type: ignore[comparison-overlap]
            return torch.clamp(y_arr + 0.1, min=0.1)
        elif xp is jnp:  # type: ignore[comparison-overlap]
            return jnp.clip(y_arr + 0.1, 0.1, None)
        return np.clip(y_arr + 0.1, 0.1, None)

    @property
    def default_link(self) -> LinkFunction:
        return self._link


__all__ = ["PoissonFamily"]
