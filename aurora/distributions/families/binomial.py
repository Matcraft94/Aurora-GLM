"""Binomial distribution family implementation."""
from __future__ import annotations

import numpy as np

from ..base import Family, LinkFunction
from .._utils import as_namespace_array, clip_probability, namespace, ones_like
from ..links import LogitLink

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _safe_log(value, xp):
    if xp is torch:  # type: ignore[comparison-overlap]
        eps = torch.finfo(value.dtype).tiny
        return torch.log(torch.clamp(value, min=eps))
    return np.log(np.clip(value, 1e-12, None))


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
        xp = namespace(y, mu)
        y_arr = as_namespace_array(y, xp, like=mu)
        mu_arr = as_namespace_array(mu, xp, like=y_arr)
        n_param = params.get("n", self._n)
        n_arr = as_namespace_array(n_param, xp, like=mu_arr)

        eps = 1e-12
        if xp is torch:
            eps_tensor = torch.tensor(eps, dtype=mu_arr.dtype, device=mu_arr.device)
            y_safe = torch.clamp(y_arr, min=eps_tensor, max=n_arr - eps_tensor)
            mu_safe = torch.clamp(mu_arr, min=eps_tensor, max=n_arr - eps_tensor)
            term1 = y_arr * torch.log(y_safe / mu_safe)
            term2 = (n_arr - y_arr) * torch.log((n_arr - y_safe) / (n_arr - mu_safe))
            return (2.0 * (term1 + term2)).sum()

        y_safe = np.clip(y_arr, eps, n_arr - eps)
        mu_safe = np.clip(mu_arr, eps, n_arr - eps)
        term1 = y_arr * np.log(y_safe / mu_safe)
        term2 = (n_arr - y_arr) * np.log((n_arr - y_safe) / (n_arr - mu_safe))
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
