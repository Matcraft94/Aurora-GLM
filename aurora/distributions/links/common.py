"""Common link function implementations."""
from __future__ import annotations

import numpy as np

from ..base import LinkFunction
from .._utils import as_namespace_array, clip_probability, namespace, ones_like

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _ensure_positive(value, xp, eps: float = 1e-12):
    if xp is torch:  # type: ignore[comparison-overlap]
        eps_tensor = torch.tensor(eps, dtype=value.dtype, device=value.device)
        return torch.clamp(value, min=eps_tensor)
    return np.clip(value, eps, None)


class IdentityLink(LinkFunction):
    """Identity link ``g(mu) = mu``."""

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        return as_namespace_array(mu, xp, like=mu)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        return as_namespace_array(eta, xp, like=eta)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        mu_arr = as_namespace_array(mu, namespace(mu), like=mu)
        return ones_like(mu_arr)


class LogLink(LinkFunction):
    """Log link ``g(mu) = log(mu)``."""

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return xp.log(mu_arr)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        return xp.exp(eta_arr)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return 1.0 / mu_arr


class LogitLink(LinkFunction):
    """Logit link ``g(mu) = log(mu / (1 - mu))``."""

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        return xp.log(mu_arr / (1.0 - mu_arr))

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        return 1.0 / (1.0 + xp.exp(-eta_arr))

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        return 1.0 / (mu_arr * (1.0 - mu_arr))


__all__ = ["IdentityLink", "LogLink", "LogitLink"]
