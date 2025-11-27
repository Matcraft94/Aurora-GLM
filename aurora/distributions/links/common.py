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


class InverseLink(LinkFunction):
    """Inverse link ``g(mu) = 1 / mu``."""

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return 1.0 / mu_arr

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = _ensure_positive(as_namespace_array(eta, xp, like=eta), xp)
        return 1.0 / eta_arr

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return -1.0 / (mu_arr**2)


class CLogLogLink(LinkFunction):
    """Complementary log-log link ``g(mu) = log(-log(1 - mu))``."""

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        one_minus = 1.0 - mu_arr
        one_minus = _ensure_positive(one_minus, xp)
        return xp.log(-xp.log(one_minus))

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        return 1.0 - xp.exp(-xp.exp(eta_arr))

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        one_minus = 1.0 - mu_arr
        one_minus = _ensure_positive(one_minus, xp)
        log_term = -xp.log(one_minus)
        return 1.0 / (log_term * one_minus)


class SqrtLink(LinkFunction):
    """Square root link ``g(mu) = sqrt(mu)``.
    
    Useful for count data where variance is proportional to mean.
    Common alternative to log link for Poisson-like data.
    """

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return xp.sqrt(mu_arr)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        return eta_arr ** 2

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return 0.5 / xp.sqrt(mu_arr)


class PowerLink(LinkFunction):
    """Power link ``g(mu) = mu^power``.
    
    General power transformation. Special cases:
    - power = 1: Identity link
    - power = 0: Log link (limit as power → 0)
    - power = -1: Inverse link
    - power = 0.5: Square root link
    - power = -2: Inverse square link
    
    Parameters
    ----------
    power : float
        Power parameter for the transformation.
        
    Notes
    -----
    The Box-Cox transformation is a special case when properly normalized.
    
    For power = 0, this class uses the log link as the limit.
    """

    def __init__(self, power: float = 1.0):
        """Initialize power link.
        
        Parameters
        ----------
        power : float
            Power parameter.
        """
        self.power = power
        self.name = f'power{power}'

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        
        if abs(self.power) < 1e-10:
            # Use log for power ≈ 0
            return xp.log(mu_arr)
        
        return mu_arr ** self.power

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        
        if abs(self.power) < 1e-10:
            # Use exp for power ≈ 0
            return xp.exp(eta_arr)
        
        # Ensure result is positive
        if self.power > 0:
            eta_arr = _ensure_positive(eta_arr, xp)
        
        return eta_arr ** (1.0 / self.power)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        
        if abs(self.power) < 1e-10:
            # Log link derivative: 1/μ
            return 1.0 / mu_arr
        
        return self.power * mu_arr ** (self.power - 1)


class InverseSquareLink(LinkFunction):
    """Inverse square link ``g(mu) = 1 / mu^2``.
    
    Canonical link for inverse Gaussian distribution.
    """

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return 1.0 / (mu_arr ** 2)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = _ensure_positive(as_namespace_array(eta, xp, like=eta), xp)
        return 1.0 / xp.sqrt(eta_arr)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = _ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return -2.0 / (mu_arr ** 3)


__all__ = [
    "IdentityLink", 
    "LogLink", 
    "LogitLink", 
    "InverseLink", 
    "CLogLogLink",
    "SqrtLink",
    "PowerLink",
    "InverseSquareLink",
]
