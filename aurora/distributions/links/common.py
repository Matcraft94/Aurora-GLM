# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Common link function implementations."""

from __future__ import annotations

import numpy as np

from ..base import LinkFunction
from .._utils import (
    as_namespace_array,
    clip_probability,
    ensure_positive,
    namespace,
    ones_like,
)

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None  # type: ignore[assignment]


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
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return xp.log(mu_arr)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        # Clamp eta to prevent overflow: log(max_float64) ≈ 709
        if xp is np:
            eta_clamped = np.clip(eta_arr, -700, 700)
        elif hasattr(xp, "clamp"):  # PyTorch
            eta_clamped = xp.clamp(eta_arr, -700, 700)
        else:  # JAX or other
            eta_clamped = xp.clip(eta_arr, -700, 700)
        return xp.exp(eta_clamped)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
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
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return 1.0 / mu_arr

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = ensure_positive(as_namespace_array(eta, xp, like=eta), xp)
        return 1.0 / eta_arr

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return -1.0 / (mu_arr**2)


class CLogLogLink(LinkFunction):
    """Complementary log-log link ``g(mu) = log(-log(1 - mu))``."""

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        one_minus = 1.0 - mu_arr
        one_minus = ensure_positive(one_minus, xp)
        return xp.log(-xp.log(one_minus))

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        return 1.0 - xp.exp(-xp.exp(eta_arr))

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)
        one_minus = 1.0 - mu_arr
        one_minus = ensure_positive(one_minus, xp)
        log_term = -xp.log(one_minus)
        return 1.0 / (log_term * one_minus)


class SqrtLink(LinkFunction):
    """Square root link ``g(mu) = sqrt(mu)``.

    Useful for count data where variance is proportional to mean.
    Common alternative to log link for Poisson-like data.
    """

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return xp.sqrt(mu_arr)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)
        return eta_arr**2

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
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
        self.name = f"power{power}"

    def link(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)

        if abs(self.power) < 1e-10:
            # Use log for power ≈ 0
            return xp.log(mu_arr)

        return mu_arr**self.power

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)

        if abs(self.power) < 1e-10:
            # Use exp for power ≈ 0
            return xp.exp(eta_arr)

        # Ensure result is positive
        if self.power > 0:
            eta_arr = ensure_positive(eta_arr, xp)

        return eta_arr ** (1.0 / self.power)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)

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
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return 1.0 / (mu_arr**2)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        xp = namespace(eta)
        eta_arr = ensure_positive(as_namespace_array(eta, xp, like=eta), xp)
        return 1.0 / xp.sqrt(eta_arr)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        xp = namespace(mu)
        mu_arr = ensure_positive(as_namespace_array(mu, xp, like=mu), xp)
        return -2.0 / (mu_arr**3)


class ProbitLink(LinkFunction):
    """Probit link ``g(mu) = Φ^{-1}(mu)``.

    The probit link uses the inverse cumulative distribution function (CDF)
    of the standard normal distribution. Common alternative to logit for
    binomial and beta regression models.

    Properties:
    - Lighter tails than logit
    - Assumes underlying normally distributed latent variable
    - Results similar to logit in practice for μ ∈ [0.2, 0.8]

    Mathematical details:
    - Link: g(μ) = Φ^{-1}(μ) where Φ is the standard normal CDF
    - Inverse: μ = Φ(η)
    - Derivative: dg/dμ = 1/φ(Φ^{-1}(μ)) where φ is the normal PDF

    Examples
    --------
    >>> from aurora.distributions.links import ProbitLink
    >>> link = ProbitLink()
    >>> import numpy as np
    >>> mu = np.array([0.1, 0.5, 0.9])
    >>> eta = link.link(mu)  # Transform to linear predictor
    >>> mu_back = link.inverse(eta)  # Should equal mu
    >>> np.allclose(mu, mu_back)
    True

    Notes
    -----
    The probit link is preferred when:
    - There is a theoretical latent normal process
    - Lighter tails than logit are desired
    - Compatibility with other software using probit (e.g., econometrics)

    Comparison with logit:
    - Both are symmetric around 0.5
    - Logit has heavier tails (more robust to outliers)
    - Probit: π × logit(μ) / √3 is a good approximation

    References
    ----------
    - Bliss, C. I. (1934). "The method of probits." Science, 79, 38-39.
    - McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models.
    """

    def link(self, mu):  # noqa: ANN001 - signature from base class
        """Transform probability to linear predictor: η = Φ^{-1}(μ)."""
        from scipy.stats import norm

        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)

        # Φ^{-1}(μ) - inverse normal CDF
        if xp is np:
            return norm.ppf(mu_arr)
        elif xp is torch:  # type: ignore[comparison-overlap]
            # PyTorch: use scipy and convert
            mu_np = mu_arr.detach().cpu().numpy()
            eta_np = norm.ppf(mu_np)
            return torch.as_tensor(eta_np, dtype=mu_arr.dtype, device=mu_arr.device)
        else:
            # JAX or other: convert through numpy
            import numpy as np_std

            mu_np = np_std.asarray(mu_arr)
            eta_np = norm.ppf(mu_np)
            return xp.asarray(eta_np)

    def inverse(self, eta):  # noqa: ANN001 - signature from base class
        """Transform linear predictor to probability: μ = Φ(η)."""
        from scipy.stats import norm

        xp = namespace(eta)
        eta_arr = as_namespace_array(eta, xp, like=eta)

        # Clamp eta to avoid extreme values
        if xp is np:
            eta_clamped = np.clip(eta_arr, -8, 8)  # norm.cdf(-8) ≈ 6e-16
            return norm.cdf(eta_clamped)
        elif xp is torch:  # type: ignore[comparison-overlap]
            eta_clamped = torch.clamp(eta_arr, -8, 8)
            eta_np = eta_clamped.detach().cpu().numpy()
            mu_np = norm.cdf(eta_np)
            return torch.as_tensor(mu_np, dtype=eta_arr.dtype, device=eta_arr.device)
        else:
            import numpy as np_std

            eta_np = np_std.clip(np_std.asarray(eta_arr), -8, 8)
            mu_np = norm.cdf(eta_np)
            return xp.asarray(mu_np)

    def derivative(self, mu):  # noqa: ANN001 - signature from base class
        """Compute derivative: dg/dμ = 1/φ(Φ^{-1}(μ)).

        The derivative is the reciprocal of the normal PDF evaluated
        at the quantile corresponding to μ.
        """
        from scipy.stats import norm

        xp = namespace(mu)
        mu_arr = clip_probability(as_namespace_array(mu, xp, like=mu), xp)

        if xp is np:
            z = norm.ppf(mu_arr)
            pdf_z = norm.pdf(z)
            # Avoid division by zero at extreme values
            pdf_z = np.clip(pdf_z, 1e-10, None)
            return 1.0 / pdf_z
        elif xp is torch:  # type: ignore[comparison-overlap]
            mu_np = mu_arr.detach().cpu().numpy()
            z = norm.ppf(mu_np)
            pdf_z = np.clip(norm.pdf(z), 1e-10, None)
            deriv_np = 1.0 / pdf_z
            return torch.as_tensor(deriv_np, dtype=mu_arr.dtype, device=mu_arr.device)
        else:
            import numpy as np_std

            mu_np = np_std.asarray(mu_arr)
            z = norm.ppf(mu_np)
            pdf_z = np_std.clip(norm.pdf(z), 1e-10, None)
            deriv_np = 1.0 / pdf_z
            return xp.asarray(deriv_np)


__all__ = [
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "InverseLink",
    "CLogLogLink",
    "SqrtLink",
    "PowerLink",
    "InverseSquareLink",
    "ProbitLink",
]
