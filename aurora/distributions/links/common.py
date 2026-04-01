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
    """Identity link function: g(mu) = mu.

    The simplest link where the linear predictor equals the response directly.
    This is the canonical link for the Gaussian family.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import IdentityLink
    >>> link = IdentityLink()
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> eta = link.link(mu)
    >>> np.allclose(eta, mu)
    True
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu_back, mu)
    True

    See Also
    --------
    LogLink : Log link for positive data.
    GaussianFamily : Default family for identity link.
    """

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
    """Log link function: g(mu) = log(mu).

    Ensures mu > 0. This is the canonical link for the Poisson family
    and is commonly used for any strictly positive response.

    Notes
    -----
    The inverse is g^{-1}(eta) = exp(eta). The linear predictor eta is
    clamped to [-700, 700] to prevent numerical overflow.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import LogLink
    >>> link = LogLink()
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> eta = link.link(mu)
    >>> np.allclose(eta, np.log(mu))
    True
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu_back, mu)
    True

    See Also
    --------
    IdentityLink : Identity link.
    PoissonFamily : Default family for log link.
    GammaFamily : Often used with log link.
    """

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
    """Logit link function: g(mu) = log(mu / (1 - mu)).

    Maps probabilities in (0, 1) to the real line. This is the canonical link
    for the binomial family. The inverse is the logistic sigmoid:
    g^{-1}(eta) = 1 / (1 + exp(-eta)).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import LogitLink
    >>> link = LogitLink()
    >>> mu = np.array([0.2, 0.5, 0.8])
    >>> eta = link.link(mu)
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu, mu_back)
    True

    See Also
    --------
    ProbitLink : Normal CDF inverse link.
    CLogLogLink : Complementary log-log link.
    BinomialFamily : Default family for logit link.
    BetaFamily : Often used with logit link.
    """

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
    """Inverse link function: g(mu) = 1 / mu.

    Ensures mu > 0. This is the canonical link for the Gamma family.
    The inverse is g^{-1}(eta) = 1 / eta.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import InverseLink
    >>> link = InverseLink()
    >>> mu = np.array([0.5, 1.0, 2.0])
    >>> eta = link.link(mu)
    >>> np.allclose(eta, 1.0 / mu)
    True
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu_back, mu)
    True

    See Also
    --------
    InverseSquareLink : Inverse square link for inverse Gaussian.
    LogLink : Log link (alternative for Gamma).
    GammaFamily : Default family for inverse link.
    """

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
    """Complementary log-log link function: g(mu) = log(-log(1 - mu)).

    Maps probabilities in (0, 1) to the real line. Asymmetric link
    that approaches 0 more slowly than 1. Commonly used for
    binomial regression when the probability of success increases
    slowly at low doses but rapidly at high doses.

    The inverse is g^{-1}(eta) = 1 - exp(-exp(eta)).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import CLogLogLink
    >>> link = CLogLogLink()
    >>> mu = np.array([0.2, 0.5, 0.8])
    >>> eta = link.link(mu)
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu, mu_back)
    True

    See Also
    --------
    LogitLink : Symmetric link for binomial.
    ProbitLink : Normal CDF inverse link.
    BinomialFamily : Distribution family using these links.
    """

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
    """Square root link function: g(mu) = sqrt(mu).

    Ensures mu > 0. Useful for count data where variance is proportional
    to the mean. Common alternative to the log link for Poisson-like data.

    The inverse is g^{-1}(eta) = eta^2.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import SqrtLink
    >>> link = SqrtLink()
    >>> mu = np.array([1.0, 4.0, 9.0])
    >>> eta = link.link(mu)
    >>> np.allclose(eta, np.sqrt(mu))
    True
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu_back, mu)
    True

    See Also
    --------
    LogLink : Log link for count data.
    IdentityLink : Identity link.
    PoissonFamily : Common family for count data.
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
    """Power link function: g(mu) = mu^power.

    General power transformation that generalizes several other link
    functions as special cases:

    - power = 1: Identity link
    - power = 0: Log link (limit as power approaches 0)
    - power = -1: Inverse link
    - power = 0.5: Square root link
    - power = -2: Inverse square link

    Parameters
    ----------
    power : float, default=1.0
        Power parameter for the transformation.

    Notes
    -----
    The Box-Cox transformation is a special case when properly normalized.

    For power = 0, this class uses the log link as the limit.

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import PowerLink
    >>> # Square root link via power
    >>> link = PowerLink(power=0.5)
    >>> mu = np.array([1.0, 4.0, 9.0])
    >>> eta = link.link(mu)
    >>> np.allclose(eta, np.sqrt(mu))
    True
    >>> # Inverse link via power
    >>> link = PowerLink(power=-1)
    >>> eta = link.link(mu)
    >>> np.allclose(eta, 1.0 / mu)
    True

    See Also
    --------
    IdentityLink : Power link with power=1.
    LogLink : Power link with power=0 (limit).
    InverseLink : Power link with power=-1.
    InverseSquareLink : Power link with power=-2.
    TweedieFamily : Uses canonical power link.
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
    """Inverse square link function: g(mu) = 1 / mu^2.

    Ensures mu > 0. This is the canonical link for the inverse Gaussian
    distribution. The inverse is g^{-1}(eta) = 1 / sqrt(eta).

    Examples
    --------
    >>> import numpy as np
    >>> from aurora.distributions.links import InverseSquareLink
    >>> link = InverseSquareLink()
    >>> mu = np.array([1.0, 2.0, 3.0])
    >>> eta = link.link(mu)
    >>> np.allclose(eta, 1.0 / mu**2)
    True
    >>> mu_back = link.inverse(eta)
    >>> np.allclose(mu_back, mu)
    True

    See Also
    --------
    InverseLink : Inverse link g(mu) = 1/mu.
    InverseGaussianFamily : Default family for inverse square link.
    PowerLink : General power link (power=-2 is equivalent).
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
