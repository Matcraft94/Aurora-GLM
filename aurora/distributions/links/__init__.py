"""Standard link functions.

Link functions map the mean μ to the linear predictor η = Xβ.

Standard links:
- IdentityLink: g(μ) = μ (Gaussian)
- LogLink: g(μ) = log(μ) (Poisson, Gamma, Negative Binomial)
- LogitLink: g(μ) = log(μ/(1-μ)) (Binomial)
- InverseLink: g(μ) = 1/μ (Gamma canonical)
- CLogLogLink: g(μ) = log(-log(1-μ)) (Binomial alternative)
- ProbitLink: g(μ) = Φ^{-1}(μ) (Binomial/Beta alternative)

Additional links (Phase 5.5):
- SqrtLink: g(μ) = √μ (count data alternative)
- PowerLink: g(μ) = μ^p (general power family)
- InverseSquareLink: g(μ) = 1/μ² (Inverse Gaussian canonical)
"""

from .common import (
    CLogLogLink,
    IdentityLink,
    InverseLink,
    InverseSquareLink,
    LogLink,
    LogitLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)

__all__ = [
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "InverseLink",
    "CLogLogLink",
    "ProbitLink",
    "SqrtLink",
    "PowerLink",
    "InverseSquareLink",
]