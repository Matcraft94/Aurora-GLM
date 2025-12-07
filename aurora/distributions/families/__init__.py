# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Built-in distribution families.

Standard exponential family distributions:
- GaussianFamily: Normal distribution for continuous data
- BinomialFamily: Binomial distribution for binary/proportion data
- PoissonFamily: Poisson distribution for count data
- GammaFamily: Gamma distribution for positive continuous data

Additional distributions (Phase 5 Milestone 1):
- BetaFamily: Beta distribution for proportions in (0, 1)
- InverseGaussianFamily: Inverse Gaussian for positive durations

Heavy-tailed and robust distributions (Phase 5.5):
- StudentTFamily: Student's t for robust regression
- CauchyFamily: Cauchy distribution (t with df=1)
- NegativeBinomialFamily: Negative Binomial for overdispersed counts
- TweedieFamily: Tweedie for zero-inflated continuous data
"""

from .binomial import BinomialFamily
from .gamma import GammaFamily
from .gaussian import GaussianFamily
from .poisson import PoissonFamily

# New distributions (Phase 5 Milestone 1)
from .beta import BetaFamily
from .inverse_gaussian import InverseGaussianFamily, WaldFamily

# Heavy-tailed distributions (Phase 5.5)
from .student_t import StudentTFamily, CauchyFamily
from .negative_binomial import NegativeBinomialFamily, NegBinFamily
from .tweedie import TweedieFamily, CompoundPoissonGammaFamily

__all__ = [
    # Standard exponential family
    "GaussianFamily",
    "BinomialFamily",
    "PoissonFamily",
    "GammaFamily",
    # Additional distributions (Phase 5 Milestone 1)
    "BetaFamily",
    "InverseGaussianFamily",
    "WaldFamily",  # Alias for InverseGaussianFamily
    # Heavy-tailed / Robust distributions
    "StudentTFamily",
    "CauchyFamily",
    "NegativeBinomialFamily",
    "NegBinFamily",  # Alias
    "TweedieFamily",
    "CompoundPoissonGammaFamily",
]
