# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Hurdle Models for Count Data with Excess Zeros.

Hurdle models are two-part models for count data where:
1. A binary model determines whether Y = 0 or Y > 0
2. A truncated count model determines Y | Y > 0

Unlike zero-inflated models, hurdle models treat ALL zeros as coming from
the binary process, not from the count distribution.

Models
------
- Hurdle Poisson: Binary + truncated Poisson for Y | Y > 0
- Hurdle Negative Binomial: Binary + truncated NB for overdispersed data

Mathematical Framework
----------------------
The hurdle model has density:

    P(Y = 0) = 1 - π
    P(Y = k) = π × f(k) / [1 - f(0)]   for k > 0

where:
- π = P(Y > 0) from binary model (logistic regression)
- f(k) = Poisson(k; λ) or NB(k; μ, θ)
- f(k) / [1 - f(0)] is the zero-truncated density

Properties:
    E[Y] = π × E[Y | Y > 0]
    E[Y | Y > 0] = μ / [1 - f(0)]  (truncated mean)

Hurdle vs Zero-Inflated
-----------------------
- **Hurdle**: All zeros from binary process; positive counts from truncated model
- **Zero-inflated**: Zeros can come from either process (mixture model)

Use hurdle when:
- There's a clear "hurdle" to cross before any positive count occurs
- The process generating zeros is fundamentally different from the count process

Examples: purchasing decisions, species presence/absence

References
----------
.. [1] Mullahy, J. (1986).
       "Specification and testing of some modified count data models."
       Journal of Econometrics, 33(3), 341-365.
.. [2] Cameron, A. C., & Trivedi, P. K. (2013).
       Regression Analysis of Count Data (2nd ed.).
       Cambridge University Press, Chapter 4.
"""

from __future__ import annotations

from .hurdle_poisson import fit_hurdle_poisson, HurdlePoissonResult
from .hurdle_negbin import fit_hurdle_negbin, HurdleNegBinResult
from .truncated import TruncatedPoissonFamily, TruncatedNegBinFamily

__all__ = [
    "fit_hurdle_poisson",
    "HurdlePoissonResult",
    "fit_hurdle_negbin",
    "HurdleNegBinResult",
    "TruncatedPoissonFamily",
    "TruncatedNegBinFamily",
]
