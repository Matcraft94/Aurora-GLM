# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Iteratively Reweighted Least Squares (IRLS) fitting for Generalized Linear Models.

Mathematical Framework
----------------------
A Generalized Linear Model (GLM) relates the mean μ_i = E[Y_i] to predictors
via a link function g(·):

    g(μ_i) = η_i = X_i^T β

where:
    - Y_i follows an exponential family distribution
    - η_i is the linear predictor
    - β is the coefficient vector
    - g(·) is a monotonic, differentiable link function

Exponential Family Representation
----------------------------------
The response Y_i has density/mass function:

    f(y_i; θ_i, φ) = exp{[y_i θ_i - b(θ_i)] / a(φ) + c(y_i, φ)}

where:
    - θ_i is the canonical parameter: μ_i = b'(θ_i)
    - φ is the dispersion parameter
    - Var(Y_i) = a(φ) V(μ_i), where V(μ) = b''(θ) is the variance function

Iteratively Reweighted Least Squares (IRLS)
--------------------------------------------
IRLS is a Newton-Raphson algorithm applied to the log-likelihood score equations.

**Score function**:
    U(β) = ∂ℓ/∂β = X^T W (y - μ)

where W = diag(w_i) with:
    w_i = [g'(μ_i)]^{-2} / V(μ_i)

**Fisher information**:
    I(β) = X^T W X

**Update step** (iteration t):

1. Compute linear predictor: η^(t) = X β^(t)
2. Compute fitted values: μ^(t) = g^{-1}(η^(t))
3. Compute working response:
   z^(t) = η^(t) + (y - μ^(t)) g'(μ^(t))

4. Compute working weights:
   w_i^(t) = [g'(μ_i^(t))]^{-2} / V(μ_i^(t))

5. Update coefficients (weighted least squares):
   β^(t+1) = (X^T W^(t) X)^{-1} X^T W^(t) z^(t)

6. Check convergence on the relative deviance change:
   |D^(t+1) - D^(t)| / (|D^(t)| + 0.1) < tolerance   (R glm convention)

Deviance and Model Fit
-----------------------
**Deviance**: Scaled likelihood ratio test statistic:

    D(y; μ) = 2 [ℓ(y; y) - ℓ(μ; y)]

where ℓ(y; y) is the saturated model log-likelihood.

**Pearson chi-squared statistic**:

    X² = Σ (y_i - μ_i)² / V(μ_i)

**Effective degrees of freedom**: p (number of coefficients)

**Information criteria**:
    - AIC = -2ℓ(β̂) + 2p
    - BIC = -2ℓ(β̂) + p log(n)

Numerical Stability
-------------------
This implementation includes several stability enhancements:

1. **Boundary protection**: Constrain μ to be strictly within valid range
   - Gaussian: no constraint
   - Poisson: μ > ε (default ε = 1e-10)
   - Binomial: ε < μ < 1-ε
   - Gamma: μ > ε

2. **Step halving**: If a full IRLS step increases the deviance (or produces
   a non-finite deviance), the step is halved up to 25 times; a step that
   increases the deviance is never accepted merely because step halving was
   exhausted — in that case iteration stops with ``converged_=False``.

3. **Linear algebra**: the weighted least-squares step is solved via LAPACK
   Cholesky factorization of XᵀWX, with a fallback to pivoted least squares
   (``numpy.linalg.lstsq``) on the weighted design when XᵀWX is not positive
   definite. Rank deficiency is reported through ``GLMResult.rank_`` and a
   ``RuntimeWarning`` instead of being silently absorbed by jitter.

4. **Convergence diagnostics**: condition number of XᵀWX is monitored each
   iteration (``RuntimeWarning`` above κ = 1e8), non-convergence after
   ``max_iter`` iterations warns explicitly, and fitted binomial
   probabilities numerically 0 or 1 trigger a separation warning (R's
   ``glm.fit`` convention).

5. **Prior weights**: deviance, log-likelihood, AIC/BIC and the null
   deviance use the weighted contributions Σ wᵢdᵢ / Σ wᵢℓᵢ (R ``glm``
   convention, McCullagh & Nelder §2.3).

Supported Families and Links
-----------------------------
**Gaussian** (identity, log, inverse):
    - Canonical link: identity
    - Variance: V(μ) = 1

**Poisson** (log, identity, sqrt):
    - Canonical link: log
    - Variance: V(μ) = μ

**Binomial** (logit, probit, cloglog):
    - Canonical link: logit
    - Variance: V(μ) = μ(1 − μ) on the probability scale; the number of
      trials n enters as a prior weight (R convention)

**Gamma** (inverse, identity, log):
    - Canonical link: inverse
    - Variance: V(μ) = μ²

References
----------
**Core theory**:

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.).
  Chapman and Hall/CRC. doi:10.1007/978-1-4899-3242-6

- Nelder, J. A., & Wedderburn, R. W. M. (1972). "Generalized linear models."
  *Journal of the Royal Statistical Society: Series A*, 135(3), 370-384.
  doi:10.2307/2344614

**IRLS algorithm**:

- Green, P. J. (1984). "Iteratively reweighted least squares for maximum
  likelihood estimation, and some robust and resistant alternatives."
  *Journal of the Royal Statistical Society: Series B*, 46(2), 149-192.
  doi:10.1111/j.2517-6161.1984.tb01288.x

**Numerical methods**:

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
  Johns Hopkins University Press.

**Model selection**:

- Akaike, H. (1974). "A new look at the statistical model identification."
  *IEEE Transactions on Automatic Control*, 19(6), 716-723.
  doi:10.1109/TAC.1974.1100705

See Also
--------
aurora.models.gam.fitting : Generalized Additive Models
aurora.models.gamm.fitting : Generalized Additive Mixed Models
aurora.distributions.families : Distribution family implementations
aurora.core.optimization : Optimization algorithms

Notes
-----
For mathematical proofs and derivations, see REFERENCES.md in the repository root.

The IRLS algorithm is equivalent to Fisher scoring when the canonical link is used.
For non-canonical links, IRLS approximates the Hessian with the expected information.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ...core.types import Array
from ...distributions._utils import (
    as_namespace_array,
    namespace,
    namespace_from_backend,
)
from ...distributions.base import Family, LinkFunction
from ...distributions.families import (
    BetaFamily,
    BinomialFamily,
    CauchyFamily,
    CompoundPoissonGammaFamily,
    GammaFamily,
    GaussianFamily,
    InverseGaussianFamily,
    NegativeBinomialFamily,
    PoissonFamily,
    StudentTFamily,
    TweedieFamily,
)
from ...distributions.links import (
    CLogLogLink,
    IdentityLink,
    InverseLink,
    InverseSquareLink,
    LogitLink,
    LogLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)
from ..base.result import GLMResult

_FAMILY_REGISTRY: dict[str, Callable[[], Family]] = {
    "gaussian": GaussianFamily,
    "poisson": PoissonFamily,
    "binomial": BinomialFamily,
    "gamma": GammaFamily,
    "beta": BetaFamily,
    "inverse_gaussian": InverseGaussianFamily,
    "negative_binomial": NegativeBinomialFamily,
    "student_t": StudentTFamily,
    "cauchy": CauchyFamily,
    "tweedie": TweedieFamily,
    "compound_poisson_gamma": CompoundPoissonGammaFamily,
}

_LINK_REGISTRY: dict[str, Callable[[], LinkFunction]] = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "inverse": InverseLink,
    "cloglog": CLogLogLink,
    "probit": ProbitLink,
    "sqrt": SqrtLink,
    "inverse_square": InverseSquareLink,
    "power": PowerLink,
}

# Step-halving and conditioning limits for the production IRLS loop
_MAX_STEP_HALVING = 25
_ILL_CONDITIONED_THRESHOLD = 1e8

# Families with a free dispersion parameter: φ̂ = deviance/(n − rank) is
# estimated after fitting and scales the coefficient covariance (R
# summary.glm convention). Poisson/Binomial keep φ ≡ 1.
_DISPERSION_FAMILIES = (GaussianFamily, GammaFamily, InverseGaussianFamily)

# Families whose deviance/log_likelihood accept and honor a ``weights``
# parameter (Σ wᵢdᵢ / Σ wᵢℓᵢ, R glm convention).
_WEIGHTED_STATS_FAMILIES = (
    GaussianFamily,
    PoissonFamily,
    BinomialFamily,
    GammaFamily,
    InverseGaussianFamily,
)


def fit_glm(
    X: Array,
    y: Array,
    *,
    family: str | Family = "gaussian",
    link: str | LinkFunction | None = None,
    weights: Array | None = None,
    offset: Array | None = None,
    backend: str | None = None,
    device: str | None = None,
    max_iter: int = 25,
    tol: float = 1e-8,
    fit_intercept: bool = True,
) -> GLMResult:
    """Fit a Generalized Linear Model using IRLS.

    Parameters
    ----------
    X : array-like
        Design matrix of shape (n_samples, n_features).
    y : array-like
        Target values of shape (n_samples,).
    family : str or Family
        Distribution family ('gaussian', 'poisson', 'binomial', 'gamma').
    link : str or LinkFunction, optional
        Link function. If None, uses family default.
    weights : array-like, optional
        Prior weights (R ``glm`` convention): each observation contributes
        wᵢ times its unit deviance/log-likelihood. For grouped binomial data,
        pass proportions as ``y`` and trial counts as ``weights``.
    offset : array-like, optional
        Offset term.
    backend : str, optional
        Computational backend: 'numpy', 'torch', or 'jax'.
        If None, infers from input data type.
    device : str, optional
        Device for computation (for torch backend): 'cpu', 'cuda', 'cuda:0', etc.
    max_iter : int
        Maximum number of IRLS iterations.
    tol : float
        Convergence tolerance.
    fit_intercept : bool
        Whether to fit an intercept term.

    Returns
    -------
    GLMResult
        Fitted model results.
    """
    # Determine backend and convert data
    if backend is not None:
        xp, device_obj = namespace_from_backend(backend, device)
        X_arr = as_namespace_array(X, xp, device=device_obj)
        y_arr = as_namespace_array(y, xp, device=device_obj)
        if weights is not None:
            weights = as_namespace_array(weights, xp, device=device_obj)
        if offset is not None:
            offset = as_namespace_array(offset, xp, device=device_obj)
    else:
        xp = namespace(X, y, weights, offset)
        X_arr = as_namespace_array(X, xp)
        y_arr = as_namespace_array(y, xp, like=X_arr)

        weights_arr = None
        if weights is not None:
            weights_arr = as_namespace_array(weights, xp, like=y_arr)

        offset_arr = None
        if offset is not None:
            offset_arr = as_namespace_array(offset, xp, like=y_arr)

    # Handle 1D input for X
    if getattr(X_arr, "ndim", 1) == 1:
        if xp is np:
            X_arr = X_arr.reshape(-1, 1)
        elif hasattr(X_arr, "unsqueeze"):  # PyTorch
            X_arr = X_arr.unsqueeze(-1)
        else:  # JAX or other backends
            X_arr = X_arr.reshape(-1, 1)

    # Ensure y is 1D
    if getattr(y_arr, "ndim", 1) != 1:
        y_arr = y_arr.reshape(-1)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Design matrix and response must share the same number of samples.")

    # Process weights and offset when backend was specified
    if backend is not None:
        weights_arr = weights
        offset_arr = offset
        if weights_arr is not None and getattr(weights_arr, "ndim", 1) != 1:
            weights_arr = weights_arr.reshape(-1)
        if offset_arr is not None and getattr(offset_arr, "ndim", 1) != 1:
            offset_arr = offset_arr.reshape(-1)
    else:
        # Process weights and offset for auto-detected backend
        if weights_arr is not None and getattr(weights_arr, "ndim", 1) != 1:
            weights_arr = weights_arr.reshape(-1)
        if offset_arr is not None and getattr(offset_arr, "ndim", 1) != 1:
            offset_arr = offset_arr.reshape(-1)

    family_obj = _coerce_family(family)
    link_obj = _coerce_link(link, family_obj)

    # Domain validation of the response (clear errors instead of silent
    # clipping inside the families).
    y_domain = np.asarray(_as_numpy(y_arr), dtype=np.float64)
    if isinstance(family_obj, (GammaFamily, InverseGaussianFamily)):
        if np.any(y_domain <= 0.0):
            raise ValueError(
                f"{type(family_obj).__name__} requires strictly positive responses "
                f"(y > 0); got {int(np.sum(y_domain <= 0.0))} non-positive value(s)."
            )
    elif isinstance(family_obj, (PoissonFamily, NegativeBinomialFamily, TweedieFamily)):
        if np.any(y_domain < 0.0):
            raise ValueError(
                f"{type(family_obj).__name__} requires non-negative responses "
                f"(y >= 0); got {int(np.sum(y_domain < 0.0))} negative value(s)."
            )

    # Binomial family works on the probability scale (R convention): the
    # response must be a proportion in [0, 1] and the number of trials is a
    # prior weight. A constant trial count set via BinomialFamily(n=k) is
    # honored by treating it as weights=k when no weights were passed.
    if isinstance(family_obj, BinomialFamily):
        y_check = np.asarray(_as_numpy(y_arr), dtype=np.float64)
        if np.any(y_check < 0.0) or np.any(y_check > 1.0):
            raise ValueError(
                "Binomial family expects a proportion response in [0, 1]. "
                "For grouped data with trial counts n_i, pass "
                "y = successes / trials together with weights = trials."
            )
        if weights_arr is None and family_obj.n_trials != 1.0:
            weights_arr = _ones_column(xp, y_arr.shape[0], like=y_arr).reshape(-1)
            weights_arr = weights_arr * family_obj.n_trials

    if weights_arr is not None and not isinstance(family_obj, _WEIGHTED_STATS_FAMILIES):
        warnings.warn(
            f"Prior weights are applied during fitting but are not propagated to "
            f"deviance, log-likelihood and AIC/BIC for "
            f"{type(family_obj).__name__}; fit statistics describe the unweighted model.",
            RuntimeWarning,
            stacklevel=2,
        )

    X_design = X_arr
    if fit_intercept:
        intercept_column = _ones_column(xp, X_arr.shape[0], like=X_arr)
        X_design = _concat_columns(xp, intercept_column, X_arr)

    beta, eta_total, mu, n_iter, converged, deviance, rank, max_cond = _irls(
        xp,
        X_design,
        y_arr,
        family_obj,
        link_obj,
        weights_arr,
        offset_arr,
        max_iter,
        tol,
    )

    if fit_intercept:
        intercept = _to_python_float(beta[0])
        coef = beta[1:]
    else:
        intercept = None
        coef = beta

    deviance_value = _to_python_float(deviance)
    n_params = X_design.shape[1]
    n_obs = X_design.shape[0]

    if fit_intercept:
        X_null = _ones_column(xp, X_arr.shape[0], like=X_arr)
        _, _, _, _, _, null_dev, _, _ = _irls(
            xp,
            X_null,
            y_arr,
            family_obj,
            link_obj,
            weights_arr,
            offset_arr,
            max_iter,
            tol,
        )
        null_deviance = _to_python_float(null_dev)
    else:
        null_deviance = deviance_value

    # Dispersion estimate (R summary.glm convention): φ̂ = D/(n − rank) for
    # families with a free dispersion parameter; φ ≡ 1 for Poisson/Binomial.
    # The covariance matrix of the coefficients is scaled by φ̂ in
    # GLMResult._compute_inference.
    if isinstance(family_obj, _DISPERSION_FAMILIES):
        df_resid = max(n_obs - rank, 1)
        dispersion = deviance_value / df_resid
    else:
        dispersion = 1.0

    # The reported log-likelihood (and hence AIC/BIC) must use the estimated
    # dispersion for families with a free scale, matching statsmodels'
    # ``res.llf`` convention: shape = 1/φ̂ for Gamma, λ = 1/φ̂ for inverse
    # Gaussian, φ = φ̂ for Tweedie. A dispersion parameter set explicitly to
    # a non-default value at family construction is honored instead.
    log_lik_params: dict[str, Any] = {"weights": weights_arr}
    if isinstance(family_obj, GammaFamily) and family_obj.shape == 1.0:
        log_lik_params["shape"] = 1.0 / dispersion
    elif isinstance(family_obj, InverseGaussianFamily) and family_obj.lambda_ in (
        1.0,
        "estimate",
    ):
        log_lik_params["lambda_"] = 1.0 / dispersion
    elif isinstance(family_obj, TweedieFamily) and family_obj.phi == 1.0:
        df_resid = max(n_obs - rank, 1)
        log_lik_params["phi"] = deviance_value / df_resid
    log_likelihood = _to_python_float(family_obj.log_likelihood(y_arr, mu, **log_lik_params))
    # AIC/BIC count only the mean parameters (statsmodels convention:
    # sm.GLM(...).fit().aic == -2*llf + 2*p). R additionally counts the
    # estimated dispersion for Gaussian-like families (+2 in AIC), so aurora
    # AIC values for Gaussian/Gamma/InverseGaussian differ from R's by 2.
    aic = -2.0 * log_likelihood + 2.0 * n_params
    bic = -2.0 * log_likelihood + math.log(max(n_obs, 1)) * n_params

    # ---- Convergence and conditioning diagnostics -----------------------------
    if not converged:
        warnings.warn(
            f"IRLS failed to converge after {n_iter} iterations (max_iter={max_iter}, tol={tol}).",
            RuntimeWarning,
            stacklevel=2,
        )
    if rank < n_params:
        warnings.warn(
            f"Design matrix is rank deficient (rank {rank} < {n_params} columns); "
            "coefficients are not unique (aliasing). The reported solution is the "
            "minimum-norm least-squares solution.",
            RuntimeWarning,
            stacklevel=2,
        )
    if max_cond > _ILL_CONDITIONED_THRESHOLD:
        warnings.warn(
            f"Ill-conditioned weighted system detected: κ = {max_cond:.2e} > "
            f"{_ILL_CONDITIONED_THRESHOLD:.0e}. Results may be numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )
    if isinstance(family_obj, BinomialFamily):
        mu_check = np.asarray(_as_numpy(mu), dtype=np.float64)
        if np.any(mu_check <= 1e-9) or np.any(mu_check >= 1.0 - 1e-9):
            warnings.warn(
                "glm.fit: fitted probabilities numerically 0 or 1 occurred "
                "(possible complete or quasi-complete separation).",
                RuntimeWarning,
                stacklevel=2,
            )

    result = GLMResult(
        coef_=coef,
        intercept_=intercept,
        family=family_obj,
        link=link_obj,
        mu_=mu,
        eta_=eta_total,
        deviance_=deviance_value,
        null_deviance_=null_deviance,
        log_likelihood_=log_likelihood,
        aic_=aic,
        bic_=bic,
        n_iter_=n_iter,
        converged_=converged,
        dispersion_=dispersion,
        rank_=rank,
        condition_number_=max_cond,
        _X=X_arr,
        _y=y_arr,
        _weights=weights_arr,
        _offset=offset_arr,
        _fit_intercept=fit_intercept,
    )

    return result


def _coerce_family(family: str | Family) -> Family:
    if isinstance(family, str):
        key = family.lower()
        try:
            return _FAMILY_REGISTRY[key]()
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown family: {family!r}") from exc
    if isinstance(family, Family):
        return family
    raise TypeError("family must be a string key or a Family instance")


def _coerce_link(link: str | LinkFunction | None, family: Family) -> LinkFunction:
    if link is None:
        return family.default_link
    if isinstance(link, str):
        key = link.lower()
        try:
            return _LINK_REGISTRY[key]()
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown link: {link!r}") from exc
    if isinstance(link, LinkFunction):
        return link
    raise TypeError("link must be None, a string key, or a LinkFunction instance")


def _irls(
    xp,
    X: Array,
    y: Array,
    family: Family,
    link: LinkFunction,
    weights: Array | None,
    offset: Array | None,
    max_iter: int,
    tol: float,
) -> tuple[Array, Array, Array, int, bool, float, int, float]:
    """Production IRLS loop.

    Returns ``(beta, eta, mu, n_iter, converged, deviance, rank, max_cond)``
    where ``rank`` is the numerically detected rank of the final weighted
    design and ``max_cond`` the largest condition number of XᵀWX seen.
    """
    mu = family.initialize(y)
    mu = as_namespace_array(mu, xp, like=y)
    eta_total = link.link(mu)
    if offset is not None:
        eta_linear = eta_total - offset
    else:
        eta_linear = eta_total

    deviance_prev = _to_python_float(family.deviance(y, mu, weights=weights))
    converged = False
    beta = _zeros_vector(xp, X.shape[1], like=X)
    iteration = 0
    rank = X.shape[1]
    max_cond = 0.0
    first_step = True

    for iteration in range(1, max_iter + 1):  # noqa: B007
        deriv = link.derivative(mu)
        variance = family.variance(mu)
        denom = _clamp_positive(deriv * deriv * variance, xp)
        weight_core = _reciprocal(denom, xp)

        if weights is not None:
            weight_core = weight_core * weights

        sqrt_w = _sqrt(weight_core, xp)
        X_weighted = X * sqrt_w[..., None]

        # Working response on the Xβ scale (offset removed) so that the WLS
        # step solves against X alone.
        working_response = eta_linear + (y - mu) * deriv
        z_weighted = working_response * sqrt_w

        beta_new, rank, cond = _weighted_least_squares(xp, X_weighted, z_weighted)
        beta_new = beta_new.reshape(-1)
        if math.isfinite(cond) and cond > max_cond:
            max_cond = cond
        elif not math.isfinite(cond):
            max_cond = math.inf

        # Step halving: accept only steps that do not increase the deviance;
        # a non-finite deviance is never accepted. The first WLS step is
        # always accepted because the data-driven initialization (e.g.
        # μ⁽⁰⁾ = y for Gaussian, deviance 0) can sit below the deviance of
        # any genuine IRLS step.
        step = 1.0
        accepted = False
        for _halving in range(_MAX_STEP_HALVING + 1):  # noqa: B007
            beta_trial = beta + step * (beta_new - beta)
            eta_linear_trial = _matvec(xp, X, beta_trial)
            eta_trial = eta_linear_trial if offset is None else eta_linear_trial + offset
            mu_trial = link.inverse(eta_trial)
            deviance_trial = _to_python_float(family.deviance(y, mu_trial, weights=weights))
            if math.isfinite(deviance_trial) and (
                first_step or deviance_trial <= deviance_prev + 1e-10 * max(1.0, abs(deviance_prev))
            ):
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # No step along the IRLS direction reduced the deviance: stop and
            # keep the previous (best) iterate instead of accepting a worse
            # fit by step exhaustion. ``converged`` stays False.
            break

        first_step = False
        beta = beta_trial
        eta_linear = eta_linear_trial
        eta_total = eta_trial
        mu = mu_trial

        dev_change = abs(deviance_trial - deviance_prev) / (abs(deviance_prev) + 0.1)
        deviance_prev = deviance_trial
        if dev_change < tol:
            converged = True
            break

    return beta, eta_total, mu, iteration, converged, deviance_prev, rank, max_cond


def _weighted_least_squares(xp, X_weighted: Array, z_weighted: Array) -> tuple[Array, int, float]:
    """Solve the weighted least-squares IRLS subproblem.

    Parameters
    ----------
    xp : module
        Array namespace (numpy, torch, or jax.numpy).
    X_weighted : array (n, p)
        Design matrix pre-multiplied by sqrt(weights).
    z_weighted : array (n,)
        Working response pre-multiplied by sqrt(weights).

    Returns
    -------
    solution : array (p,)
        Weighted least-squares solution.
    rank : int
        Numerically detected rank of the weighted design.
    cond : float
        Condition number of XᵀWX (inf when singular).
    """
    if xp is np:
        X_mat = np.asarray(X_weighted, dtype=np.float64)
        z_vec = np.asarray(z_weighted, dtype=np.float64)
        gram = X_mat.T @ X_mat
        rhs = X_mat.T @ z_vec
        try:
            sv = np.linalg.svd(gram, compute_uv=False)
            cond = float(sv[0] / sv[-1]) if sv[-1] > 0.0 else math.inf
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            cond = math.inf
        # SVD-based rank of the weighted design: Cholesky can succeed on
        # numerically rank-deficient Gram matrices, so the rank must be
        # detected independently (R dqrls-style pivoted tolerance).
        rank = int(np.linalg.matrix_rank(X_mat))
        try:
            # Fast path: LAPACK Cholesky on XᵀWX (positive definite case)
            factor = cho_factor(gram, lower=True)
            solution = cho_solve(factor, rhs)
        except np.linalg.LinAlgError:
            # Rank-deficient or indefinite: pivoted least squares on the
            # weighted design itself (does not square the condition number)
            # with SVD-based rank detection.
            solution, _, rank_lstsq, _ = np.linalg.lstsq(X_mat, z_vec, rcond=None)
            rank = int(rank_lstsq)
        return solution, rank, cond

    # Handle PyTorch contiguous requirement
    if hasattr(X_weighted, "contiguous"):
        X_weighted = X_weighted.contiguous()

    # Transpose - different APIs for PyTorch vs JAX
    if hasattr(X_weighted, "transpose") and callable(X_weighted.transpose):
        # Check if it's PyTorch (transpose takes args) or JAX (.T property)
        try:
            X_t = X_weighted.transpose(-1, -2)
        except TypeError:
            # JAX uses .T for 2D transpose
            X_t = X_weighted.T
    else:
        X_t = X_weighted.T

    rhs = X_t @ z_weighted
    gram = X_t @ X_weighted

    dtype = getattr(gram, "dtype", None)
    device = getattr(gram, "device", None)

    # Create eye matrix
    n_features = gram.shape[-1]
    if hasattr(xp, "eye"):
        if device is not None:
            # PyTorch
            eye = xp.eye(n_features, dtype=dtype, device=device)
        else:
            # JAX or others
            eye = xp.eye(n_features, dtype=dtype)
    else:
        eye = np.eye(n_features)

    # Add ridge regularization - different scalar creation for PyTorch vs JAX
    if hasattr(xp, "tensor"):
        # PyTorch
        tensor_kwargs: dict[str, Any] = {}
        if dtype is not None:
            tensor_kwargs["dtype"] = dtype
        if device is not None:
            tensor_kwargs["device"] = device
        ridge_scalar = xp.tensor(1e-8, **tensor_kwargs)
    else:
        # JAX or others - just use float
        ridge_scalar = 1e-8

    gram = gram + ridge_scalar * eye

    # Reshape rhs to column vector
    rhs_column = rhs.reshape(-1, 1)

    # Solve
    try:
        solution = xp.linalg.solve(gram, rhs_column)
    except Exception:  # pragma: no cover - fallback to pseudoinverse on failure
        pinv = xp.linalg.pinv(gram)
        solution = pinv @ rhs_column

    # Rank/condition diagnostics via NumPy on the (small) Gram matrix
    gram_np = np.asarray(_as_numpy(gram), dtype=np.float64)
    rank = int(np.linalg.matrix_rank(gram_np))
    try:
        sv = np.linalg.svd(gram_np, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0.0 else math.inf
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        cond = math.inf

    return solution.reshape(-1), rank, cond


def _matvec(xp, matrix: Array, vector: Array) -> Array:
    if xp is np:
        matrix_np = np.asarray(matrix, dtype=np.float64)
        vector_np = np.asarray(vector, dtype=np.float64)
        return matrix_np @ vector_np

    matmul = getattr(matrix, "matmul", None)
    if callable(matmul):
        return matmul(vector).reshape(-1)
    return (matrix @ vector).reshape(-1)


def _as_numpy(array: Any) -> np.ndarray:
    """Convert an array from any supported backend to a NumPy array."""
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):  # PyTorch
        return array.detach().cpu().numpy()
    return np.asarray(array)  # JAX and array-likes


def _zeros_vector(xp, length: int, *, like: Array) -> Array:
    dtype = getattr(like, "dtype", None)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    device = getattr(like, "device", None)
    if device is not None:
        kwargs["device"] = device
    shape = (length,)
    return xp.zeros(shape, **kwargs)


def _ones_column(xp, rows: int, *, like: Array) -> Array:
    dtype = getattr(like, "dtype", None)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    device = getattr(like, "device", None)
    if device is not None:
        kwargs["device"] = device
    shape = (rows, 1)
    return xp.ones(shape, **kwargs)


def _concat_columns(xp, left: Array, right: Array) -> Array:
    if xp is np:
        return np.concatenate((left, right), axis=1)
    # PyTorch uses cat with dim, JAX uses concatenate with axis
    if hasattr(xp, "cat"):
        # PyTorch
        return xp.cat((left, right), dim=1)
    else:
        # JAX
        return xp.concatenate((left, right), axis=1)


def _clamp_positive(value: Array, xp, eps: float = 1e-12) -> Array:
    if xp is np:
        return np.clip(value, eps, None)
    # Check if it's PyTorch (has clamp) or JAX (uses clip)
    if hasattr(xp, "clamp"):
        # PyTorch
        tensor = xp.tensor
        tensor_kwargs: dict[str, Any] = {}
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        if dtype is not None:
            tensor_kwargs["dtype"] = dtype
        if device is not None:
            tensor_kwargs["device"] = device
        eps_tensor = tensor(eps, **tensor_kwargs)
        return xp.clamp(value, min=eps_tensor)
    else:
        # JAX - uses clip like NumPy
        return xp.clip(value, eps, None)


def _reciprocal(value: Array, xp) -> Array:
    return 1.0 / value


def _sqrt(value: Array, xp) -> Array:
    if xp is np:
        return np.sqrt(value)
    sqrt = xp.sqrt
    return sqrt(value)


def _to_python_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


__all__ = ["fit_glm"]
