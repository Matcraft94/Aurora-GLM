"""Optimization algorithms for Aurora-GLM."""
from __future__ import annotations

from .base import Optimizer
from .irls import irls
from .lbfgs import lbfgs
from .newton import newton_raphson, modified_newton
from .result import OptimizationResult
from .sparse_solvers import solve_sparse_penalized_ls

__all__ = [
	"OptimizationResult",
	"Optimizer",
	"newton_raphson",
	"modified_newton",
	"irls",
	"lbfgs",
	"optimize",
	"solve_sparse_penalized_ls",
]


def optimize(
	loss_fn,
	init_params,
	method: str = "lbfgs",
	backend: str = "jax",
	**kwargs,
):
	"""Unified optimization interface."""
	from ..backends import get_backend

	backend_obj = get_backend(backend)

	optimizers = {
		"newton": newton_raphson,
		"newton-raphson": newton_raphson,
		"modified-newton": modified_newton,
		"levenberg-marquardt": modified_newton,
		"irls": irls,
		"lbfgs": lbfgs,
		"l-bfgs": lbfgs,
	}

	key = method.lower()
	if key not in optimizers:
		raise ValueError(f"Unknown optimization method: {method}. Available: {list(optimizers)}")

	optimizer_fn = optimizers[key]
	return optimizer_fn(loss_fn, init_params, backend=backend_obj, **kwargs)