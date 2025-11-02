# Aurora-GLM Smoothing Module Design (Draft)

## Scope

This document captures the initial design for the `aurora.smoothing` package, which underpins the GAM functionality scheduled for Phase 3. The goal is to deliver a modular, backend-agnostic toolkit for spline construction, penalisation, and smoothing parameter selection that integrates seamlessly with existing GLM infrastructure.

## Guiding Principles

- **Functional composition**: favour pure functions returning immutable dataclasses over stateful objects.
- **Namespace-awareness**: reuse `aurora.distributions._utils.namespace` utilities to keep NumPy/PyTorch/JAX support consistent.
- **Extensibility**: expose lightweight protocols so downstream users can register custom bases or penalties.
- **Separation of concerns**: decouple basis generation, penalty construction, and smoothing selection to keep experimentation easy.

## Package Layout

```
aurora/smoothing/
├── __init__.py
├── design.md              # ← this file
├── splines/
│   ├── __init__.py
│   ├── basis_registry.py  # Registry for spline basis builders
│   ├── cubic.py           # Natural cubic splines
│   ├── bspline.py         # B-splines (uniform & adaptive knots)
│   ├── pspline.py         # Penalised B-splines (P-splines)
│   ├── thinplate.py       # Thin plate regression splines
│   └── tensor.py          # Tensor-product splines for interactions
├── penalties/
│   ├── __init__.py
│   ├── ridge.py           # Ridge-type penalties
│   ├── difference.py      # Difference penalties (orders 1/2)
│   └── custom.py          # Registration hooks for user-defined penalties
└── selection/
    ├── __init__.py
    ├── gcv.py             # Generalised cross-validation
    ├── reml.py            # Restricted maximum likelihood
    ├── aic.py             # Information criterion-based selection
    └── grid.py            # Generic search over smoothing parameters
```

> **Note**: existing filenames (`bsplines.py`, `psplines.py`) will be consolidated into the structure above. Renames will be handled when implementation begins to avoid churn during planning.

## Core Interfaces

### Spline Basis

```python
from typing import Protocol
from aurora.core.types import ArrayLike

class SplineBasis(Protocol):
    name: str

    def design_matrix(
        self,
        x: ArrayLike,
        *,
        df: int | None = None,
        knots: ArrayLike | None = None,
        constraints: str | None = None,
    ) -> ArrayLike: ...

    def penalty_matrix(
        self,
        x: ArrayLike,
        order: int = 2,
    ) -> ArrayLike: ...
```

Each concrete basis module will expose a factory function returning a callable that conforms to the protocol above. Factories will be registered in `basis_registry.py` to allow string-based selection (e.g. `s(x, bs="tp")`).

### Penalty

```python
class Penalty(Protocol):
    name: str

    def matrix(
        self,
        basis: ArrayLike,
        order: int = 2,
    ) -> ArrayLike: ...
```

Penalties operate on basis design matrices and return symmetric penalty matrices. Default implementations will support ridge and finite-difference penalties.

### Smoothing Selector

```python
class SmoothingSelector(Protocol):
    def select(
        self,
        X: ArrayLike,
        y: ArrayLike,
        family: str,
        penalties: dict[str, ArrayLike],
        initial_lambda: float | dict[str, float] | None = None,
    ) -> dict[str, float]: ...
```

Selectors will wrap optimisation loops (GCV, REML, AIC) and operate on partially fitted models produced by the GLM/GAM solver.

## Integration with GAM

- `aurora.models.gam.fitting.fit_gam` will orchestrate: formula parsing → basis construction → penalty assembly → smoothing selection → IRLS solve.
- The existing `GLMResult` will be extended (or a new `GAMResult` introduced) to store smooth-term metadata, smoothing parameters, and helper methods (`plot_smooth`, `partial_dependence`).
- Design matrices for smooth terms will be cached to support diagnostics and refits without recomputation.

## Incremental Delivery Plan

1. **Foundations (Week 1-2)**
   - Implement cubic spline basis (`cubic.py`) with unit tests covering namespace parity.
   - Provide minimal penalty utilities (ridge, difference) and ensure compatibility with GLM backend.
   - Draft `basis_registry` with registration/lookup semantics.

2. **Penalisation & Selection (Week 3-4)**
   - Deliver P-spline and thin-plate bases.
   - Implement GCV selector (closed-form trace computation) and wire into experimental `fit_gam` prototype.
   - Add benchmark script comparing simple smooth to mgcv baseline.

3. **Formula & Result Layer (Week 5-6)**
   - Introduce lightweight parser (`aurora.models.gam.formula`) supporting `s()` terms and linear predictors.
   - Create `GAMResult` with `plot_smooth` hooks (placeholder implementations using matplotlib/seaborn).
   - Expand documentation with notebook covering a univariate smooth example.

4. **Stabilisation (Week 7-8)**
   - Harden selection methods (REML, AIC), add tensor-product bases, and finalise API docs.
   - Run comprehensive validation vs mgcv/statsmodels GAM equivalents.
   - Prepare release notes for Phase 3 milestone.

## Open Questions / Risks

- **Namespace performance**: review whether repeated conversions for PyTorch/JAX hurt performance in smoothing selectors; consider backend-specific optimisations.
- **Formula parsing dependency**: evaluate reusing `patsy` vs building a minimal parser—integration cost and dependency footprint must be weighed.
- **Visualisation stack**: confirm preferred dependency (matplotlib vs plotnine) and plan for optional installs.
- **Large-scale datasets**: tensor-product splines can inflate design matrices; investigate sparse support early to avoid rewrites.

## Next Steps

- Review this draft with the core team and capture feedback in GitHub issues.
- Once approved, scaffold the directory structure and begin with cubic spline implementation.
- Track progress using the Sprint milestones defined in the Phase 2 backlog to keep alignment across teams.
