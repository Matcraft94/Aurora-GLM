#!/usr/bin/env python
"""Compare Aurora-GLM results against statsmodels for core GLM families."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - friendly message at runtime
    sm = None  # type: ignore[assignment]

from aurora.models.glm import fit_glm


@dataclass(frozen=True)
class FamilyConfig:
    """Configuration describing a family/link pairing to benchmark."""

    name: str
    link: str


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome for a single comparison run."""

    family: str
    link: str
    replicate: int
    n_obs: int
    n_features: int
    coef_max_abs_diff: float
    intercept_abs_diff: float | None
    deviance_abs_diff: float
    mean_response_abs_diff: float
    aurora_converged: bool
    statsmodels_converged: bool
    success: bool


DEFAULT_CONFIGS: tuple[FamilyConfig, ...] = (
    FamilyConfig("gaussian", "identity"),
    FamilyConfig("poisson", "log"),
    FamilyConfig("binomial", "logit"),
    FamilyConfig("gamma", "log"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=[cfg.name for cfg in DEFAULT_CONFIGS],
        help="Families to benchmark (subset of: gaussian, poisson, binomial, gamma)",
    )
    parser.add_argument("--n-samples", type=int, default=250, help="Number of observations to simulate")
    parser.add_argument("--n-features", type=int, default=4, help="Number of features (excluding intercept)")
    parser.add_argument("--replicates", type=int, default=5, help="Number of random replicates per family")
    parser.add_argument("--seed", type=int, default=12345, help="Seed used for the RNG")
    parser.add_argument(
        "--coef-tol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for coefficient differences",
    )
    parser.add_argument(
        "--deviance-tol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for deviance differences",
    )
    parser.add_argument(
        "--mean-tol",
        type=float,
        default=1e-3,
        help="Tolerance on the mean absolute difference of fitted responses",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a JSON summary of the benchmark results",
    )
    parser.add_argument(
        "--gamma-inverse",
        action="store_true",
        help="Include an additional Gamma comparison using the inverse link (numerically fragile)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if sm is None:
        print("statsmodels is not installed; install it to run the benchmark.", file=sys.stderr)
        return 1

    families = _select_configs(args.families)
    if args.gamma_inverse and all(cfg.link != "inverse" or cfg.name != "gamma" for cfg in families):
        families.append(FamilyConfig("gamma", "inverse"))
    if not families:
        print("No valid families selected.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    all_results: list[ComparisonResult] = []

    for config in families:
        for replicate in range(1, args.replicates + 1):
            seed = int(rng.integers(0, 2**32 - 1))
            result = _run_single_comparison(
                config=config,
                n_obs=args.n_samples,
                n_features=args.n_features,
                seed=seed,
                replicate=replicate,
                coef_tol=args.coef_tol,
                deviance_tol=args.deviance_tol,
                mean_tol=args.mean_tol,
            )
            all_results.append(result)
            _print_row(result)

    _print_summary(all_results)

    if args.output is not None:
        _write_output(args.output, all_results)

    failures = sum(not item.success for item in all_results)
    return 0 if failures == 0 else 2


def _select_configs(names: Iterable[str]) -> list[FamilyConfig]:
    available = {cfg.name: cfg for cfg in DEFAULT_CONFIGS}
    selected: list[FamilyConfig] = []
    for name in names:
        key = name.lower()
        cfg = available.get(key)
        if cfg is not None:
            selected.append(cfg)
        else:
            print(f"Skipping unsupported family: {name}", file=sys.stderr)
    return selected


def _run_single_comparison(
    *,
    config: FamilyConfig,
    n_obs: int,
    n_features: int,
    seed: int,
    replicate: int,
    coef_tol: float,
    deviance_tol: float,
    mean_tol: float,
) -> ComparisonResult:
    rng = np.random.default_rng(seed)
    X, y = _simulate_dataset(config.name, config.link, n_obs, n_features, rng)

    aurora_result = fit_glm(X, y, family=config.name, link=config.link)
    sm_result = _fit_statsmodels(X, y, config)

    aurora_params = _concat_params(aurora_result.intercept_, aurora_result.coef_)
    sm_params = np.asarray(sm_result.params, dtype=np.float64)

    coef_diff = aurora_params - sm_params
    coef_max_abs = float(np.max(np.abs(coef_diff))) if coef_diff.size else 0.0

    intercept_diff = None
    if aurora_result.intercept_ is not None:
        intercept_diff = abs(float(aurora_result.intercept_) - float(sm_params[0]))

    deviance_diff = abs(float(aurora_result.deviance_) - float(sm_result.deviance))

    aurora_mean = _as_numpy(aurora_result.predict(X, type="response"))
    sm_mean = np.asarray(sm_result.predict(), dtype=np.float64)
    mean_abs_diff = float(np.mean(np.abs(aurora_mean - sm_mean)))

    success = (
        coef_max_abs <= coef_tol
        and deviance_diff <= deviance_tol
        and mean_abs_diff <= mean_tol
        and aurora_result.converged_
        and bool(sm_result.converged)
    )

    return ComparisonResult(
        family=config.name,
        link=config.link,
        replicate=replicate,
        n_obs=n_obs,
        n_features=n_features,
        coef_max_abs_diff=coef_max_abs,
        intercept_abs_diff=intercept_diff,
        deviance_abs_diff=deviance_diff,
        mean_response_abs_diff=mean_abs_diff,
        aurora_converged=aurora_result.converged_,
        statsmodels_converged=bool(sm_result.converged),
        success=success,
    )


def _simulate_dataset(
    family: str,
    link: str,
    n_obs: int,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_obs, n_features))
    coef = rng.normal(scale=0.7, size=n_features)
    intercept = rng.normal(scale=0.5)
    linear = intercept + X @ coef

    link_key = link.lower()

    if family == "gaussian":
        mu = linear
        y = mu + rng.normal(scale=0.5, size=n_obs)
    elif family == "poisson":
        mu = np.exp(linear)
        y = rng.poisson(mu)
    elif family == "binomial":
        probs = 1.0 / (1.0 + np.exp(-linear))
        y = rng.binomial(1, probs)
    elif family == "gamma":
        shape = 2.0
        if link_key == "inverse":
            eta = np.exp(np.clip(linear, -3.0, 3.0))
            mu = 1.0 / eta
        else:  # default to log link for stability
            mu = np.exp(np.clip(linear, -3.0, 3.0))
        scale = np.clip(mu / shape, 1e-3, None)
        y = rng.gamma(shape, scale)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported family: {family}")

    return X.astype(np.float64), y.astype(np.float64)


def _fit_statsmodels(X: np.ndarray, y: np.ndarray, config: FamilyConfig):
    X_design = sm.add_constant(X, has_constant="add")
    family = _build_statsmodels_family(config)
    model = sm.GLM(y, X_design, family=family)
    return model.fit(maxiter=100, atol=1e-12)


def _build_statsmodels_family(config: FamilyConfig):
    link = _build_statsmodels_link(config.link)
    name = config.name
    if name == "gaussian":
        return sm.families.Gaussian(link=link)
    if name == "poisson":
        return sm.families.Poisson(link=link)
    if name == "binomial":
        return sm.families.Binomial(link=link)
    if name == "gamma":
        return sm.families.Gamma(link=link)
    raise ValueError(f"Unsupported family: {config.name}")


def _build_statsmodels_link(link_name: str):
    key = link_name.lower()
    if key == "identity":
        return sm.families.links.Identity()
    if key == "log":
        return sm.families.links.Log()
    if key == "logit":
        return sm.families.links.Logit()
    if key == "inverse":
        return sm.families.links.InversePower()
    if key == "cloglog":
        return sm.families.links.CLogLog()
    raise ValueError(f"Unsupported link: {link_name}")


def _concat_params(intercept: float | None, coef) -> np.ndarray:
    coef_arr = _as_numpy(coef)
    if intercept is None:
        return coef_arr
    return np.concatenate(([float(intercept)], coef_arr))


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _print_row(result: ComparisonResult) -> None:
    status = "OK" if result.success else "FAIL"
    intercept_display = f"{result.intercept_abs_diff:.3e}" if result.intercept_abs_diff is not None else "-"
    print(
        f"{status:>4} | {result.family:<8} | {result.link:<8} | rep={result.replicate:02d} | "
        f"max|delta_coef|={result.coef_max_abs_diff:.3e} | delta_dev={result.deviance_abs_diff:.3e} | "
        f"mean|delta_mu|={result.mean_response_abs_diff:.3e} | delta_intercept={intercept_display}",
        flush=True,
    )


def _print_summary(results: Sequence[ComparisonResult]) -> None:
    total = len(results)
    failures = sum(not item.success for item in results)
    if total == 0:
        return
    worst_coef = max((item.coef_max_abs_diff for item in results), default=0.0)
    worst_dev = max((item.deviance_abs_diff for item in results), default=0.0)
    worst_mean = max((item.mean_response_abs_diff for item in results), default=0.0)
    print("-" * 94)
    print(
        f"Completed {total} comparisons: {total - failures} passed, {failures} failed. "
        f"Worst max|delta_coef|={worst_coef:.3e}, worst delta_dev={worst_dev:.3e}, "
        f"worst mean|delta_mu|={worst_mean:.3e}"
    )


def _write_output(path: Path, results: Sequence[ComparisonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [asdict(item) for item in results],
        "summary": {
            "total": len(results),
            "successes": sum(1 for item in results if item.success),
            "failures": sum(1 for item in results if not item.success),
            "max_coef_diff": max((item.coef_max_abs_diff for item in results), default=math.nan),
            "max_deviance_diff": max((item.deviance_abs_diff for item in results), default=math.nan),
            "max_mean_diff": max((item.mean_response_abs_diff for item in results), default=math.nan),
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote benchmark summary to {path}")


if __name__ == "__main__":  # pragma: no mutate - script entry point
    raise SystemExit(main())
