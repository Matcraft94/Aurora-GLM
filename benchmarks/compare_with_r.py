#!/usr/bin/env python
"""Compare Aurora-GLM results against R glm() function.

This script:
1. Runs the R validation script to fit GLMs in R
2. Fits the same datasets using Aurora-GLM
3. Compares coefficients, deviance, AIC, and fitted values
4. Reports differences and validation status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aurora.models.glm import fit_glm


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing Aurora-GLM against R glm()."""

    family: str
    link: str
    replicate: int
    n_obs: int
    n_features: int
    intercept_abs_diff: float | None
    coef_max_abs_diff: float
    deviance_abs_diff: float
    aic_abs_diff: float
    mean_fitted_abs_diff: float
    aurora_converged: bool
    r_converged: bool
    success: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-script",
        type=Path,
        default=Path(__file__).parent / "run_r_checks.R",
        help="Path to R validation script",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON summary of comparison results",
    )
    parser.add_argument(
        "--coef-tol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for coefficient differences",
    )
    parser.add_argument(
        "--deviance-tol",
        type=float,
        default=0.02,
        help="Absolute tolerance for deviance differences",
    )
    parser.add_argument(
        "--aic-tol",
        type=float,
        default=500.0,
        help="Absolute tolerance for AIC differences (R uses different constant terms)",
    )
    parser.add_argument(
        "--fitted-tol",
        type=float,
        default=2e-4,
        help="Tolerance on mean absolute difference of fitted values",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Check R is available
    try:
        subprocess.run(["R", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: R is not installed or not in PATH", file=sys.stderr)
        return 1

    # Check R script exists
    if not args.r_script.exists():
        print(f"Error: R script not found at {args.r_script}", file=sys.stderr)
        return 1

    # Run R validation script
    print("Running R validation script...")
    r_output_file = Path("/tmp/aurora_r_results.json")
    try:
        result = subprocess.run(
            ["Rscript", str(args.r_script), "--output", str(r_output_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout)
        if result.stderr:
            print("R stderr:", result.stderr, file=sys.stderr)
    except Exception as e:
        print(f"Error running R script: {e}", file=sys.stderr)
        return 1

    # Load R results
    try:
        with open(r_output_file) as f:
            r_results = json.load(f)
    except Exception as e:
        print(f"Error reading R output: {e}", file=sys.stderr)
        return 1

    # Compare each result
    print("\nComparing Aurora-GLM vs R glm()...")
    print(
        f"{'Status':>6} | {'Family':<10} | {'Link':<10} | {'Rep':<3} | "
        f"{'max|Δcoef|':>12} | {'|Δdev|':>10} | {'|ΔAIC|':>10} | {'mean|Δμ|':>12}"
    )
    print("-" * 90)

    comparisons: list[ComparisonResult] = []

    for r_result in r_results:
        if not r_result.get("success", False):
            print(
                f"{'SKIP':>6} | {r_result['family']:<10} | {r_result['link']:<10} | "
                f"{r_result['replicate']:<3} | R fit failed"
            )
            continue

        comparison = _compare_single(
            r_result=r_result,
            coef_tol=args.coef_tol,
            deviance_tol=args.deviance_tol,
            aic_tol=args.aic_tol,
            fitted_tol=args.fitted_tol,
        )
        comparisons.append(comparison)

        status = "OK" if comparison.success else "FAIL"
        print(
            f"{status:>6} | {comparison.family:<10} | {comparison.link:<10} | "
            f"{comparison.replicate:<3} | {comparison.coef_max_abs_diff:>12.3e} | "
            f"{comparison.deviance_abs_diff:>10.3e} | {comparison.aic_abs_diff:>10.3e} | "
            f"{comparison.mean_fitted_abs_diff:>12.3e}"
        )

    # Summary
    print("-" * 90)
    n_total = len(comparisons)
    n_failures = sum(not c.success for c in comparisons)
    print(f"Completed {n_total} comparisons: {n_total - n_failures} passed, {n_failures} failed")

    if comparisons:
        worst_coef = max(c.coef_max_abs_diff for c in comparisons)
        worst_dev = max(c.deviance_abs_diff for c in comparisons)
        worst_aic = max(c.aic_abs_diff for c in comparisons)
        worst_fitted = max(c.mean_fitted_abs_diff for c in comparisons)
        print(
            f"Worst differences: coef={worst_coef:.3e}, dev={worst_dev:.3e}, "
            f"AIC={worst_aic:.3e}, fitted={worst_fitted:.3e}"
        )

    # Write output if requested
    if args.output is not None:
        _write_output(args.output, comparisons)

    return 0 if n_failures == 0 else 2


def _compare_single(
    *,
    r_result: dict[str, Any],
    coef_tol: float,
    deviance_tol: float,
    aic_tol: float,
    fitted_tol: float,
) -> ComparisonResult:
    """Compare a single Aurora-GLM fit against R result."""

    # Extract R results
    family = r_result["family"]
    link = r_result["link"]

    # Reconstruct X matrix (was flattened row-major)
    X_flat = np.array(r_result["X"], dtype=np.float64)
    n_obs = r_result["n_obs"]
    n_features = r_result["n_features"]
    X = X_flat.reshape(n_obs, n_features)

    y = np.array(r_result["y"], dtype=np.float64)

    # Fit with Aurora-GLM
    # Note: R uses "Gamma" with capital G
    aurora_family = family.lower() if family != "Gamma" else "gamma"
    aurora_result = fit_glm(X, y, family=aurora_family, link=link)

    # Extract Aurora results
    aurora_intercept = float(aurora_result.intercept_) if aurora_result.intercept_ is not None else None
    aurora_coef = _to_numpy(aurora_result.coef_)
    aurora_deviance = float(aurora_result.deviance_)
    aurora_aic = float(aurora_result.aic_)
    aurora_fitted = _to_numpy(aurora_result.predict(X, type="response"))

    # Extract R results
    r_intercept = r_result.get("intercept")
    r_coef = np.array(r_result["coefficients"], dtype=np.float64)
    r_deviance = r_result["deviance"]
    r_aic = r_result["aic"]
    r_fitted = np.array(r_result["fitted_values"], dtype=np.float64)

    # Compute differences
    intercept_diff = (
        abs(aurora_intercept - r_intercept) if aurora_intercept is not None and r_intercept is not None else None
    )
    coef_diff = np.abs(aurora_coef - r_coef)
    coef_max_abs = float(np.max(coef_diff)) if coef_diff.size > 0 else 0.0

    deviance_diff = abs(aurora_deviance - r_deviance)
    aic_diff = abs(aurora_aic - r_aic)

    fitted_diff = np.abs(aurora_fitted - r_fitted)
    mean_fitted_diff = float(np.mean(fitted_diff))

    # Check success criteria
    success = (
        coef_max_abs <= coef_tol
        and deviance_diff <= deviance_tol
        and aic_diff <= aic_tol
        and mean_fitted_diff <= fitted_tol
        and aurora_result.converged_
        and r_result.get("converged", False)
    )

    return ComparisonResult(
        family=family,
        link=link,
        replicate=r_result["replicate"],
        n_obs=n_obs,
        n_features=n_features,
        intercept_abs_diff=intercept_diff,
        coef_max_abs_diff=coef_max_abs,
        deviance_abs_diff=deviance_diff,
        aic_abs_diff=aic_diff,
        mean_fitted_abs_diff=mean_fitted_diff,
        aurora_converged=aurora_result.converged_,
        r_converged=r_result.get("converged", False),
        success=success,
    )


def _to_numpy(value: Any) -> np.ndarray:
    """Convert value to NumPy array."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):  # PyTorch tensor
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(value, "cpu") and hasattr(value, "numpy"):  # JAX array
        return value.cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _write_output(path: Path, comparisons: Sequence[ComparisonResult]) -> None:
    """Write comparison results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "comparisons": [
            {
                "family": c.family,
                "link": c.link,
                "replicate": c.replicate,
                "n_obs": c.n_obs,
                "n_features": c.n_features,
                "intercept_abs_diff": c.intercept_abs_diff,
                "coef_max_abs_diff": c.coef_max_abs_diff,
                "deviance_abs_diff": c.deviance_abs_diff,
                "aic_abs_diff": c.aic_abs_diff,
                "mean_fitted_abs_diff": c.mean_fitted_abs_diff,
                "aurora_converged": c.aurora_converged,
                "r_converged": c.r_converged,
                "success": c.success,
            }
            for c in comparisons
        ],
        "summary": {
            "total": len(comparisons),
            "successes": sum(1 for c in comparisons if c.success),
            "failures": sum(1 for c in comparisons if not c.success),
            "max_coef_diff": max((c.coef_max_abs_diff for c in comparisons), default=0.0),
            "max_deviance_diff": max((c.deviance_abs_diff for c in comparisons), default=0.0),
            "max_aic_diff": max((c.aic_abs_diff for c in comparisons), default=0.0),
            "max_fitted_diff": max((c.mean_fitted_abs_diff for c in comparisons), default=0.0),
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote comparison results to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
