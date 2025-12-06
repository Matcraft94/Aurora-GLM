#!/usr/bin/env python
"""Comprehensive benchmarks for Aurora-GLM.

This script compares:
1. Multi-backend accuracy (NumPy vs PyTorch vs JAX)
2. GPU vs CPU performance
3. Accuracy validation against statsmodels
4. R comparison (when rpy2/R available)

Usage:
    PYTHONPATH=. python benchmarks/comprehensive_benchmarks.py
    PYTHONPATH=. python benchmarks/comprehensive_benchmarks.py --gpu-only
    PYTHONPATH=. python benchmarks/comprehensive_benchmarks.py --accuracy-only
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

# =============================================================================
# Backend Detection
# =============================================================================

HAS_TORCH = False
HAS_TORCH_CUDA = False
try:
    import torch
    HAS_TORCH = True
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None

HAS_JAX = False
HAS_JAX_GPU = False
try:
    import jax
    # Enable float64 for JAX (must be done before any jnp operations)
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    HAS_JAX = True
    HAS_JAX_GPU = any('gpu' in str(d).lower() or 'cuda' in str(d).lower()
                      for d in jax.devices())
except ImportError:
    jax = None
    jnp = None

HAS_STATSMODELS = False
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    sm = None

HAS_RPY2 = False
try:
    import os
    # Configure R environment for micromamba installation
    r_home = os.path.expanduser("~/.local/share/mamba/envs/rvenv/lib/R")
    if os.path.exists(r_home):
        os.environ['R_HOME'] = r_home
        os.environ['R_LIBS'] = os.path.join(r_home, 'library')
        os.environ['RENV_AUTOLOADER_ENABLED'] = 'FALSE'
        ld_lib = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = os.path.join(r_home, 'lib') + ':' + ld_lib

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    # Verify R packages are available
    _stats = importr('stats')
    _mgcv = importr('mgcv')
    _lme4 = importr('lme4')
    HAS_RPY2 = True
except (ImportError, Exception):
    ro = None

# Aurora imports
from aurora.models.glm import fit_glm
from aurora.models.gam import fit_gam
from aurora.distributions.families import (
    GaussianFamily, PoissonFamily, BinomialFamily, GammaFamily
)
from aurora.distributions.links import IdentityLink, LogLink, LogitLink


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class AccuracyResult:
    """Result from accuracy comparison."""
    name: str
    backend: str
    reference: str
    n_samples: int
    coef_max_diff: float
    coef_rmse: float
    deviance_diff: float
    prediction_rmse: float
    passed: bool
    tolerance: float
    time_seconds: float = 0.0
    notes: str = ""


@dataclass
class TimingResult:
    """Result from timing comparison."""
    name: str
    backend: str
    device: str
    n_samples: int
    n_features: int
    time_seconds: float
    memory_mb: float | None = None
    converged: bool = True
    throughput: float = 0.0  # samples/second
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    timestamp: str
    system_info: dict
    accuracy_results: list[dict]
    timing_results: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def convert_types(obj):
            """Convert numpy types to Python native types for JSON."""
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, 'w') as f:
            json.dump(convert_types(self.to_dict()), f, indent=2)


def get_system_info() -> dict:
    """Collect system information."""
    import aurora

    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "aurora_version": getattr(aurora, '__version__', 'unknown'),
        "numpy_version": np.__version__,
        "backends": {
            "numpy": True,
            "torch": HAS_TORCH,
            "torch_cuda": HAS_TORCH_CUDA,
            "jax": HAS_JAX,
            "jax_gpu": HAS_JAX_GPU,
        },
        "references": {
            "statsmodels": sm.__version__ if HAS_STATSMODELS else None,
            "rpy2": HAS_RPY2,
        }
    }

    if HAS_TORCH:
        info["torch_version"] = torch.__version__
        if HAS_TORCH_CUDA:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda

    if HAS_JAX:
        info["jax_version"] = jax.__version__
        info["jax_devices"] = [str(d) for d in jax.devices()]

    return info


# =============================================================================
# Data Generation
# =============================================================================

def generate_glm_data(
    n: int,
    p: int,
    family: str,
    backend: str = 'numpy',
    device: str = 'cpu',
    seed: int = 42
) -> tuple:
    """Generate GLM data for specified backend.

    Note: Returns X WITHOUT intercept column. Aurora's fit_glm
    should use fit_intercept=True (the default) to add intercept.
    """
    rng = np.random.default_rng(seed)

    # Generate in NumPy first (NO intercept - fit_glm adds it)
    X_np = rng.standard_normal((n, p))

    # True coefficients include intercept
    beta = rng.standard_normal(p + 1) * 0.5
    # For data generation, add intercept temporarily
    X_with_const = np.column_stack([np.ones(n), X_np])
    eta = X_with_const @ beta

    if family == 'gaussian':
        y_np = eta + rng.standard_normal(n) * 0.5
    elif family == 'poisson':
        mu = np.exp(np.clip(eta, -10, 10))
        y_np = rng.poisson(mu).astype(float)
    elif family == 'binomial':
        mu = 1 / (1 + np.exp(-eta))
        y_np = rng.binomial(1, mu).astype(float)
    elif family == 'gamma':
        mu = np.exp(np.clip(eta, -5, 5))
        y_np = rng.gamma(2.0, mu / 2.0)
        y_np = np.maximum(y_np, 1e-6)
    else:
        raise ValueError(f"Unknown family: {family}")

    # Convert to backend
    if backend == 'numpy':
        return X_np, y_np, beta
    elif backend == 'torch':
        X = torch.tensor(X_np, dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.float64)
        if device == 'cuda' and HAS_TORCH_CUDA:
            X = X.cuda()
            y = y.cuda()
        return X, y, beta
    elif backend == 'jax':
        X = jnp.array(X_np, dtype=jnp.float64)
        y = jnp.array(y_np, dtype=jnp.float64)
        return X, y, beta
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Multi-Backend Accuracy Tests
# =============================================================================

def compare_backends_accuracy(
    n: int = 500,
    p: int = 5,
    families: list[str] = None,
    seed: int = 42,
    tolerance: float = 1e-6
) -> list[AccuracyResult]:
    """Compare accuracy across backends."""
    if families is None:
        families = ['gaussian', 'poisson', 'binomial']

    results = []

    for family in families:
        print(f"  Testing {family} family across backends...", end=" ", flush=True)

        # Generate reference data in NumPy
        X_np, y_np, true_beta = generate_glm_data(n, p, family, 'numpy', seed=seed)

        # Fit with NumPy (reference)
        start = time.perf_counter()
        result_np = fit_glm(X_np, y_np, family=family)
        time_np = time.perf_counter() - start
        coef_np = result_np.coef_
        mu_np = result_np.mu_

        # Test PyTorch
        if HAS_TORCH:
            X_torch, y_torch, _ = generate_glm_data(n, p, family, 'torch', 'cpu', seed)

            start = time.perf_counter()
            result_torch = fit_glm(X_torch, y_torch, family=family)
            time_torch = time.perf_counter() - start

            # Convert back to numpy for comparison
            coef_torch = result_torch.coef_
            if hasattr(coef_torch, 'cpu'):
                coef_torch = coef_torch.cpu().numpy()
            mu_torch = result_torch.mu_
            if hasattr(mu_torch, 'cpu'):
                mu_torch = mu_torch.cpu().numpy()

            coef_diff = np.abs(coef_np - coef_torch)
            mu_diff = np.abs(mu_np - mu_torch)

            results.append(AccuracyResult(
                name=f"Backend-{family}",
                backend="torch-cpu",
                reference="numpy",
                n_samples=n,
                coef_max_diff=float(np.max(coef_diff)),
                coef_rmse=float(np.sqrt(np.mean(coef_diff**2))),
                deviance_diff=0.0,  # TODO
                prediction_rmse=float(np.sqrt(np.mean(mu_diff**2))),
                passed=np.max(coef_diff) < tolerance,
                tolerance=tolerance,
                time_seconds=time_torch,
            ))

        # Test JAX
        if HAS_JAX:
            # Enable float64 for JAX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_jax, y_jax, _ = generate_glm_data(n, p, family, 'jax', 'cpu', seed)

                start = time.perf_counter()
                result_jax = fit_glm(X_jax, y_jax, family=family)
                time_jax = time.perf_counter() - start

                coef_jax = np.array(result_jax.coef_)
                mu_jax = np.array(result_jax.mu_)

                coef_diff = np.abs(coef_np - coef_jax)
                mu_diff = np.abs(mu_np - mu_jax)

                # JAX uses float32 by default, so relax tolerance
                jax_tol = max(tolerance, 1e-5)

                results.append(AccuracyResult(
                    name=f"Backend-{family}",
                    backend="jax-cpu",
                    reference="numpy",
                    n_samples=n,
                    coef_max_diff=float(np.max(coef_diff)),
                    coef_rmse=float(np.sqrt(np.mean(coef_diff**2))),
                    deviance_diff=0.0,
                    prediction_rmse=float(np.sqrt(np.mean(mu_diff**2))),
                    passed=np.max(coef_diff) < jax_tol,
                    tolerance=jax_tol,
                    time_seconds=time_jax,
                    notes="JAX float32 precision" if np.max(coef_diff) > tolerance else ""
                ))

        print("done")

    return results


# =============================================================================
# Statsmodels Accuracy Comparison
# =============================================================================

def compare_with_statsmodels(
    n: int = 1000,
    p: int = 5,
    families: list[str] = None,
    seed: int = 42,
    tolerance: float = 1e-4
) -> list[AccuracyResult]:
    """Compare Aurora accuracy with statsmodels."""
    if not HAS_STATSMODELS:
        print("  statsmodels not available, skipping...")
        return []

    if families is None:
        families = ['gaussian', 'poisson', 'binomial', 'gamma']

    results = []

    family_map = {
        'gaussian': (sm.families.Gaussian(), 'identity'),
        'poisson': (sm.families.Poisson(), 'log'),
        'binomial': (sm.families.Binomial(), 'logit'),
        'gamma': (sm.families.Gamma(link=sm.families.links.Log()), 'log'),
    }

    for family in families:
        if family not in family_map:
            continue

        print(f"  Comparing {family} with statsmodels...", end=" ", flush=True)

        sm_family, link = family_map[family]
        X, y, true_beta = generate_glm_data(n, p, family, 'numpy', seed=seed)

        # Fit with Aurora (fit_intercept=True by default)
        start = time.perf_counter()
        result_aurora = fit_glm(X, y, family=family, link=link)
        time_aurora = time.perf_counter() - start

        # Fit with statsmodels (need to add constant since X has no intercept)
        X_sm = sm.add_constant(X)
        start = time.perf_counter()
        model_sm = sm.GLM(y, X_sm, family=sm_family)
        result_sm = model_sm.fit(disp=False)
        time_sm = time.perf_counter() - start

        # Compare coefficients (include intercept from Aurora)
        coef_aurora = np.concatenate([[result_aurora.intercept_], result_aurora.coef_])
        coef_sm = result_sm.params

        coef_diff = np.abs(coef_aurora - coef_sm)

        # Compare predictions
        mu_aurora = result_aurora.mu_
        mu_sm = result_sm.fittedvalues
        mu_diff = np.abs(mu_aurora - mu_sm)

        # Compare deviance
        dev_aurora = result_aurora.deviance_ if hasattr(result_aurora, 'deviance_') else 0
        dev_sm = result_sm.deviance
        dev_diff = abs(dev_aurora - dev_sm)

        results.append(AccuracyResult(
            name=f"vs-statsmodels-{family}",
            backend="aurora-numpy",
            reference="statsmodels",
            n_samples=n,
            coef_max_diff=float(np.max(coef_diff)),
            coef_rmse=float(np.sqrt(np.mean(coef_diff**2))),
            deviance_diff=float(dev_diff),
            prediction_rmse=float(np.sqrt(np.mean(mu_diff**2))),
            passed=np.max(coef_diff) < tolerance,
            tolerance=tolerance,
            time_seconds=time_aurora,
            notes=f"Aurora: {time_aurora:.4f}s, SM: {time_sm:.4f}s"
        ))

        print(f"max diff: {np.max(coef_diff):.2e}")

    return results


# =============================================================================
# GPU Performance Benchmarks
# =============================================================================

def benchmark_gpu_performance(
    sample_sizes: list[int] = None,
    n_features: int = 10,
    families: list[str] = None,
    seed: int = 42,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> list[TimingResult]:
    """Benchmark GPU vs CPU performance."""
    if sample_sizes is None:
        sample_sizes = [1000, 5000, 10000, 50000]
    if families is None:
        families = ['gaussian', 'poisson']

    results = []

    for n in sample_sizes:
        for family in families:
            print(f"  GPU benchmark {family} n={n}...", end=" ", flush=True)

            # NumPy CPU baseline
            X_np, y_np, _ = generate_glm_data(n, n_features, family, 'numpy', seed=seed)

            # Warmup
            for _ in range(n_warmup):
                fit_glm(X_np, y_np, family=family)

            # Timed runs
            times_np = []
            for _ in range(n_runs):
                gc.collect()
                start = time.perf_counter()
                result = fit_glm(X_np, y_np, family=family)
                times_np.append(time.perf_counter() - start)

            time_np = np.median(times_np)
            results.append(TimingResult(
                name=f"GPU-{family}",
                backend="numpy",
                device="cpu",
                n_samples=n,
                n_features=n_features,
                time_seconds=time_np,
                throughput=n / time_np,
            ))

            # PyTorch CPU
            if HAS_TORCH:
                X_torch, y_torch, _ = generate_glm_data(n, n_features, family, 'torch', 'cpu', seed)

                for _ in range(n_warmup):
                    fit_glm(X_torch, y_torch, family=family)

                times_torch = []
                for _ in range(n_runs):
                    gc.collect()
                    start = time.perf_counter()
                    result = fit_glm(X_torch, y_torch, family=family)
                    times_torch.append(time.perf_counter() - start)

                time_torch = np.median(times_torch)
                results.append(TimingResult(
                    name=f"GPU-{family}",
                    backend="torch",
                    device="cpu",
                    n_samples=n,
                    n_features=n_features,
                    time_seconds=time_torch,
                    throughput=n / time_torch,
                ))

            # PyTorch CUDA
            if HAS_TORCH_CUDA:
                X_cuda, y_cuda, _ = generate_glm_data(n, n_features, family, 'torch', 'cuda', seed)

                # Warmup with sync
                for _ in range(n_warmup):
                    fit_glm(X_cuda, y_cuda, family=family)
                    torch.cuda.synchronize()

                times_cuda = []
                for _ in range(n_runs):
                    gc.collect()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = fit_glm(X_cuda, y_cuda, family=family)
                    torch.cuda.synchronize()
                    times_cuda.append(time.perf_counter() - start)

                time_cuda = np.median(times_cuda)
                results.append(TimingResult(
                    name=f"GPU-{family}",
                    backend="torch",
                    device="cuda",
                    n_samples=n,
                    n_features=n_features,
                    time_seconds=time_cuda,
                    throughput=n / time_cuda,
                ))

            # JAX CPU
            if HAS_JAX:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_jax, y_jax, _ = generate_glm_data(n, n_features, family, 'jax', 'cpu', seed)

                    for _ in range(n_warmup):
                        fit_glm(X_jax, y_jax, family=family)

                    times_jax = []
                    for _ in range(n_runs):
                        gc.collect()
                        start = time.perf_counter()
                        result = fit_glm(X_jax, y_jax, family=family)
                        # JAX operations are synchronous by default on CPU
                        times_jax.append(time.perf_counter() - start)

                    time_jax = np.median(times_jax)
                    results.append(TimingResult(
                        name=f"GPU-{family}",
                        backend="jax",
                        device="cpu" if not HAS_JAX_GPU else "gpu",
                        n_samples=n,
                        n_features=n_features,
                        time_seconds=time_jax,
                        throughput=n / time_jax,
                    ))

            print("done")

    return results


# =============================================================================
# R Comparison (when available)
# =============================================================================

def compare_with_r(
    n: int = 1000,
    p: int = 5,
    seed: int = 42,
    tolerance: float = 1e-4
) -> list[AccuracyResult]:
    """Compare Aurora with R's glm() and mgcv."""
    if not HAS_RPY2:
        print("  rpy2 not available, skipping R comparison...")
        return []

    results = []

    try:
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import default_converter
        import pandas as pd

        stats = importr('stats')

        # Generate data (X has p columns, no intercept - fit_glm adds it)
        X, y, true_beta = generate_glm_data(n, p, 'gaussian', 'numpy', seed=seed)

        # Fit with Aurora (fit_intercept=True by default)
        result_aurora = fit_glm(X, y, family='gaussian')

        # Create pandas DataFrame for R
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(p)])
        df['y'] = y

        # Use conversion context for pandas + numpy
        converter = default_converter + numpy2ri.converter + pandas2ri.converter
        with converter.context():
            # Convert to R data frame
            df_r = ro.conversion.get_conversion().py2rpy(df)

            # Build formula string
            formula_str = 'y ~ ' + ' + '.join([f'x{i}' for i in range(p)])

            # Fit with R's glm - uses formula with intercept
            r_result = stats.glm(
                ro.Formula(formula_str),
                data=df_r,
                family=stats.gaussian()
            )

            coef_r = np.array(stats.coef(r_result))

        # Aurora returns intercept_ separately, combine for comparison
        coef_aurora = np.concatenate([[result_aurora.intercept_], result_aurora.coef_])

        coef_diff = np.abs(coef_aurora - coef_r)

        print(f"max diff: {np.max(coef_diff):.2e}")

        results.append(AccuracyResult(
            name="vs-R-gaussian",
            backend="aurora-numpy",
            reference="R-glm",
            n_samples=n,
            coef_max_diff=float(np.max(coef_diff)),
            coef_rmse=float(np.sqrt(np.mean(coef_diff**2))),
            deviance_diff=0.0,
            prediction_rmse=0.0,
            passed=np.max(coef_diff) < tolerance,
            tolerance=tolerance,
        ))

    except Exception as e:
        import traceback
        print(f"  R comparison failed: {e}")
        traceback.print_exc()

    return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(report: BenchmarkReport) -> str:
    """Generate markdown report from benchmark results."""
    lines = [
        "# Aurora-GLM Comprehensive Benchmarks",
        "",
        f"Generated: {report.timestamp}",
        "",
        "## System Information",
        "",
        f"- Python: {report.system_info['python_version']}",
        f"- Aurora-GLM: {report.system_info['aurora_version']}",
        f"- NumPy: {report.system_info['numpy_version']}",
        "",
        "### Backend Availability",
        "",
    ]

    backends = report.system_info['backends']
    for backend, available in backends.items():
        status = "✅" if available else "❌"
        lines.append(f"- {backend}: {status}")

    if 'cuda_device' in report.system_info:
        lines.append(f"- GPU: {report.system_info['cuda_device']}")

    lines.append("")

    # Accuracy Results
    if report.accuracy_results:
        lines.extend([
            "## Accuracy Comparison",
            "",
            "### Multi-Backend Consistency",
            "",
            "| Test | Backend | Reference | Max Coef Diff | RMSE | Status |",
            "|------|---------|-----------|---------------|------|--------|",
        ])

        for r in report.accuracy_results:
            status = "✅ PASS" if r.get('passed', False) else "❌ FAIL"
            lines.append(
                f"| {r['name']} | {r['backend']} | {r['reference']} | "
                f"{r['coef_max_diff']:.2e} | {r['coef_rmse']:.2e} | {status} |"
            )

        lines.append("")

    # Timing Results
    if report.timing_results:
        lines.extend([
            "## Performance Benchmarks",
            "",
            "### GPU vs CPU",
            "",
            "| Test | Backend | Device | n | Time (s) | Throughput (samples/s) |",
            "|------|---------|--------|---|----------|------------------------|",
        ])

        for r in report.timing_results:
            lines.append(
                f"| {r['name']} | {r['backend']} | {r['device']} | "
                f"{r['n_samples']} | {r['time_seconds']:.4f} | {r['throughput']:.0f} |"
            )

        lines.append("")

        # Calculate speedups
        lines.extend([
            "### GPU Speedup Summary",
            "",
        ])

        # Group by test name and n_samples
        by_test = {}
        for r in report.timing_results:
            key = (r['name'], r['n_samples'])
            if key not in by_test:
                by_test[key] = {}
            by_test[key][f"{r['backend']}-{r['device']}"] = r['time_seconds']

        for (name, n), times in by_test.items():
            if 'numpy-cpu' in times and 'torch-cuda' in times:
                speedup = times['numpy-cpu'] / times['torch-cuda']
                lines.append(f"- {name} (n={n}): PyTorch CUDA is **{speedup:.1f}x** faster than NumPy")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('benchmarks/results'))
    parser.add_argument('--accuracy-only', action='store_true',
                        help='Run only accuracy tests')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Run only GPU benchmarks')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick subset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Aurora-GLM Comprehensive Benchmarks")
    print("=" * 70)

    system_info = get_system_info()
    print(f"\nAurora-GLM: {system_info['aurora_version']}")
    print(f"Backends: NumPy ✓, PyTorch {'✓' if HAS_TORCH else '✗'}, JAX {'✓' if HAS_JAX else '✗'}")
    print(f"GPU: {'✓ ' + system_info.get('cuda_device', '') if HAS_TORCH_CUDA else '✗'}")
    print()

    accuracy_results = []
    timing_results = []

    # Accuracy tests
    if not args.gpu_only:
        print("Running accuracy comparisons...")

        # Multi-backend
        print("\n1. Multi-backend consistency:")
        accuracy_results.extend(compare_backends_accuracy(
            n=500 if args.quick else 1000,
            p=5,
            seed=args.seed,
        ))

        # statsmodels comparison
        print("\n2. Statsmodels comparison:")
        accuracy_results.extend(compare_with_statsmodels(
            n=500 if args.quick else 1000,
            p=5,
            seed=args.seed,
        ))

        # R comparison
        print("\n3. R comparison:")
        accuracy_results.extend(compare_with_r(
            n=500 if args.quick else 1000,
            p=5,
            seed=args.seed,
        ))

    # GPU benchmarks
    if not args.accuracy_only:
        print("\nRunning GPU performance benchmarks...")

        sample_sizes = [1000, 5000] if args.quick else [1000, 5000, 10000, 50000]
        timing_results.extend(benchmark_gpu_performance(
            sample_sizes=sample_sizes,
            n_features=10,
            families=['gaussian', 'poisson'],
            seed=args.seed,
            n_warmup=1 if args.quick else 2,
            n_runs=3 if args.quick else 5,
        ))

    # Generate report
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        accuracy_results=[asdict(r) for r in accuracy_results],
        timing_results=[asdict(r) for r in timing_results],
    )

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / 'comprehensive_results.json'
    report.save(json_path)
    print(f"\nResults saved to: {json_path}")

    md_path = args.output / 'comprehensive_report.md'
    with open(md_path, 'w') as f:
        f.write(generate_markdown_report(report))
    print(f"Report saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if accuracy_results:
        passed = sum(1 for r in accuracy_results if r.passed)
        total = len(accuracy_results)
        print(f"Accuracy tests: {passed}/{total} passed")

        failed = [r for r in accuracy_results if not r.passed]
        if failed:
            print("Failed tests:")
            for r in failed:
                print(f"  - {r.name} ({r.backend}): max diff = {r.coef_max_diff:.2e}")

    if timing_results:
        # Find best GPU speedup
        by_test = {}
        for r in timing_results:
            key = r.name
            if key not in by_test:
                by_test[key] = {'cpu': [], 'gpu': []}
            if r.device == 'cpu':
                by_test[key]['cpu'].append(r.time_seconds)
            else:
                by_test[key]['gpu'].append(r.time_seconds)

        print("\nGPU Performance:")
        for name, times in by_test.items():
            if times['cpu'] and times['gpu']:
                cpu_avg = np.mean(times['cpu'])
                gpu_avg = np.mean(times['gpu'])
                speedup = cpu_avg / gpu_avg
                print(f"  {name}: GPU is {speedup:.1f}x faster")

    print("\nBenchmarks complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
