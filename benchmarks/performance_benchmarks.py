#!/usr/bin/env python
"""Performance benchmarks for Aurora-GLM vs statsmodels/mgcv.

This script measures execution time and memory usage for:
- GLM fitting across different sample sizes and families
- GAM fitting with varying numbers of basis functions
- GAMM fitting with different covariance structures
- Sparse vs dense matrix performance

Results are saved to JSON for reproducibility and can be rendered as markdown.

Usage:
    python benchmarks/performance_benchmarks.py --output benchmarks/results/
    python benchmarks/performance_benchmarks.py --quick  # Fast subset for CI
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Optional imports
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    sm = None
    HAS_STATSMODELS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Aurora imports
from aurora.models.glm import fit_glm
from aurora.models.gam import fit_gam
from aurora.models.gamm import fit_gamm, RandomEffect


@dataclass
class TimingResult:
    """Result from a single timing run."""
    name: str
    library: str
    n_samples: int
    n_features: int
    time_seconds: float
    memory_mb: float | None = None
    converged: bool = True
    error: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    timestamp: str
    python_version: str
    aurora_version: str
    statsmodels_version: str | None
    results: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Aurora-GLM Performance Benchmarks",
            "",
            f"Generated: {self.timestamp}",
            f"Python: {self.python_version}",
            f"Aurora-GLM: {self.aurora_version}",
            f"statsmodels: {self.statsmodels_version or 'not installed'}",
            "",
        ]

        # Group by benchmark type
        glm_results = [r for r in self.results if r.get('name', '').startswith('GLM')]
        gam_results = [r for r in self.results if r.get('name', '').startswith('GAM') and 'GAMM' not in r.get('name', '')]
        gamm_results = [r for r in self.results if r.get('name', '').startswith('GAMM')]
        sparse_results = [r for r in self.results if 'sparse' in r.get('name', '').lower()]

        if glm_results:
            lines.extend(self._format_table("GLM Benchmarks", glm_results))
        if gam_results:
            lines.extend(self._format_table("GAM Benchmarks", gam_results))
        if gamm_results:
            lines.extend(self._format_table("GAMM Benchmarks", gamm_results))
        if sparse_results:
            lines.extend(self._format_table("Sparse Matrix Performance", sparse_results))

        return "\n".join(lines)

    def _format_table(self, title: str, results: list[dict]) -> list[str]:
        """Format results as markdown table."""
        lines = [
            f"## {title}",
            "",
            "| Benchmark | Library | n | p | Time (s) | Memory (MB) | Speedup |",
            "|-----------|---------|---|---|----------|-------------|---------|",
        ]

        # Group by benchmark name to compute speedup
        by_name = {}
        for r in results:
            name = r.get('name', '')
            if name not in by_name:
                by_name[name] = {}
            by_name[name][r.get('library', '')] = r

        for name, libs in by_name.items():
            aurora_time = libs.get('aurora', {}).get('time_seconds', float('inf'))
            for lib, r in libs.items():
                t = r.get('time_seconds', 0)
                mem = r.get('memory_mb')
                mem_str = f"{mem:.1f}" if mem else "N/A"

                if lib == 'aurora':
                    speedup = "baseline"
                else:
                    if t > 0 and aurora_time < float('inf'):
                        speedup = f"{t / aurora_time:.2f}x slower" if aurora_time < t else f"{aurora_time / t:.2f}x faster"
                    else:
                        speedup = "N/A"

                lines.append(
                    f"| {name} | {lib} | {r.get('n_samples', 'N/A')} | "
                    f"{r.get('n_features', 'N/A')} | {t:.4f} | {mem_str} | {speedup} |"
                )

        lines.append("")
        return lines


def get_memory_mb() -> float | None:
    """Get current process memory in MB."""
    if not HAS_PSUTIL:
        return None
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def time_function(func: Callable, *args, **kwargs) -> tuple[float, Any, float | None]:
    """Time a function execution with memory tracking."""
    gc.collect()
    mem_before = get_memory_mb()

    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        mem_after = get_memory_mb()
        mem_used = (mem_after - mem_before) if mem_before and mem_after else None

        return elapsed, result, mem_used
    except Exception as e:
        elapsed = time.perf_counter() - start
        raise


# =============================================================================
# GLM Benchmarks
# =============================================================================

def generate_glm_data(n: int, p: int, family: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for GLM benchmarks."""
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, p))
    X = np.column_stack([np.ones(n), X])  # Add intercept

    beta = rng.standard_normal(p + 1) * 0.5
    eta = X @ beta

    if family == 'gaussian':
        y = eta + rng.standard_normal(n) * 0.5
    elif family == 'poisson':
        mu = np.exp(np.clip(eta, -10, 10))
        y = rng.poisson(mu).astype(float)
    elif family == 'binomial':
        mu = 1 / (1 + np.exp(-eta))
        y = rng.binomial(1, mu).astype(float)
    elif family == 'gamma':
        mu = np.exp(np.clip(eta, -5, 5))
        shape = 2.0
        y = rng.gamma(shape, mu / shape)
        y = np.maximum(y, 1e-6)
    else:
        raise ValueError(f"Unknown family: {family}")

    return X, y


def benchmark_glm_aurora(X: np.ndarray, y: np.ndarray, family: str) -> TimingResult:
    """Benchmark Aurora GLM fitting."""
    n, p = X.shape

    try:
        elapsed, result, mem = time_function(
            fit_glm, X, y, family=family
        )
        return TimingResult(
            name=f"GLM-{family}",
            library="aurora",
            n_samples=n,
            n_features=p,
            time_seconds=elapsed,
            memory_mb=mem,
            converged=result.converged if hasattr(result, 'converged') else True,
        )
    except Exception as e:
        return TimingResult(
            name=f"GLM-{family}",
            library="aurora",
            n_samples=n,
            n_features=p,
            time_seconds=0,
            converged=False,
            error=str(e),
        )


def benchmark_glm_statsmodels(X: np.ndarray, y: np.ndarray, family: str) -> TimingResult | None:
    """Benchmark statsmodels GLM fitting."""
    if not HAS_STATSMODELS:
        return None

    n, p = X.shape

    # Map family names
    family_map = {
        'gaussian': sm.families.Gaussian(),
        'poisson': sm.families.Poisson(),
        'binomial': sm.families.Binomial(),
        'gamma': sm.families.Gamma(link=sm.families.links.Log()),
    }

    if family not in family_map:
        return None

    try:
        def fit_sm():
            model = sm.GLM(y, X, family=family_map[family])
            return model.fit(disp=False)

        elapsed, result, mem = time_function(fit_sm)
        return TimingResult(
            name=f"GLM-{family}",
            library="statsmodels",
            n_samples=n,
            n_features=p,
            time_seconds=elapsed,
            memory_mb=mem,
            converged=result.converged,
        )
    except Exception as e:
        return TimingResult(
            name=f"GLM-{family}",
            library="statsmodels",
            n_samples=n,
            n_features=p,
            time_seconds=0,
            converged=False,
            error=str(e),
        )


def run_glm_benchmarks(
    sample_sizes: list[int],
    n_features: int = 10,
    families: list[str] = None,
    seed: int = 42,
) -> list[TimingResult]:
    """Run GLM benchmarks across sample sizes."""
    if families is None:
        families = ['gaussian', 'poisson', 'binomial', 'gamma']

    results = []

    for n in sample_sizes:
        for family in families:
            print(f"  GLM {family} n={n}...", end=" ", flush=True)

            X, y = generate_glm_data(n, n_features, family, seed)

            # Aurora
            r = benchmark_glm_aurora(X, y, family)
            r.name = f"GLM-{family}-n{n}"
            results.append(r)

            # statsmodels
            r_sm = benchmark_glm_statsmodels(X, y, family)
            if r_sm:
                r_sm.name = f"GLM-{family}-n{n}"
                results.append(r_sm)

            print("done")

    return results


# =============================================================================
# GAM Benchmarks
# =============================================================================

def generate_gam_data(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for GAM benchmarks."""
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 10, n)
    # True function: sin curve with noise
    y = 2 * np.sin(x) + rng.standard_normal(n) * 0.5

    return x, y


def benchmark_gam_aurora(
    x: np.ndarray,
    y: np.ndarray,
    n_basis: int,
    use_sparse: bool = False
) -> TimingResult:
    """Benchmark Aurora GAM fitting."""
    n = len(x)

    try:
        elapsed, result, mem = time_function(
            fit_gam, x, y, n_basis=n_basis, use_sparse=use_sparse
        )

        return TimingResult(
            name=f"GAM-k{n_basis}" + ("-sparse" if use_sparse else "-dense"),
            library="aurora",
            n_samples=n,
            n_features=n_basis,
            time_seconds=elapsed,
            memory_mb=mem,
            converged=True,
            extra={"use_sparse": use_sparse}
        )
    except Exception as e:
        return TimingResult(
            name=f"GAM-k{n_basis}" + ("-sparse" if use_sparse else "-dense"),
            library="aurora",
            n_samples=n,
            n_features=n_basis,
            time_seconds=0,
            converged=False,
            error=str(e),
        )


def run_gam_benchmarks(
    sample_sizes: list[int],
    basis_sizes: list[int] = None,
    seed: int = 42,
) -> list[TimingResult]:
    """Run GAM benchmarks."""
    if basis_sizes is None:
        basis_sizes = [10, 20, 50]

    results = []

    for n in sample_sizes:
        for k in basis_sizes:
            print(f"  GAM n={n} k={k}...", end=" ", flush=True)

            x, y = generate_gam_data(n, seed)

            # Dense
            r = benchmark_gam_aurora(x, y, k, use_sparse=False)
            r.name = f"GAM-n{n}-k{k}-dense"
            results.append(r)

            # Sparse (for larger problems)
            if n >= 500 and k >= 20:
                r_sparse = benchmark_gam_aurora(x, y, k, use_sparse=True)
                r_sparse.name = f"GAM-n{n}-k{k}-sparse"
                results.append(r_sparse)

            print("done")

    return results


# =============================================================================
# GAMM Benchmarks
# =============================================================================

def generate_gamm_data(
    n_groups: int,
    n_per_group: int,
    covariance: str = 'identity',
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for GAMM benchmarks."""
    rng = np.random.default_rng(seed)

    n = n_groups * n_per_group

    # Fixed effects
    X = np.column_stack([
        np.ones(n),
        rng.standard_normal(n)
    ])
    beta = np.array([2.0, 0.5])

    # Random effects
    groups = np.repeat(np.arange(n_groups), n_per_group)
    random_intercepts = rng.standard_normal(n_groups)

    # Response
    y = X @ beta + random_intercepts[groups] + rng.standard_normal(n) * 0.5

    return X, y, groups, random_intercepts


def benchmark_gamm_aurora(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    covariance: str = 'identity'
) -> TimingResult:
    """Benchmark Aurora GAMM fitting."""
    n = len(y)
    n_groups = len(np.unique(groups))

    try:
        re = RandomEffect(grouping='group', covariance=covariance)
        groups_data = {'group': groups}

        elapsed, result, mem = time_function(
            fit_gamm,
            y=y,
            X=X,
            random_effects=[re],
            groups_data=groups_data,
        )

        return TimingResult(
            name=f"GAMM-{covariance}",
            library="aurora",
            n_samples=n,
            n_features=X.shape[1],
            time_seconds=elapsed,
            memory_mb=mem,
            converged=result.converged if hasattr(result, 'converged') else True,
            extra={"n_groups": n_groups, "covariance": covariance}
        )
    except Exception as e:
        return TimingResult(
            name=f"GAMM-{covariance}",
            library="aurora",
            n_samples=n,
            n_features=X.shape[1],
            time_seconds=0,
            converged=False,
            error=str(e),
        )


def run_gamm_benchmarks(
    group_configs: list[tuple[int, int]] = None,
    covariances: list[str] = None,
    seed: int = 42,
) -> list[TimingResult]:
    """Run GAMM benchmarks."""
    if group_configs is None:
        group_configs = [(20, 10), (50, 10), (100, 5)]
    if covariances is None:
        covariances = ['identity', 'diagonal']

    results = []

    for n_groups, n_per_group in group_configs:
        for cov in covariances:
            n = n_groups * n_per_group
            print(f"  GAMM {cov} g={n_groups} n/g={n_per_group}...", end=" ", flush=True)

            X, y, groups, _ = generate_gamm_data(n_groups, n_per_group, cov, seed)

            r = benchmark_gamm_aurora(X, y, groups, cov)
            r.name = f"GAMM-{cov}-n{n}-g{n_groups}"
            results.append(r)

            print("done")

    return results


# =============================================================================
# Sparse vs Dense Benchmarks
# =============================================================================

def run_sparse_benchmarks(seed: int = 42) -> list[TimingResult]:
    """Compare sparse vs dense matrix performance."""
    results = []

    configs = [
        (1000, 30),
        (2000, 50),
        (5000, 50),
    ]

    for n, k in configs:
        print(f"  Sparse vs Dense n={n} k={k}...", end=" ", flush=True)

        x, y = generate_gam_data(n, seed)

        # Dense
        r_dense = benchmark_gam_aurora(x, y, k, use_sparse=False)
        r_dense.name = f"Sparse-comparison-n{n}-k{k}"
        r_dense.extra['mode'] = 'dense'
        results.append(r_dense)

        # Sparse
        r_sparse = benchmark_gam_aurora(x, y, k, use_sparse=True)
        r_sparse.name = f"Sparse-comparison-n{n}-k{k}"
        r_sparse.library = "aurora-sparse"
        r_sparse.extra['mode'] = 'sparse'
        results.append(r_sparse)

        # Compute speedup
        if r_dense.time_seconds > 0 and r_sparse.time_seconds > 0:
            speedup = r_dense.time_seconds / r_sparse.time_seconds
            print(f"sparse {speedup:.1f}x faster")
        else:
            print("done")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('benchmarks/results'),
        help='Output directory for results',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick subset of benchmarks',
    )
    parser.add_argument(
        '--glm-only',
        action='store_true',
        help='Run only GLM benchmarks',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    args = parser.parse_args()

    # Get version info
    import aurora
    aurora_version = getattr(aurora, '__version__', 'unknown')
    sm_version = sm.__version__ if HAS_STATSMODELS else None

    print("=" * 60)
    print("Aurora-GLM Performance Benchmarks")
    print("=" * 60)
    print(f"Aurora-GLM version: {aurora_version}")
    print(f"statsmodels: {sm_version or 'not installed'}")
    print(f"psutil: {'available' if HAS_PSUTIL else 'not installed (memory tracking disabled)'}")
    print()

    all_results = []

    # Configure benchmark sizes
    if args.quick:
        glm_sizes = [500, 1000]
        gam_sizes = [500, 1000]
        basis_sizes = [10, 20]
        gamm_configs = [(20, 10)]
    else:
        glm_sizes = [500, 1000, 5000, 10000]
        gam_sizes = [500, 1000, 5000]
        basis_sizes = [10, 20, 50]
        gamm_configs = [(20, 10), (50, 10), (100, 10)]

    # GLM benchmarks
    print("Running GLM benchmarks...")
    glm_results = run_glm_benchmarks(
        sample_sizes=glm_sizes,
        n_features=10,
        seed=args.seed,
    )
    all_results.extend(glm_results)

    if not args.glm_only:
        # GAM benchmarks
        print("\nRunning GAM benchmarks...")
        gam_results = run_gam_benchmarks(
            sample_sizes=gam_sizes,
            basis_sizes=basis_sizes,
            seed=args.seed,
        )
        all_results.extend(gam_results)

        # GAMM benchmarks
        print("\nRunning GAMM benchmarks...")
        gamm_results = run_gamm_benchmarks(
            group_configs=gamm_configs,
            covariances=['identity', 'diagonal'],
            seed=args.seed,
        )
        all_results.extend(gamm_results)

        # Sparse benchmarks
        if not args.quick:
            print("\nRunning sparse vs dense benchmarks...")
            sparse_results = run_sparse_benchmarks(seed=args.seed)
            all_results.extend(sparse_results)

    # Create suite
    suite = BenchmarkSuite(
        timestamp=datetime.now().isoformat(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        aurora_version=aurora_version,
        statsmodels_version=sm_version,
        results=[asdict(r) for r in all_results],
    )

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / 'performance_results.json'
    suite.save(json_path)
    print(f"\nResults saved to: {json_path}")

    md_path = args.output / 'performance_report.md'
    with open(md_path, 'w') as f:
        f.write(suite.to_markdown())
    print(f"Report saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # GLM summary
    aurora_glm = [r for r in all_results if r.library == 'aurora' and 'GLM' in r.name]
    sm_glm = [r for r in all_results if r.library == 'statsmodels' and 'GLM' in r.name]

    if aurora_glm and sm_glm:
        aurora_total = sum(r.time_seconds for r in aurora_glm)
        sm_total = sum(r.time_seconds for r in sm_glm)
        print(f"GLM total time - Aurora: {aurora_total:.2f}s, statsmodels: {sm_total:.2f}s")
        if sm_total > 0:
            print(f"  Aurora is {sm_total/aurora_total:.1f}x faster overall" if aurora_total < sm_total
                  else f"  statsmodels is {aurora_total/sm_total:.1f}x faster overall")

    # Sparse summary
    sparse_results_only = [r for r in all_results if 'Sparse-comparison' in r.name]
    if sparse_results_only:
        dense = [r for r in sparse_results_only if r.extra.get('mode') == 'dense']
        sparse = [r for r in sparse_results_only if r.extra.get('mode') == 'sparse']
        if dense and sparse:
            dense_total = sum(r.time_seconds for r in dense)
            sparse_total = sum(r.time_seconds for r in sparse)
            if sparse_total > 0:
                print(f"\nSparse speedup: {dense_total/sparse_total:.1f}x average")

    print("\nBenchmarks complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
