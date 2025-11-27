#!/usr/bin/env python
"""
Backend Compatibility Verification Script for Aurora-GLM

This script comprehensively tests multi-backend support:
- NumPy (CPU)
- PyTorch (CPU and GPU)
- JAX (CPU and GPU)

Generates a detailed compatibility report.
"""

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

from aurora.models.glm import fit_glm
from aurora.models.gam import fit_gam
from aurora.models import fit_gamm
from aurora.models.gamm import RandomEffect
from aurora.distributions._utils import namespace


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BackendInfo:
    """Information about a backend."""
    name: str
    available: bool
    version: Optional[str] = None
    device: Optional[str] = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    backend: str
    device: str
    success: bool
    time_seconds: float
    error: Optional[str] = None
    coefficients: Optional[np.ndarray] = None


# ============================================================================
# Utility Functions
# ============================================================================

def get_backend_info() -> dict[str, BackendInfo]:
    """Collect information about available backends."""
    info = {}

    # NumPy
    info["numpy"] = BackendInfo(
        name="NumPy",
        available=True,
        version=np.__version__,
        device="cpu"
    )

    # PyTorch
    if HAS_TORCH:
        cuda_available = torch.cuda.is_available()
        info["torch"] = BackendInfo(
            name="PyTorch",
            available=True,
            version=torch.__version__,
            device="cuda" if cuda_available else "cpu",
            cuda_available=cuda_available,
            cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
            cuda_device_name=torch.cuda.get_device_name(0) if cuda_available else None
        )
    else:
        info["torch"] = BackendInfo(name="PyTorch", available=False)

    # JAX
    if HAS_JAX:
        devices = jax.devices()
        default_backend = jax.default_backend()
        info["jax"] = BackendInfo(
            name="JAX",
            available=True,
            version=jax.__version__,
            device=default_backend,
            cuda_available="gpu" in default_backend or "cuda" in str(devices[0]).lower(),
            cuda_device_count=len([d for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower()])
        )
    else:
        info["jax"] = BackendInfo(name="JAX", available=False)

    return info


def to_backend(data: Any, backend: str, device: str = "cpu") -> Any:
    """Convert data to specified backend."""
    if isinstance(data, (list, tuple)):
        return tuple(to_backend(d, backend, device) for d in data)

    if backend == "numpy":
        return np.asarray(data, dtype=np.float64)
    elif backend == "torch":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        tensor = torch.tensor(data, dtype=torch.float64)
        if device == "cuda" and torch.cuda.is_available():
            return tensor.cuda()
        return tensor
    elif backend == "jax":
        if not HAS_JAX:
            raise RuntimeError("JAX not available")
        return jnp.array(data, dtype=jnp.float64)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def to_numpy(data: Any) -> np.ndarray:
    """Convert any backend array to NumPy."""
    if HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif HAS_JAX and isinstance(data, jnp.ndarray):
        return np.array(data)
    return np.asarray(data)


# ============================================================================
# Test Functions
# ============================================================================

def test_glm_poisson(backend: str, device: str = "cpu") -> TestResult:
    """Test Poisson GLM on specified backend."""
    test_name = "GLM Poisson"
    start_time = time.time()

    try:
        # Generate data
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 3)
        eta = X @ np.array([0.5, -0.3, 0.8]) + 0.2
        y = np.random.poisson(np.exp(eta))

        # Convert to backend
        X_b, y_b = to_backend((X, y), backend, device)

        # Fit model
        result = fit_glm(X_b, y_b, family="poisson", link="log", max_iter=100)

        # Extract coefficients
        coef = to_numpy(result.coef_)

        elapsed = time.time() - start_time

        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=result.converged_,
            time_seconds=elapsed,
            coefficients=coef
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=False,
            time_seconds=elapsed,
            error=str(e)
        )


def test_glm_binomial(backend: str, device: str = "cpu") -> TestResult:
    """Test Binomial GLM on specified backend."""
    test_name = "GLM Binomial"
    start_time = time.time()

    try:
        np.random.seed(42)
        n = 250
        X = np.random.randn(n, 4)
        eta = X @ np.array([0.8, -0.6, 0.3, 0.5]) - 0.1
        y = np.random.binomial(1, 1 / (1 + np.exp(-eta)))

        X_b, y_b = to_backend((X, y), backend, device)
        result = fit_glm(X_b, y_b, family="binomial", link="logit", max_iter=100)
        coef = to_numpy(result.coef_)
        elapsed = time.time() - start_time

        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=result.converged_,
            time_seconds=elapsed,
            coefficients=coef
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=False,
            time_seconds=elapsed,
            error=str(e)
        )


def test_glm_gaussian(backend: str, device: str = "cpu") -> TestResult:
    """Test Gaussian GLM on specified backend."""
    test_name = "GLM Gaussian"
    start_time = time.time()

    try:
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        y = X @ np.array([1.5, -0.8]) + 0.5 + np.random.randn(n) * 0.3

        X_b, y_b = to_backend((X, y), backend, device)
        result = fit_glm(X_b, y_b, family="gaussian", link="identity", max_iter=100)
        coef = to_numpy(result.coef_)
        elapsed = time.time() - start_time

        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=result.converged_,
            time_seconds=elapsed,
            coefficients=coef
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=False,
            time_seconds=elapsed,
            error=str(e)
        )


def test_gam_bspline(backend: str, device: str = "cpu") -> TestResult:
    """Test GAM with B-splines on specified backend."""
    test_name = "GAM B-spline"
    start_time = time.time()

    try:
        np.random.seed(42)
        n = 150
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)

        x_b, y_b = to_backend((x, y), backend, device)
        result = fit_gam(x_b, y_b, n_basis=12, basis_type="bspline", lambda_=0.1)
        coef = to_numpy(result.coefficients)  # Fixed: coefficients not coefficients_
        elapsed = time.time() - start_time

        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=True,
            time_seconds=elapsed,
            coefficients=coef
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=False,
            time_seconds=elapsed,
            error=str(e)
        )


def test_gamm_random_intercept(backend: str, device: str = "cpu") -> TestResult:
    """Test GAMM with random intercepts on specified backend."""
    test_name = "GAMM Random Intercept"
    start_time = time.time()

    try:
        np.random.seed(42)
        n_groups, n_per_group = 10, 15
        n = n_groups * n_per_group

        group_id = np.repeat(np.arange(n_groups), n_per_group)
        time_var = np.tile(np.arange(n_per_group), n_groups)
        X = np.column_stack([np.ones(n), time_var])

        b_group = np.random.randn(n_groups) * 0.5
        y = 2.0 + 0.3 * time_var + b_group[group_id] + np.random.randn(n) * 0.2

        X_b, y_b, group_b = to_backend((X, y, group_id), backend, device)

        re = RandomEffect(grouping="group")
        result = fit_gamm(
            y=y_b,
            X=X_b,
            random_effects=[re],
            groups_data={"group": group_b},
            covariance="identity"
        )

        beta = to_numpy(result.beta_parametric)
        elapsed = time.time() - start_time

        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=True,
            time_seconds=elapsed,
            coefficients=beta
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return TestResult(
            test_name=test_name,
            backend=backend,
            device=device,
            success=False,
            time_seconds=elapsed,
            error=str(e)
        )


# ============================================================================
# Report Generation
# ============================================================================

def print_separator(char: str = "=", length: int = 80):
    """Print a separator line."""
    print(char * length)


def print_backend_info(info: dict[str, BackendInfo]):
    """Print backend availability information."""
    print_separator()
    print("BACKEND AVAILABILITY")
    print_separator()
    print()

    for backend_name, backend in info.items():
        print(f"{backend.name}:")
        print(f"  Available: {'YES' if backend.available else 'NO'}")

        if backend.available:
            print(f"  Version: {backend.version}")
            print(f"  Default Device: {backend.device}")

            if backend_name == "torch":
                print(f"  CUDA Available: {'YES' if backend.cuda_available else 'NO'}")
                if backend.cuda_available:
                    print(f"  CUDA Devices: {backend.cuda_device_count}")
                    print(f"  CUDA Device Name: {backend.cuda_device_name}")

            if backend_name == "jax":
                print(f"  GPU Available: {'YES' if backend.cuda_available else 'NO'}")
                if backend.cuda_available:
                    print(f"  GPU Devices: {backend.cuda_device_count}")

        print()


def print_test_results(results: list[TestResult]):
    """Print test results in table format."""
    print_separator()
    print("TEST RESULTS")
    print_separator()
    print()

    # Header
    print(f"{'Test':<25} {'Backend':<10} {'Device':<8} {'Status':<10} {'Time (s)':<12}")
    print_separator("-")

    # Results
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"{result.test_name:<25} {result.backend:<10} {result.device:<8} {status:<10} {result.time_seconds:<12.4f}")

        if not result.success and result.error:
            print(f"  Error: {result.error}")

    print()


def print_numerical_consistency(results: list[TestResult]):
    """Check and print numerical consistency across backends."""
    print_separator()
    print("NUMERICAL CONSISTENCY CHECK")
    print_separator()
    print()

    # Group results by test name
    by_test = {}
    for r in results:
        if r.success and r.coefficients is not None:
            key = r.test_name
            if key not in by_test:
                by_test[key] = []
            by_test[key].append(r)

    # Compare coefficients
    for test_name, test_results in by_test.items():
        print(f"{test_name}:")

        if len(test_results) < 2:
            print("  Not enough backends to compare")
            continue

        # Use first result as reference (usually NumPy)
        ref = test_results[0]
        print(f"  Reference: {ref.backend} ({ref.device})")

        for other in test_results[1:]:
            try:
                max_diff = np.max(np.abs(ref.coefficients - other.coefficients))
                rel_diff = max_diff / (np.max(np.abs(ref.coefficients)) + 1e-10)

                status = "MATCH" if max_diff < 1e-4 else "MISMATCH"
                print(f"  vs {other.backend} ({other.device}): {status}")
                print(f"    Max absolute difference: {max_diff:.2e}")
                print(f"    Max relative difference: {rel_diff:.2e}")

            except Exception as e:
                print(f"  vs {other.backend} ({other.device}): ERROR - {e}")

        print()


def print_summary(results: list[TestResult], info: dict[str, BackendInfo]):
    """Print summary statistics."""
    print_separator()
    print("SUMMARY")
    print_separator()
    print()

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {100 * passed_tests / total_tests:.1f}%")
    print()

    # Backend coverage
    backends_tested = set(r.backend for r in results)
    print(f"Backends Tested: {', '.join(sorted(backends_tested))}")

    # Device coverage
    devices_tested = set(f"{r.backend}-{r.device}" for r in results)
    print(f"Backend-Device Combinations: {len(devices_tested)}")
    print()


# ============================================================================
# Main Execution
# ============================================================================

def run_all_tests() -> list[TestResult]:
    """Run all backend compatibility tests."""
    results = []

    # Get backend info
    info = get_backend_info()

    # Define test suite
    test_functions = [
        test_glm_poisson,
        test_glm_binomial,
        test_glm_gaussian,
        test_gam_bspline,
        test_gamm_random_intercept,
    ]

    # Test configurations
    configs = [
        ("numpy", "cpu"),
    ]

    if HAS_TORCH:
        configs.append(("torch", "cpu"))
        if torch.cuda.is_available():
            configs.append(("torch", "cuda"))

    if HAS_JAX:
        configs.append(("jax", "cpu"))
        # JAX auto-detects GPU

    # Run tests
    for test_func in test_functions:
        for backend, device in configs:
            # Skip CUDA tests if not available
            if device == "cuda" and backend == "torch" and not torch.cuda.is_available():
                continue

            result = test_func(backend, device)
            results.append(result)

    return results, info


def main():
    """Main entry point."""
    print("\n")
    print_separator("=")
    print("Aurora-GLM Backend Compatibility Check")
    print_separator("=")
    print()

    # Run tests
    print("Running tests...")
    results, info = run_all_tests()
    print(f"Completed {len(results)} tests\n")

    # Print reports
    print_backend_info(info)
    print_test_results(results)
    print_numerical_consistency(results)
    print_summary(results, info)

    # Exit code
    all_passed = all(r.success for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
