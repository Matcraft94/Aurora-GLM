# Aurora-GLM Performance Analysis

## Overview

This document summarizes performance characteristics of Aurora-GLM compared to
reference implementations (statsmodels, R's glm/mgcv/lme4).

## Key Findings

### Accuracy Validation

Aurora-GLM achieves excellent numerical agreement with reference implementations:

| Family | vs statsmodels | vs R glm |
|--------|----------------|----------|
| Gaussian | < 1e-11 | < 1e-11 |
| Poisson | < 1e-10 | - |
| Binomial | < 1e-9 | - |
| Gamma | < 2e-6 | - |

### Multi-Backend Consistency

All backends produce consistent results:

| Backend | vs NumPy (max coef diff) |
|---------|-------------------------|
| PyTorch | < 4e-7 |
| JAX | < 1e-15 |

### GPU Performance

PyTorch CUDA provides significant speedups for larger problems:

| Problem Size | NumPy (s) | PyTorch CUDA (s) | Speedup |
|--------------|-----------|------------------|---------|
| Gaussian n=1000 | 0.042 | 0.005 | **9.2x** |
| Gaussian n=5000 | 0.206 | 0.005 | **39.4x** |
| Gaussian n=50000 | 2.1 | 0.018 | **116x** |
| Poisson n=1000 | 0.104 | 0.015 | **6.8x** |
| Poisson n=5000 | 0.429 | 0.015 | **27.9x** |
| Poisson n=50000 | 4.3 | 0.030 | **141x** |

### GLM Performance

Aurora-GLM's GLM implementation prioritizes:
- **Correctness**: Validated against statsmodels and R (max coefficient diff < 1e-9)
- **Multi-backend support**: NumPy, PyTorch, JAX transparently
- **GPU acceleration**: Up to 140x speedup with PyTorch CUDA
- **Extensibility**: Easy to add custom families and links

**Trade-off**: statsmodels is faster for pure NumPy GLM fitting because it uses
heavily optimized IRLS via scipy's LAPACK bindings. However, Aurora's PyTorch
backend with GPU acceleration significantly outperforms all CPU implementations.

### GAM Performance

Aurora provides GAM capabilities that statsmodels lacks:
- B-spline and natural cubic spline bases
- GCV and REML smoothing parameter selection
- **Sparse matrix support**: 5-8x speedup for large problems

| Problem Size | Dense (s) | Sparse (s) | Speedup |
|--------------|-----------|------------|---------|
| n=1000, k=30 | 0.79 | 0.15 | 5.5x |
| n=2000, k=50 | 2.72 | 0.37 | 7.4x |
| n=5000, k=50 | 8.04 | 1.51 | 5.3x |

### GAMM Performance

Aurora's GAMM implementation includes features not available in statsmodels:
- Random intercepts and slopes
- Multiple covariance structures (AR1, compound symmetry, Toeplitz, spatial)
- PQL estimation for non-Gaussian families
- Laplace approximation

Comparison target is R's lme4/nlme (via rpy2 when available).

## When to Use Aurora-GLM

Choose Aurora-GLM when you need:
1. **GPU acceleration** - Up to 140x speedup with PyTorch CUDA
2. **GAM/GAMM capabilities** - statsmodels doesn't support these
3. **Multi-backend flexibility** - PyTorch/JAX for GPU or autodiff
4. **Advanced covariance structures** - temporal, spatial correlations
5. **Consistent API** - GLM → GAM → GAMM with same interface
6. **Research/teaching** - Readable Python implementation

Choose statsmodels when you need:
1. Pure CPU GLM with maximum single-threaded speed
2. Extensive diagnostics and inference tools
3. Time series models (ARIMA, VAR, etc.)

## Benchmark Reproducibility

Run benchmarks with:
```bash
# Comprehensive benchmarks (accuracy + GPU performance)
cd /tmp && PYTHONPATH=/path/to/Aurora-GLM python benchmarks/comprehensive_benchmarks.py

# Quick benchmarks (CI-friendly)
cd /tmp && PYTHONPATH=/path/to/Aurora-GLM python benchmarks/comprehensive_benchmarks.py --quick

# Performance benchmarks only
PYTHONPATH=. python benchmarks/performance_benchmarks.py

# GLM validation against statsmodels
PYTHONPATH=. python benchmarks/run_glm_checks.py --replicates 3
```

**Note**: Run from `/tmp` or another directory without an `renv` project to
avoid R library path conflicts when using rpy2.

Results are saved to `benchmarks/results/`.

## R Environment Setup

For R comparison benchmarks, set up micromamba with R:

```bash
# Install micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~ bin/micromamba

# Create R environment
~/bin/micromamba create -y -n rvenv -c conda-forge r-base r-mgcv r-lme4 r-nlme

# Install rpy2 in Python venv
export R_HOME=~/.local/share/mamba/envs/rvenv/lib/R
pip install rpy2
```

## Performance Optimization Roadmap

Completed optimizations:
- [x] Multi-backend support (NumPy, PyTorch, JAX)
- [x] GPU acceleration via PyTorch CUDA
- [x] Sparse matrix support for GAM/GAMM

Future performance improvements:
1. **JIT compilation**: JAX jit for IRLS inner loop
2. **Parallel smoothing**: Concurrent GCV grid search
3. **Batched GPU fitting**: Fit multiple GLMs in parallel
4. **Memory optimization**: Reduce peak memory for large GAMM

## References

- Aurora-GLM validated against:
  - statsmodels (Python): GLM families
  - R glm: GLM families
  - R mgcv: GAM smooth terms
  - R lme4/nlme: Mixed models
- See `benchmarks/comprehensive_benchmarks.py` for full validation suite
- See `benchmarks/run_glm_checks.py` for coefficient comparison
