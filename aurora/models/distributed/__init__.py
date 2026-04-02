# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Distributed and large-scale GLM fitting.

This module provides tools for fitting GLMs on large datasets that
may not fit in memory, using:

- Mini-batch SGD for streaming/online learning
- Data-parallel IRLS for distributed sufficient statistics
- Chunked data loading utilities

Examples
--------
>>> from aurora.models.distributed import fit_glm_sgd, fit_glm_parallel
>>>
>>> # Mini-batch SGD for streaming data
>>> def data_stream():
...     for i in range(100):
...         yield load_batch(i)
>>> result = fit_glm_sgd(data_stream(), family='poisson')
>>>
>>> # Data-parallel IRLS for chunked data
>>> X_chunks = np.array_split(X, 10)
>>> y_chunks = np.array_split(y, 10)
>>> result = fit_glm_parallel(X_chunks, y_chunks, family='gaussian')

See Also
--------
aurora.models.glm : Standard GLM fitting
aurora.models.bayes : Bayesian GLM fitting
"""

from .backends import HAS_DASK, HAS_RAY, available_backends
from .chunked import (
    ArrayChunker,
    ChunkedDataLoader,
    data_iterator_from_arrays,
)
from .data_parallel import DataParallelIRLS, ParallelResult, fit_glm_parallel
from .minibatch import SGDResult, fit_glm_sgd
from .optimizers import (
    AdaGradOptimizer,
    AdamOptimizer,
    Optimizer,
    SGDOptimizer,
)

__all__ = [
    # Main API
    "fit_glm_sgd",
    "fit_glm_parallel",
    # Results
    "SGDResult",
    "ParallelResult",
    # Optimizers
    "Optimizer",
    "SGDOptimizer",
    "AdamOptimizer",
    "AdaGradOptimizer",
    # Data utilities
    "ChunkedDataLoader",
    "ArrayChunker",
    "data_iterator_from_arrays",
    # Classes
    "DataParallelIRLS",
    # Backend availability
    "HAS_DASK",
    "HAS_RAY",
    "available_backends",
]
