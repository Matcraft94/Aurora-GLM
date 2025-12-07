# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Utilities for chunked data loading.

This module provides utilities for loading and iterating over
large datasets in chunks, supporting various file formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "ChunkedDataLoader",
    "ArrayChunker",
    "data_iterator_from_arrays",
]


@dataclass
class ChunkedDataLoader:
    """Load data in chunks from files.

    Supports CSV, NPY, and NPZ formats. For large files,
    loads data incrementally to avoid memory issues.

    Parameters
    ----------
    X_path : str or Path
        Path to design matrix file
    y_path : str or Path
        Path to response vector file
    chunk_size : int
        Number of rows per chunk
    shuffle : bool
        Shuffle row indices before chunking

    Examples
    --------
    >>> loader = ChunkedDataLoader('X.npy', 'y.npy', chunk_size=1000)
    >>> for X_chunk, y_chunk in loader:
    ...     # Process chunk
    ...     pass
    """

    X_path: str | Path
    y_path: str | Path
    chunk_size: int = 10000
    shuffle: bool = False
    seed: int | None = None

    def __post_init__(self):
        self.X_path = Path(self.X_path)
        self.y_path = Path(self.y_path)

    def __iter__(self) -> Iterator[tuple[NDArray, NDArray]]:
        """Iterate over data chunks with true streaming support.

        For .npy files, uses memory mapping to avoid loading entire
        file into memory. For other formats, loads fully then chunks.

        Yields
        ------
        X_chunk : ndarray
            Design matrix chunk
        y_chunk : ndarray
            Response vector chunk
        """
        # Use memory mapping for .npy files to enable true streaming
        X = self._load_array_streaming(self.X_path)
        y = self._load_array_streaming(self.y_path).ravel()

        n = len(y)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            idx = indices[start:end]
            # Use .copy() to materialize memory-mapped chunks
            yield X[idx].copy(), y[idx].copy()

    def _load_array_streaming(self, path: Path) -> NDArray:
        """Load array with streaming support (memory mapping for .npy).

        For .npy files, uses memory mapping to enable true streaming
        without loading the entire file into memory. For other formats,
        falls back to full loading.

        Parameters
        ----------
        path : Path
            Path to array file

        Returns
        -------
        array : ndarray
            Loaded array (memory-mapped if .npy, otherwise in-memory)

        Notes
        -----
        Memory-mapped arrays are read-only and lazily loaded. Only the
        accessed chunks are loaded into RAM. Use .copy() to materialize.
        """
        suffix = path.suffix.lower()

        if suffix == ".npy":
            # Use memory mapping for streaming
            return np.load(path, mmap_mode='r')
        elif suffix == ".npz":
            # NPZ doesn't support mmap, load fully
            data = np.load(path)
            return data[list(data.keys())[0]]
        elif suffix == ".csv":
            # CSV requires full load (could use pandas chunking in future)
            return np.loadtxt(path, delimiter=",", skiprows=1)
        else:
            # Try numpy load with mmap if possible
            try:
                return np.load(path, mmap_mode='r')
            except (ValueError, OSError):
                # Fall back to regular load
                return np.load(path)

    def _load_array(self, path: Path) -> NDArray:
        """Load array from file (legacy method, loads fully into memory).

        This method is kept for compatibility and for cases where
        in-memory arrays are needed. Use _load_array_streaming for
        large files.

        Parameters
        ----------
        path : Path
            Path to array file

        Returns
        -------
        array : ndarray
            Loaded array in memory
        """
        suffix = path.suffix.lower()

        if suffix == ".npy":
            return np.load(path)
        elif suffix == ".npz":
            data = np.load(path)
            # Return first array
            return data[list(data.keys())[0]]
        elif suffix == ".csv":
            return np.loadtxt(path, delimiter=",", skiprows=1)
        else:
            # Try numpy load
            return np.load(path)

    @property
    def n_chunks(self) -> int:
        """Estimate number of chunks."""
        # Load y to get total size
        y = self._load_array(self.y_path)
        return int(np.ceil(len(y) / self.chunk_size))


class ArrayChunker:
    """Split numpy arrays into chunks.

    Parameters
    ----------
    X : ndarray
        Design matrix
    y : ndarray
        Response vector
    chunk_size : int
        Rows per chunk
    shuffle : bool
        Shuffle before chunking
    seed : int, optional
        Random seed for shuffling

    Examples
    --------
    >>> X = np.random.randn(10000, 5)
    >>> y = np.random.randn(10000)
    >>> chunker = ArrayChunker(X, y, chunk_size=1000)
    >>> for X_chunk, y_chunk in chunker:
    ...     # Process chunk
    ...     pass
    """

    def __init__(
        self,
        X: NDArray,
        y: NDArray,
        chunk_size: int = 10000,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        self.X = np.asarray(X)
        self.y = np.asarray(y).ravel()
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have same number of rows")

    def __iter__(self) -> Iterator[tuple[NDArray, NDArray]]:
        """Iterate over chunks."""
        n = len(self.y)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            idx = indices[start:end]
            yield self.X[idx], self.y[idx]

    def __len__(self) -> int:
        """Return number of chunks."""
        return int(np.ceil(len(self.y) / self.chunk_size))

    def get_chunks(self) -> tuple[list[NDArray], list[NDArray]]:
        """Return all chunks as lists.

        Returns
        -------
        X_chunks : list of ndarray
        y_chunks : list of ndarray
        """
        X_chunks = []
        y_chunks = []
        for X_chunk, y_chunk in self:
            X_chunks.append(X_chunk)
            y_chunks.append(y_chunk)
        return X_chunks, y_chunks


def data_iterator_from_arrays(
    X: NDArray,
    y: NDArray,
    batch_size: int = 1000,
    n_epochs: int = 1,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[NDArray, NDArray]]:
    """Create a data iterator from numpy arrays.

    Parameters
    ----------
    X : ndarray
        Design matrix
    y : ndarray
        Response vector
    batch_size : int
        Samples per batch
    n_epochs : int
        Number of passes through data
    shuffle : bool
        Shuffle data each epoch
    seed : int, optional
        Random seed

    Yields
    ------
    X_batch : ndarray
    y_batch : ndarray

    Examples
    --------
    >>> X = np.random.randn(1000, 3)
    >>> y = np.random.randn(1000)
    >>> iterator = data_iterator_from_arrays(X, y, batch_size=100, n_epochs=5)
    >>> for X_batch, y_batch in iterator:
    ...     # Process batch
    ...     pass
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n = len(y)

    rng = np.random.default_rng(seed)

    for epoch in range(n_epochs):
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield X[idx], y[idx]
