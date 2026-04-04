# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias
#
# """Tests for aurora.models.distributed.chunked module."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from aurora.models.distributed.chunked import (
    ArrayChunker,
    ChunkedDataLoader,
    data_iterator_from_arrays,
)


# ---------------------------------------------------------------------------
# ArrayChunker
# ---------------------------------------------------------------------------


class TestArrayChunker:
    def test_basic_iteration(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        chunker = ArrayChunker(X, y, chunk_size=30)
        chunks = list(chunker)
        assert len(chunks) == 4  # ceil(100/30) = 4
        total_rows = sum(len(c[1]) for c in chunks)
        assert total_rows == 100

    def test_exact_chunk_size(self):
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        chunker = ArrayChunker(X, y, chunk_size=50)
        chunks = list(chunker)
        assert len(chunks) == 2
        for Xc, yc in chunks:
            assert Xc.shape == (50, 3)
            assert yc.shape == (50,)

    def test_shuffle(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = np.arange(100).astype(float)
        chunker = ArrayChunker(X, y, chunk_size=100, shuffle=True, seed=42)
        chunks = list(chunker)
        assert len(chunks) == 1
        # With shuffle, order should differ
        assert not np.array_equal(chunks[0][1], np.arange(100))

    def test_len(self):
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        chunker = ArrayChunker(X, y, chunk_size=30)
        assert len(chunker) == 4

    def test_get_chunks(self):
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        chunker = ArrayChunker(X, y, chunk_size=25)
        X_chunks, y_chunks = chunker.get_chunks()
        assert len(X_chunks) == 2
        assert len(y_chunks) == 2

    def test_mismatch_raises(self):
        X = np.random.randn(100, 3)
        y = np.random.randn(50)
        with pytest.raises(ValueError, match="same number"):
            ArrayChunker(X, y, chunk_size=25)

    def test_single_row_chunks(self):
        X = np.random.randn(5, 2)
        y = np.random.randn(5)
        chunker = ArrayChunker(X, y, chunk_size=1)
        assert len(list(chunker)) == 5

    def test_large_chunk_size(self):
        """Chunk size larger than data yields single chunk."""
        X = np.random.randn(10, 3)
        y = np.random.randn(10)
        chunker = ArrayChunker(X, y, chunk_size=1000)
        chunks = list(chunker)
        assert len(chunks) == 1
        assert chunks[0][0].shape == (10, 3)


# ---------------------------------------------------------------------------
# ChunkedDataLoader
# ---------------------------------------------------------------------------


class TestChunkedDataLoader:
    def test_npy_files(self, tmp_path):
        """Load from .npy files."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        X_path = tmp_path / "X.npy"
        y_path = tmp_path / "y.npy"
        np.save(X_path, X)
        np.save(y_path, y)

        loader = ChunkedDataLoader(X_path, y_path, chunk_size=25)
        chunks = list(loader)
        assert len(chunks) == 2
        # Check data integrity
        X_reconstructed = np.vstack([c[0] for c in chunks])
        np.testing.assert_array_almost_equal(X_reconstructed, X)

    def test_npz_files(self, tmp_path):
        """Load from .npz files."""
        X = np.random.randn(40, 2)
        y = np.random.randn(40)
        X_path = tmp_path / "X.npz"
        y_path = tmp_path / "y.npz"
        np.savez(X_path, X)
        np.savez(y_path, y)

        loader = ChunkedDataLoader(X_path, y_path, chunk_size=20)
        chunks = list(loader)
        assert len(chunks) == 2

    def test_csv_files(self, tmp_path):
        """Load from CSV files."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        X_path = tmp_path / "X.csv"
        y_path = tmp_path / "y.csv"
        np.savetxt(X_path, X, delimiter=",", header="a,b", comments="")
        np.savetxt(y_path, y, delimiter=",", header="val", comments="")

        loader = ChunkedDataLoader(X_path, y_path, chunk_size=2)
        chunks = list(loader)
        assert len(chunks) == 2

    def test_n_chunks(self, tmp_path):
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        X_path = tmp_path / "X.npy"
        y_path = tmp_path / "y.npy"
        np.save(X_path, X)
        np.save(y_path, y)

        loader = ChunkedDataLoader(X_path, y_path, chunk_size=30)
        assert loader.n_chunks == 4

    def test_shuffle(self, tmp_path):
        X = np.arange(50).reshape(-1, 1).astype(float)
        y = np.arange(50).astype(float)
        X_path = tmp_path / "X.npy"
        y_path = tmp_path / "y.npy"
        np.save(X_path, X)
        np.save(y_path, y)

        loader = ChunkedDataLoader(X_path, y_path, chunk_size=50, shuffle=True, seed=42)
        chunks = list(loader)
        assert len(chunks) == 1
        assert not np.array_equal(chunks[0][1], y)


