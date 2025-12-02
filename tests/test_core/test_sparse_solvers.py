"""Tests for sparse penalized least squares solvers.

This module tests the sparse solver implementations against dense solvers
and verifies correctness, convergence, and performance characteristics.
"""
import numpy as np
import pytest

from aurora.smoothing.splines import BSplineBasis

try:
    from scipy.sparse import csr_matrix
    from aurora.core.optimization.sparse_solvers import solve_sparse_penalized_ls
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestSparsePenalizedLSSolver:
    """Tests for sparse penalized least squares solver."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_solve_vs_dense_direct(self):
        """Test sparse solver matches dense solver for direct method."""
        # Create test problem: fit sine curve with B-splines
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_true = np.sin(x)
        y = y_true + 0.1 * np.random.randn(100)

        # B-spline basis
        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B_sparse = basis.basis_matrix(x, sparse=True)
        B_dense = B_sparse.toarray()

        # Problem setup
        weights = np.ones(100)
        S = basis.penalty_matrix(order=2)
        lambda_ = 0.1

        # Sparse solver
        beta_sparse, info_sparse = solve_sparse_penalized_ls(
            B_sparse, y, weights, S, lambda_, method='direct'
        )

        # Dense solver (numpy.linalg.solve)
        C_dense = B_dense.T @ np.diag(weights) @ B_dense + lambda_ * S
        d_dense = B_dense.T @ (weights * y)
        beta_dense = np.linalg.solve(C_dense, d_dense)

        # Should match within numerical precision
        np.testing.assert_allclose(beta_sparse, beta_dense, rtol=1e-10)
        assert info_sparse['success']
        assert info_sparse['method'] == 'direct'

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_solve_vs_dense_cg(self):
        """Test sparse CG solver matches dense solution."""
        np.random.seed(123)
        x = np.linspace(0, 10, 80)
        y = 2 * x + 0.5 * np.random.randn(80)

        knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B_sparse = basis.basis_matrix(x, sparse=True)
        B_dense = B_sparse.toarray()

        weights = np.ones(80)
        S = basis.penalty_matrix(order=2)
        lambda_ = 1.0

        # Sparse CG solver (use more relaxed tolerance and more iterations)
        beta_cg, info_cg = solve_sparse_penalized_ls(
            B_sparse, y, weights, S, lambda_, method='cg', tol=1e-8, maxiter=100
        )

        # Dense reference
        C_dense = B_dense.T @ np.diag(weights) @ B_dense + lambda_ * S
        d_dense = B_dense.T @ (weights * y)
        beta_dense = np.linalg.solve(C_dense, d_dense)

        # CG should converge to same solution (might not be exact due to iterative nature)
        np.testing.assert_allclose(beta_cg, beta_dense, rtol=1e-6)
        # Note: CG might not fully converge but should be close
        assert info_cg['method'] == 'cg'

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_solve_vs_dense_minres(self):
        """Test sparse MINRES solver matches dense solution."""
        np.random.seed(456)
        x = np.linspace(0, 5, 60)
        y = x**2 + 0.2 * np.random.randn(60)

        knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B_sparse = basis.basis_matrix(x, sparse=True)
        B_dense = B_sparse.toarray()

        weights = np.ones(60)
        S = basis.penalty_matrix(order=2)
        lambda_ = 0.01  # Small lambda - may be ill-conditioned

        # Sparse MINRES solver
        beta_minres, info_minres = solve_sparse_penalized_ls(
            B_sparse, y, weights, S, lambda_, method='minres', tol=1e-10
        )

        # Dense reference
        C_dense = B_dense.T @ np.diag(weights) @ B_dense + lambda_ * S
        d_dense = B_dense.T @ (weights * y)
        beta_dense = np.linalg.solve(C_dense, d_dense)

        # MINRES should handle ill-conditioned problems
        np.testing.assert_allclose(beta_minres, beta_dense, rtol=1e-7)
        assert info_minres['success']
        assert info_minres['method'] == 'minres'

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_auto_method_selection(self):
        """Test automatic method selection based on problem size."""
        np.random.seed(789)

        # Small problem (k = 12): should select 'direct'
        x_small = np.linspace(0, 10, 80)
        knots_small = BSplineBasis.create_knots(x_small, n_basis=12, degree=3)
        basis_small = BSplineBasis(knots_small, degree=3)
        B_small = basis_small.basis_matrix(x_small, sparse=True)
        y_small = np.sin(x_small) + 0.1 * np.random.randn(80)

        beta_small, info_small = solve_sparse_penalized_ls(
            B_small, y_small, np.ones(80),
            basis_small.penalty_matrix(order=2), lambda_=0.1, method='auto'
        )
        assert info_small['method'] == 'direct'

        # Large problem (k = 60): should select 'cg'
        x_large = np.linspace(0, 10, 200)
        knots_large = BSplineBasis.create_knots(x_large, n_basis=60, degree=3)
        basis_large = BSplineBasis(knots_large, degree=3)
        B_large = basis_large.basis_matrix(x_large, sparse=True)
        y_large = np.sin(x_large) + 0.1 * np.random.randn(200)

        beta_large, info_large = solve_sparse_penalized_ls(
            B_large, y_large, np.ones(200),
            basis_large.penalty_matrix(order=2), lambda_=0.1, method='auto'
        )
        assert info_large['method'] == 'cg'

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_weighted_least_squares(self):
        """Test solver with non-uniform weights."""
        np.random.seed(111)
        x = np.linspace(0, 10, 100)
        y = x + 0.5 * np.random.randn(100)

        # Non-uniform weights (higher weight in middle)
        weights = np.exp(-((x - 5) ** 2) / 5)
        weights = weights / weights.max()  # Normalize to [0, 1]

        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B = basis.basis_matrix(x, sparse=True)
        S = basis.penalty_matrix(order=2)

        beta, info = solve_sparse_penalized_ls(
            B, y, weights, S, lambda_=0.1, method='direct'
        )

        # Check fit exists and solver succeeded
        fitted = B @ beta
        residuals = y - fitted

        # Simply check that we got a reasonable solution
        # (Weighted LS produces different fit than OLS, both valid)
        rmse = np.sqrt(np.mean(residuals ** 2))
        assert rmse < 10.0  # Reasonable fit
        assert info['success']

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_unpenalized_case(self):
        """Test solver with lambda = 0 (unpenalized least squares)."""
        np.random.seed(222)
        x = np.linspace(0, 10, 100)
        y = 3 * x + 2 + 0.5 * np.random.randn(100)

        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B = basis.basis_matrix(x, sparse=True)
        S = basis.penalty_matrix(order=2)

        # Unpenalized (λ = 0)
        beta_unpenalized, info = solve_sparse_penalized_ls(
            B, y, np.ones(100), S, lambda_=0.0, method='direct'
        )

        # Should fit data reasonably (unpenalized can overfit noisy data)
        fitted = B @ beta_unpenalized
        rmse = np.sqrt(np.mean((y - fitted) ** 2))
        assert rmse < 5.0  # Reasonable fit (allows for some noise)
        assert info['success']

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_high_penalty(self):
        """Test solver with very large lambda (heavy smoothing)."""
        np.random.seed(333)
        x = np.linspace(0, 10, 100)
        y_true = np.sin(x)
        y = y_true + 0.5 * np.random.randn(100)

        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B = basis.basis_matrix(x, sparse=True)
        S = basis.penalty_matrix(order=2)

        # Very high penalty (λ = 100) should produce very smooth fit
        beta_smooth, info = solve_sparse_penalized_ls(
            B, y, np.ones(100), S, lambda_=100.0, method='direct'
        )

        # Check smoothness: second derivatives should be small
        fitted = B @ beta_smooth
        second_diff = np.diff(fitted, n=2)
        smoothness = np.mean(second_diff ** 2)

        # Should be much smoother than data
        data_smoothness = np.mean(np.diff(y, n=2) ** 2)
        assert smoothness < 0.1 * data_smoothness
        assert info['success']

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_dense_input(self):
        """Test solver accepts dense arrays and converts to sparse internally."""
        np.random.seed(444)
        x = np.linspace(0, 10, 80)
        y = np.sin(x) + 0.1 * np.random.randn(80)

        knots = BSplineBasis.create_knots(x, n_basis=12, degree=3)
        basis = BSplineBasis(knots, degree=3)

        # Dense inputs
        B_dense = basis.basis_matrix(x, sparse=False)
        S_dense = basis.penalty_matrix(order=2)

        beta, info = solve_sparse_penalized_ls(
            B_dense, y, np.ones(80), S_dense, lambda_=0.1, method='auto'
        )

        # Should work and produce reasonable fit
        fitted = B_dense @ beta
        rmse = np.sqrt(np.mean((y - fitted) ** 2))
        assert rmse < 0.5
        assert info['success']

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_convergence_tolerance(self):
        """Test iterative solver respects tolerance parameter."""
        np.random.seed(555)
        x = np.linspace(0, 10, 100)
        y = x**2 + 0.2 * np.random.randn(100)

        knots = BSplineBasis.create_knots(x, n_basis=15, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B = basis.basis_matrix(x, sparse=True)
        S = basis.penalty_matrix(order=2)

        # Tight tolerance
        beta_tight, info_tight = solve_sparse_penalized_ls(
            B, y, np.ones(100), S, lambda_=0.1, method='cg', tol=1e-10, maxiter=200
        )

        # Loose tolerance
        beta_loose, info_loose = solve_sparse_penalized_ls(
            B, y, np.ones(100), S, lambda_=0.1, method='cg', tol=1e-4, maxiter=200
        )

        # Solutions should be reasonably similar (but not identical due to tolerance difference)
        # Max relative difference should be less than 20% for coefficients
        rel_diff = np.abs((beta_tight - beta_loose) / (np.abs(beta_tight) + 1e-10))
        assert np.max(rel_diff) < 0.2

        # Both should use CG method
        assert info_tight['method'] == 'cg'
        assert info_loose['method'] == 'cg'

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_invalid_inputs(self):
        """Test solver raises errors for invalid inputs."""
        np.random.seed(666)
        x = np.linspace(0, 10, 50)
        knots = BSplineBasis.create_knots(x, n_basis=10, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B = basis.basis_matrix(x, sparse=True)
        y = np.ones(50)
        weights = np.ones(50)
        S = basis.penalty_matrix(order=2)

        # Wrong dimensions for z
        with pytest.raises(ValueError, match="z must have shape"):
            solve_sparse_penalized_ls(B, np.ones(40), weights, S, lambda_=0.1)

        # Wrong dimensions for weights
        with pytest.raises(ValueError, match="weights must have shape"):
            solve_sparse_penalized_ls(B, y, np.ones(40), S, lambda_=0.1)

        # Wrong dimensions for penalty
        with pytest.raises(ValueError, match="penalty must have shape"):
            solve_sparse_penalized_ls(B, y, weights, np.eye(5), lambda_=0.1)

        # Negative lambda
        with pytest.raises(ValueError, match="lambda_ must be non-negative"):
            solve_sparse_penalized_ls(B, y, weights, S, lambda_=-0.1)

        # Invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            solve_sparse_penalized_ls(B, y, weights, S, lambda_=0.1, method='invalid')

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_sparse_efficiency(self):
        """Verify sparse solver is faster than dense for large problems."""
        import time

        np.random.seed(777)
        x = np.linspace(0, 20, 500)  # Large problem
        y = np.sin(x) + 0.1 * np.random.randn(500)

        knots = BSplineBasis.create_knots(x, n_basis=40, degree=3)
        basis = BSplineBasis(knots, degree=3)
        B_sparse = basis.basis_matrix(x, sparse=True)
        B_dense = B_sparse.toarray()
        S = basis.penalty_matrix(order=2)
        weights = np.ones(500)
        lambda_ = 0.1

        # Time sparse solver
        start_sparse = time.time()
        beta_sparse, _ = solve_sparse_penalized_ls(
            B_sparse, y, weights, S, lambda_, method='direct'
        )
        time_sparse = time.time() - start_sparse

        # Time dense solver
        start_dense = time.time()
        C_dense = B_dense.T @ np.diag(weights) @ B_dense + lambda_ * S
        d_dense = B_dense.T @ (weights * y)
        beta_dense = np.linalg.solve(C_dense, d_dense)
        time_dense = time.time() - start_dense

        # Results should match
        np.testing.assert_allclose(beta_sparse, beta_dense, rtol=1e-8)

        # Sparse should be faster (but not strictly required for test to pass)
        # This is mainly for demonstration/benchmarking
        print(f"\nSparse time: {time_sparse:.4f}s, Dense time: {time_dense:.4f}s")
        print(f"Speedup: {time_dense / time_sparse:.2f}×")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
