import unittest

import numpy as np

from deeponet_darcy.darcy import nonlinear_residual, solve_nonlinear_darcy


class DarcySolverTests(unittest.TestCase):
    def test_zero_source_returns_zero_solution(self) -> None:
        x = np.linspace(0.0, 1.0, 21)
        u, info = solve_nonlinear_darcy(np.zeros_like(x), x)
        self.assertTrue(info.converged)
        np.testing.assert_allclose(u, 0.0, atol=0.0)

    def test_discrete_manufactured_solution(self) -> None:
        x = np.linspace(0.0, 1.0, 41)
        expected = 0.1 * np.sin(np.pi * x)
        source = np.zeros_like(x)
        # Construct the source with exactly the same nonlinear discrete flux.
        source[1:-1] = nonlinear_residual(expected, np.zeros_like(x), x)
        actual, info = solve_nonlinear_darcy(
            source,
            x,
            max_iter=200,
            update_tol=1e-12,
            residual_tol=1e-11,
            face_scheme="cornell_left",
        )
        self.assertTrue(info.converged, info)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
        self.assertEqual(actual[0], 0.0)
        self.assertEqual(actual[-1], 0.0)

    def test_invalid_relaxation_is_rejected(self) -> None:
        x = np.linspace(0.0, 1.0, 5)
        with self.assertRaises(ValueError):
            solve_nonlinear_darcy(np.ones_like(x), x, relaxation=0.0)


if __name__ == "__main__":
    unittest.main()
