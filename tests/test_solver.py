import unittest

from solver import HeatEquationConfig, solve_heat_equation_1d, stability_limit


class SolverTests(unittest.TestCase):
    def test_stability_limit_positive(self):
        self.assertGreater(stability_limit(1.0, 0.1), 0)

    def test_heat_solver_error_reasonable(self):
        cfg = HeatEquationConfig(alpha=1.0, x_start=0.0, x_end=1.0, nx=40, t_end=0.05, dt=0.0002)
        out = solve_heat_equation_1d(cfg)
        self.assertTrue(out["stability"]["is_stable"])
        self.assertLess(out["errors"]["l2"], 0.02)


if __name__ == "__main__":
    unittest.main()
