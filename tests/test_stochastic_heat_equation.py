import math
import random
import unittest
from dataclasses import replace

from solver import (
    BrownianPath,
    HeatEquationConfig,
    StochasticHeatEquationConfig,
    solve_heat_equation_1d,
    solve_stochastic_heat_equation_1d,
    stochastic_analytical_solution,
)


class StochasticHeatEquationTests(unittest.TestCase):
    def setUp(self):
        self.config = StochasticHeatEquationConfig(
            alpha=1.0, x_start=0.0, x_end=1.0, nx=40,
            t_end=0.05, dt=0.0002, sigma=0.5, seed=12345,
        )

    def test_explicit_solution_contains_ito_correction(self):
        value = stochastic_analytical_solution(0.5, 0.7, 0.3, 0.8, 0.4)
        self.assertAlmostEqual(
            math.log(value), (-0.3 * math.pi ** 2 - 0.8 ** 2 / 2) * 0.7 + 0.8 * 0.4,
        )

    def test_one_step_euler_maruyama_with_prescribed_increment(self):
        config = replace(self.config, nx=2, alpha=0.7, sigma=0.3, dt=0.01, t_end=0.01)
        out = solve_stochastic_heat_equation_1d(config, brownian_increments=[0.12])
        numerical = 1.0 - 8.0 * 0.7 * 0.01 + 0.3 * 0.12
        reference = math.exp((-0.7 * math.pi ** 2 - 0.3 ** 2 / 2) * 0.01 + 0.3 * 0.12)
        self.assertAlmostEqual(out["solution"]["numerical_final"][1], numerical)
        self.assertAlmostEqual(out["solution"]["reference_final"][1], reference)
        self.assertAlmostEqual(out["errors"]["l2"], math.sqrt(0.5) * abs(numerical - reference))

    def test_default_solution_is_finite_and_has_zero_boundaries(self):
        out = solve_stochastic_heat_equation_1d(self.config)
        self.assertTrue(out["stability"]["is_stable"])
        self.assertLess(out["errors"]["l2"], 0.01)
        for key in ["numerical_final", "reference_final"]:
            self.assertEqual(out["solution"][key][0], 0.0)
            self.assertEqual(out["solution"][key][-1], 0.0)
            self.assertTrue(all(math.isfinite(value) for value in out["solution"][key]))

    def test_zero_noise_reduces_to_deterministic_heat_solver(self):
        out = solve_stochastic_heat_equation_1d(replace(self.config, sigma=0.0))
        deterministic = solve_heat_equation_1d(HeatEquationConfig(
            alpha=self.config.alpha, x_start=0.0, x_end=1.0, nx=40,
            t_end=self.config.t_end, dt=self.config.dt,
        ))
        for numerical, expected in zip(
            out["solution"]["numerical_final"], deterministic["solution"]["numerical_final"],
        ):
            self.assertAlmostEqual(numerical, expected, places=12)

    def test_seed_reproducibility_and_recorded_increment_replay(self):
        out = solve_stochastic_heat_equation_1d(self.config)
        self.assertEqual(out, solve_stochastic_heat_equation_1d(self.config))
        replay = solve_stochastic_heat_equation_1d(
            self.config, brownian_increments=out["noise"]["brownian_increments"],
        )
        self.assertEqual(out["solution"]["numerical_final"], replay["solution"]["numerical_final"])
        self.assertAlmostEqual(out["errors"]["l2"], replay["errors"]["l2"], places=14)
        changed = solve_stochastic_heat_equation_1d(replace(self.config, seed=54321))
        self.assertNotEqual(out["noise"]["brownian_terminal"], changed["noise"]["brownian_terminal"])
        self.assertNotEqual(out["solution"]["numerical_final"], changed["solution"]["numerical_final"])

    def test_partial_final_time_step_and_shifted_domain(self):
        config = replace(self.config, x_start=2.0, x_end=4.0, nx=4, dt=0.002, t_end=0.005)
        out = solve_stochastic_heat_equation_1d(config, brownian_increments=[0.01, -0.02, 0.03])
        self.assertEqual(out["noise"]["time_grid"], [0.0, 0.002, 0.004, 0.005])
        self.assertAlmostEqual(out["noise"]["brownian_terminal"], 0.02)
        self.assertAlmostEqual(out["solution"]["reference_final"][2], math.exp(
            (-math.pi ** 2 / 4 - 0.5 ** 2 / 2) * 0.005 + 0.5 * 0.02,
        ))

    def test_shared_path_couples_coarse_and_fine_increments(self):
        path = BrownianPath(1.0, 91)
        coarse = path.increments([0.0, 0.5, 1.0])
        midpoint = path.value(0.5)
        fine = path.increments([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(path.value(0.5), midpoint)
        self.assertAlmostEqual(sum(fine[:2]), coarse[0])
        self.assertAlmostEqual(sum(fine[2:]), coarse[1])
        self.assertAlmostEqual(sum(fine), path.value(1.0))

    def test_brownian_bridge_variance_and_covariance(self):
        seeds = random.Random(71)
        pairs = []
        for _ in range(4096):
            path = BrownianPath(1.0, seeds.getrandbits(64))
            pairs.append((path.value(0.5), path.value(1.0)))
        mean_half = sum(a for a, _ in pairs) / len(pairs)
        mean_full = sum(b for _, b in pairs) / len(pairs)
        variance_half = sum((a - mean_half) ** 2 for a, _ in pairs) / len(pairs)
        variance_full = sum((b - mean_full) ** 2 for _, b in pairs) / len(pairs)
        covariance = sum((a - mean_half) * (b - mean_full) for a, b in pairs) / len(pairs)
        self.assertLess(abs(mean_half), 0.05)
        self.assertLess(abs(mean_full), 0.05)
        self.assertAlmostEqual(variance_half, 0.5, delta=0.05)
        self.assertAlmostEqual(variance_full, 1.0, delta=0.08)
        self.assertAlmostEqual(covariance, 0.5, delta=0.06)

    def test_invalid_config_and_increment_length_are_rejected(self):
        for config in [replace(self.config, dt=0.0), replace(self.config, nx=1), replace(self.config, sigma=float("nan"))]:
            with self.assertRaises(ValueError):
                solve_stochastic_heat_equation_1d(config)
        with self.assertRaises(ValueError):
            solve_stochastic_heat_equation_1d(self.config, brownian_increments=[0.1])

    def test_zero_terminal_time(self):
        out = solve_stochastic_heat_equation_1d(replace(self.config, t_end=0.0))
        self.assertEqual(out["grid"]["time_steps"], 0)
        self.assertEqual(out["errors"]["l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
