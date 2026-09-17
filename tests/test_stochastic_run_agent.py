import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_agent
from solver import StochasticHeatEquationConfig, solve_stochastic_heat_equation_1d
from tasks import load_task


class StochasticRunAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.task = json.loads((run_agent.ROOT / "tasks" / "stochastic_heat_equation_1d.json").read_text())

    def _run(self):
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps(self.task), encoding="utf-8")
        with patch("run_agent.ROOT", self.root), patch("run_agent._plot_solution"):
            return run_agent.run(task_path)

    def test_spde_dispatch_raw_solution_and_noise_replay(self):
        outcome = self._run()
        self.assertTrue(outcome["evaluation"]["pass"])
        self.assertEqual(outcome["evaluation"]["metrics"]["error_type"], "single_path_l2")
        self.assertIsNone(outcome["evaluation"]["metrics"]["estimated_convergence_order"])
        outputs = outcome["outputs"]
        with (self.root / outputs["brownian_path"]).open(newline="") as file:
            path_rows = list(csv.DictReader(file))
        with (self.root / outputs["solution"]).open(newline="") as file:
            solution_rows = list(csv.DictReader(file))
        increments = [float(row["increment"]) for row in path_rows[1:]]
        self.assertEqual(float(path_rows[-1]["t"]), self.task["t_end"])
        config = StochasticHeatEquationConfig(
            alpha=self.task["alpha"], x_start=0.0, x_end=1.0, nx=self.task["nx"],
            t_end=self.task["t_end"], dt=self.task["dt"], sigma=self.task["sigma"], seed=self.task["seed"],
        )
        replay = solve_stochastic_heat_equation_1d(config, brownian_increments=increments)
        self.assertEqual([float(row["numerical"]) for row in solution_rows], replay["solution"]["numerical_final"])
        self.assertAlmostEqual(outcome["evaluation"]["metrics"]["l2_error"], replay["errors"]["l2"], places=14)
        self.assertTrue((self.root / outputs["report"]).is_file())

    def test_parameter_retries_keep_the_same_brownian_path(self):
        self.task.update(error_threshold=0.0, max_retries=1)
        outcome = self._run()
        self.assertFalse(outcome["evaluation"]["pass"])
        self.assertEqual(len(outcome["attempts"]), 2)
        first, second = outcome["attempts"]
        self.assertNotEqual(first["dt"], second["dt"])
        self.assertEqual(first["brownian_terminal"], second["brownian_terminal"])
        self.assertTrue((self.root / outcome["outputs"]["brownian_path"]).is_file())

    def test_spde_sigma_is_required(self):
        del self.task["sigma"]
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps(self.task))
        with self.assertRaisesRegex(ValueError, "sigma"):
            load_task(task_path)

    def test_unknown_equation_is_rejected(self):
        self.task["equation"] = "unknown_spde"
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps(self.task))
        with self.assertRaisesRegex(ValueError, "Unsupported equation"):
            load_task(task_path)


if __name__ == "__main__":
    unittest.main()
