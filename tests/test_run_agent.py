import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_agent


class RunAgentOutputContractTests(unittest.TestCase):
    def setUp(self):
        self._paths_to_cleanup = []

    def tearDown(self):
        for path in self._paths_to_cleanup:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    def _write_task(self, error_threshold: float) -> Path:
        base_task = json.loads(
            (run_agent.ROOT / "tasks" / "heat_equation_1d.json").read_text(encoding="utf-8")
        )
        base_task["error_threshold"] = error_threshold
        temp_dir = Path(tempfile.mkdtemp())
        task_path = temp_dir / "task.json"
        task_path.write_text(json.dumps(base_task), encoding="utf-8")
        self._paths_to_cleanup.append(temp_dir)
        return task_path

    def _assert_outputs_under_runs(self, outcome):
        outputs = outcome["outputs"]
        run_dir_rel = outputs["run_dir"]
        self.assertTrue(run_dir_rel.startswith("runs/"))

        run_dir_abs = run_agent.ROOT / run_dir_rel
        self._paths_to_cleanup.append(run_dir_abs)
        self._paths_to_cleanup.append(run_agent.ROOT / outputs["report"])

        self.assertTrue(run_dir_abs.exists())
        self.assertTrue((run_agent.ROOT / outputs["config"]).exists())
        self.assertTrue((run_agent.ROOT / outputs["metrics"]).exists())
        self.assertTrue((run_agent.ROOT / outputs["artifacts_dir"]).exists())

    def test_pass_run_outputs_under_runs(self):
        task_path = self._write_task(error_threshold=0.01)
        with patch("run_agent._plot_solution"):
            outcome = run_agent.run(task_path)
        self.assertTrue(outcome["evaluation"]["pass"])
        self._assert_outputs_under_runs(outcome)

    def test_fail_run_outputs_under_runs(self):
        task_path = self._write_task(error_threshold=1e-12)
        with patch("run_agent._plot_solution"):
            outcome = run_agent.run(task_path)
        self.assertFalse(outcome["evaluation"]["pass"])
        self._assert_outputs_under_runs(outcome)

    def test_plot_failure_does_not_break_run_outputs(self):
        task_path = self._write_task(error_threshold=0.01)
        with patch("run_agent._plot_solution", side_effect=ModuleNotFoundError("matplotlib")):
            outcome = run_agent.run(task_path)
        self.assertIn("warnings", outcome)
        self._assert_outputs_under_runs(outcome)


if __name__ == "__main__":
    unittest.main()
