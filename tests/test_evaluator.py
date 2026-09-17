import unittest

from evaluator import evaluate_result


class EvaluatorTests(unittest.TestCase):
    def test_evaluator_pass_case(self):
        task = {
            "error_threshold": 0.01,
            "budget": {"max_runtime_seconds": 1.0, "max_memory_mb": 64.0},
        }
        result = {
            "errors": {"l2": 0.005},
            "resources": {"runtime_seconds": 0.01, "memory_mb": 1.0},
            "stability": {"is_stable": True},
            "reproducible": True,
            "robustness_pass_rate": 1.0,
        }
        out = evaluate_result(task, result)
        self.assertTrue(out["pass"])
        self.assertEqual(out["score_breakdown"]["correctness_50"], 50.0)

    def test_evaluator_failure_types(self):
        task = {
            "error_threshold": 0.01,
            "budget": {"max_runtime_seconds": 0.001, "max_memory_mb": 0.001},
        }
        result = {
            "errors": {"l2": 0.1},
            "resources": {"runtime_seconds": 1.0, "memory_mb": 2.0},
            "stability": {"is_stable": False},
            "reproducible": False,
            "robustness_pass_rate": 0.0,
        }
        out = evaluate_result(task, result)
        self.assertFalse(out["pass"])
        self.assertIn("accuracy_not_met", out["failure_types"])
        self.assertIn("timeout", out["failure_types"])
        self.assertIn("memory_exceeded", out["failure_types"])
        self.assertIn("divergence_or_instability", out["failure_types"])


if __name__ == "__main__":
    unittest.main()
