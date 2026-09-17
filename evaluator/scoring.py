from __future__ import annotations

from math import log
from typing import Dict, List, Optional


def estimate_convergence_order(convergence_data: List[Dict[str, float]]) -> Optional[float]:
    if len(convergence_data) < 2:
        return None
    h1 = convergence_data[-2]["dx"]
    h2 = convergence_data[-1]["dx"]
    e1 = convergence_data[-2]["l2_error"]
    e2 = convergence_data[-1]["l2_error"]
    if h1 <= 0 or h2 <= 0 or e1 <= 0 or e2 <= 0:
        return None
    return log(e1 / e2) / log(h1 / h2)


def _efficiency_score(runtime_s: float, memory_mb: float, budget: Dict[str, float]) -> float:
    time_budget = max(budget["max_runtime_seconds"], 1e-12)
    mem_budget = max(budget["max_memory_mb"], 1e-12)

    time_ratio = min(runtime_s / time_budget, 1.0)
    mem_ratio = min(memory_mb / mem_budget, 1.0)

    time_sub = max(0.0, (1.0 - time_ratio) * 10.0)
    mem_sub = max(0.0, (1.0 - mem_ratio) * 10.0)
    return time_sub + mem_sub


def evaluate_result(task: Dict[str, object], result: Dict[str, object]) -> Dict[str, object]:
    l2_error = float(result["errors"]["l2"])
    error_threshold = float(task["error_threshold"])
    runtime_s = float(result["resources"]["runtime_seconds"])
    memory_mb = float(result["resources"]["memory_mb"])
    budget = task["budget"]
    stable = bool(result["stability"]["is_stable"])
    reproducible = bool(result.get("reproducible", True))
    robustness_pass_rate = float(result.get("robustness_pass_rate", 1.0))

    failure_types: List[str] = []
    if not stable:
        failure_types.append("divergence_or_instability")
    if l2_error > error_threshold:
        failure_types.append("accuracy_not_met")
    if runtime_s > float(budget["max_runtime_seconds"]):
        failure_types.append("timeout")
    if memory_mb > float(budget["max_memory_mb"]):
        failure_types.append("memory_exceeded")

    hard_pass = len(failure_types) == 0

    correctness = 50.0 if stable and l2_error <= error_threshold else max(0.0, 50.0 * (1.0 - l2_error / max(error_threshold, 1e-12)))
    efficiency = _efficiency_score(runtime_s, memory_mb, budget)
    robustness = max(0.0, min(20.0, 20.0 * robustness_pass_rate))
    reproducibility = 10.0 if reproducible else 0.0

    total_score = correctness + efficiency + robustness + reproducibility

    suggestions = []
    if "accuracy_not_met" in failure_types:
        suggestions.append("Reduce dt and refine nx to improve accuracy.")
    if "divergence_or_instability" in failure_types:
        suggestions.append("Use smaller dt or a more stable scheme.")
    if "timeout" in failure_types:
        suggestions.append("Coarsen nx or increase dt within stability range.")
    if "memory_exceeded" in failure_types:
        suggestions.append("Reduce nx or store fewer intermediate arrays.")
    if not suggestions:
        suggestions.append("Current setup passes hard gates; optimize efficiency if needed.")

    convergence_order = estimate_convergence_order(result.get("convergence", []))

    return {
        "pass": hard_pass,
        "hard_gates": {
            "error_within_threshold": l2_error <= error_threshold,
            "within_runtime_budget": runtime_s <= float(budget["max_runtime_seconds"]),
            "within_memory_budget": memory_mb <= float(budget["max_memory_mb"]),
            "stability_ok": stable,
        },
        "score_breakdown": {
            "correctness_50": round(correctness, 4),
            "efficiency_20": round(efficiency, 4),
            "robustness_20": round(robustness, 4),
            "reproducibility_10": round(reproducibility, 4),
            "total_100": round(total_score, 4),
        },
        "metrics": {
            "l2_error": l2_error,
            "error_threshold": error_threshold,
            "runtime_seconds": runtime_s,
            "memory_mb": memory_mb,
            "estimated_convergence_order": convergence_order,
        },
        "failure_types": failure_types,
        "next_round_suggestions": suggestions,
    }
