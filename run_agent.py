from __future__ import annotations

import argparse
import csv
import json
import random
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from evaluator import evaluate_result
from solver import HeatEquationConfig, solve_heat_equation_1d
from tasks import load_task


ROOT = Path(__file__).resolve().parent


def _measure_solve(config: HeatEquationConfig) -> Dict[str, object]:
    tracemalloc.start()
    start = perf_counter()
    solve_out = solve_heat_equation_1d(config)
    runtime = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    solve_out["resources"] = {
        "runtime_seconds": runtime,
        "memory_mb": peak / (1024 * 1024),
    }
    return solve_out


def _build_convergence_data(config: HeatEquationConfig) -> List[Dict[str, float]]:
    coarse = solve_heat_equation_1d(config)
    fine_cfg = HeatEquationConfig(
        alpha=config.alpha,
        x_start=config.x_start,
        x_end=config.x_end,
        nx=config.nx * 2,
        t_end=config.t_end,
        dt=config.dt / 4.0,
    )
    fine = solve_heat_equation_1d(fine_cfg)
    return [
        {"dx": coarse["grid"]["dx"], "l2_error": coarse["errors"]["l2"]},
        {"dx": fine["grid"]["dx"], "l2_error": fine["errors"]["l2"]},
    ]


def _check_reproducible(config: HeatEquationConfig, baseline_error: float) -> bool:
    rerun = solve_heat_equation_1d(config)
    return abs(rerun["errors"]["l2"] - baseline_error) < 1e-14


def _auto_correct_params(task: Dict[str, object], failure_types: List[str]) -> None:
    if "divergence_or_instability" in failure_types:
        task["dt"] = task["dt"] * 0.5
    if "accuracy_not_met" in failure_types:
        task["dt"] = task["dt"] * 0.5
        task["nx"] = int(task["nx"] * 2)
    if "timeout" in failure_types or "memory_exceeded" in failure_types:
        task["nx"] = max(10, int(task["nx"] * 0.75))
        task["dt"] = task["dt"] * 1.2

    dx = (task["domain"]["x_end"] - task["domain"]["x_start"]) / task["nx"]
    max_dt = (dx * dx) / (2.0 * task["alpha"])
    task["dt"] = min(task["dt"], max_dt)


def _write_outputs(run_id: str, task: Dict[str, object], metrics: Dict[str, object], attempts: List[Dict[str, object]]) -> None:
    run_dir = ROOT / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_metrics": metrics,
        "attempts": attempts,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = artifacts_dir / "error_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attempt", "dt", "nx", "l2_error", "pass"])
        writer.writeheader()
        for row in attempts:
            writer.writerow(
                {
                    "attempt": row["attempt"],
                    "dt": row["dt"],
                    "nx": row["nx"],
                    "l2_error": row["l2_error"],
                    "pass": row["pass"],
                }
            )

    report_lines = [
        f"# Run Report: {run_id}",
        "",
        f"- Pass: **{metrics['pass']}**",
        f"- Total Score: **{metrics['score_breakdown']['total_100']} / 100**",
        f"- Failure Types: {', '.join(metrics['failure_types']) if metrics['failure_types'] else 'none'}",
        "",
        "## Next-round Suggestions",
    ]
    for suggestion in metrics["next_round_suggestions"]:
        report_lines.append(f"- {suggestion}")
    report_lines.extend(["", "## Attempts", ""])
    for row in attempts:
        report_lines.append(
            f"- Attempt {row['attempt']}: dt={row['dt']}, nx={row['nx']}, l2={row['l2_error']:.6e}, pass={row['pass']}"
        )

    (ROOT / "reports" / f"{run_id}.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run(task_path: Path) -> Dict[str, object]:
    task = load_task(task_path)
    random.seed(task["seed"])

    attempts: List[Dict[str, object]] = []
    evaluation = None

    for attempt in range(1, int(task["max_retries"]) + 2):
        config = HeatEquationConfig(
            alpha=float(task["alpha"]),
            x_start=float(task["domain"]["x_start"]),
            x_end=float(task["domain"]["x_end"]),
            nx=int(task["nx"]),
            t_end=float(task["t_end"]),
            dt=float(task["dt"]),
        )

        solved = _measure_solve(config)
        solved["reproducible"] = _check_reproducible(config, solved["errors"]["l2"])
        solved["robustness_pass_rate"] = 1.0 if solved["errors"]["l2"] <= float(task["error_threshold"]) else 0.0
        solved["convergence"] = _build_convergence_data(config)
        evaluation = evaluate_result(task, solved)

        attempts.append(
            {
                "attempt": attempt,
                "dt": task["dt"],
                "nx": task["nx"],
                "l2_error": solved["errors"]["l2"],
                "pass": evaluation["pass"],
            }
        )

        if evaluation["pass"]:
            break

        if attempt <= int(task["max_retries"]):
            _auto_correct_params(task, evaluation["failure_types"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    _write_outputs(run_id, task, evaluation, attempts)
    return {"run_id": run_id, "evaluation": evaluation, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal PDE Research Agent workflow")
    parser.add_argument(
        "--task",
        default=str(ROOT / "tasks" / "heat_equation_1d.json"),
        help="Path to task JSON file",
    )
    args = parser.parse_args()

    outcome = run(Path(args.task))
    print(json.dumps(outcome, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
