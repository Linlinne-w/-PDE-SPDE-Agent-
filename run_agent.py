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
from typing import Dict, List, Optional, Sequence, Union

from evaluator import evaluate_result
from solver import (
    BrownianPath,
    HeatEquationConfig,
    StochasticHeatEquationConfig,
    solve_heat_equation_1d,
    solve_stochastic_heat_equation_1d,
)
from tasks import load_task


ROOT = Path(__file__).resolve().parent
SolverConfig = Union[HeatEquationConfig, StochasticHeatEquationConfig]


def _plot_solution(solved: Dict[str, object], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = solved["grid"]["x"]
    u_num = solved["solution"]["numerical_final"]
    u_ref = solved["solution"]["reference_final"]
    error = [abs(a - b) for a, b in zip(u_num, u_ref)]
    cfg = solved["config"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, layout="constrained"
    )
    try:
        reference_label = "Analytical solution (same Brownian path)" if "noise" in solved else "Analytical solution"
        ax1.plot(x, u_ref, "k-", label=reference_label)
        ax1.plot(
            x, u_num, "ro", markersize=4, fillstyle="none",
            label="Numerical solution"
        )
        ax1.set_ylabel("u(x, t)")
        ax1.set_title(
            f"t = {cfg['t_end']:g}, nx = {cfg['nx']}, dt = {cfg['dt']:g}\n"
            f"L2 error = {solved['errors']['l2']:.3e}"
        )
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(x, error, color="tab:blue")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Absolute error")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax2.grid(alpha=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def _solve_config(
    config: SolverConfig,
    *,
    brownian_path: Optional[BrownianPath] = None,
    brownian_increments: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    if isinstance(config, StochasticHeatEquationConfig):
        return solve_stochastic_heat_equation_1d(
            config, brownian_path=brownian_path,
            brownian_increments=brownian_increments,
        )
    return solve_heat_equation_1d(config)


def _measure_solve(
    config: SolverConfig, *, brownian_path: Optional[BrownianPath] = None,
) -> Dict[str, object]:
    tracemalloc.start()
    start = perf_counter()
    solve_out = _solve_config(config, brownian_path=brownian_path)
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


def _check_reproducible(
    config: SolverConfig,
    baseline_error: float,
    *,
    brownian_increments: Optional[Sequence[float]] = None,
) -> bool:
    rerun = _solve_config(config, brownian_increments=brownian_increments)
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


def _write_outputs(
    run_id: str,
    task: Dict[str, object],
    metrics: Dict[str, object],
    attempts: List[Dict[str, object]],
    solved: Optional[Dict[str, object]] = None,
) -> None:
    run_dir = ROOT / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)

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
            writer.writerow({
                "attempt": row["attempt"], "dt": row["dt"], "nx": row["nx"],
                "l2_error": row["l2_error"], "pass": row["pass"],
            })

    if solved is not None:
        with (artifacts_dir / "solution_final.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "numerical", "reference", "absolute_error"])
            for x, numerical, reference in zip(
                solved["grid"]["x"], solved["solution"]["numerical_final"],
                solved["solution"]["reference_final"],
            ):
                writer.writerow([x, numerical, reference, abs(numerical - reference)])
        if "noise" in solved:
            noise = solved["noise"]
            with (artifacts_dir / "brownian_path.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "brownian_value", "increment"])
                writer.writerow([0.0, 0.0, 0.0])
                brownian_value = 0.0
                for t, dw in zip(noise["time_grid"][1:], noise["brownian_increments"]):
                    brownian_value += dw
                    if t == noise["time_grid"][-1]:
                        brownian_value = noise["brownian_terminal"]
                    writer.writerow([t, brownian_value, dw])

    report_lines = [
        f"# Run Report: {run_id}",
        "",
        f"- Equation: {task['equation']}",
        f"- Pass: **{metrics['pass']}**",
        f"- Total Score: **{metrics['score_breakdown']['total_100']} / 100**",
        f"- Failure Types: {', '.join(metrics['failure_types']) if metrics['failure_types'] else 'none'}",
        "",
        "## Next-round Suggestions",
    ]
    if task["equation"] == "stochastic_heat_1d":
        report_lines[2:2] = [
            "- Error: single-path spatial L2 error against the explicit Itô solution.",
            "- Numerical and reference solutions use the same Brownian path; retries are coupled.",
            "- No Monte Carlo strong-convergence order is estimated in this run.",
            "- Stability gate checks diffusion CFL and finite output, not a stochastic mean-square stability theorem.",
        ]
    for suggestion in metrics["next_round_suggestions"]:
        report_lines.append(f"- {suggestion}")
    report_lines.extend(["", "## Attempts", ""])
    for row in attempts:
        report_lines.append(
            f"- Attempt {row['attempt']}: dt={row['dt']}, nx={row['nx']}, l2={row['l2_error']:.6e}, pass={row['pass']}"
        )

    (ROOT / "reports" / f"{run_id}.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _build_output_paths(run_id: str, *, stochastic: bool = False) -> Dict[str, str]:
    run_dir = Path("runs") / run_id
    artifacts_dir = run_dir / "artifacts"
    outputs = {
        "run_dir": run_dir.as_posix(),
        "config": (run_dir / "config.json").as_posix(),
        "metrics": (run_dir / "metrics.json").as_posix(),
        "artifacts_dir": artifacts_dir.as_posix(),
        "report": (Path("reports") / f"{run_id}.md").as_posix(),
        "solution": (artifacts_dir / "solution_final.csv").as_posix(),
    }
    if stochastic:
        outputs["brownian_path"] = (artifacts_dir / "brownian_path.csv").as_posix()
    return outputs


def run(task_path: Path) -> Dict[str, object]:
    task = load_task(task_path)
    random.seed(task["seed"])
    stochastic = task["equation"] == "stochastic_heat_1d"
    brownian_path = BrownianPath(float(task["t_end"]), int(task["seed"])) if stochastic else None

    attempts: List[Dict[str, object]] = []
    evaluation = None

    for attempt in range(1, int(task["max_retries"]) + 2):
        config_fields = dict(
            alpha=float(task["alpha"]),
            x_start=float(task["domain"]["x_start"]),
            x_end=float(task["domain"]["x_end"]),
            nx=int(task["nx"]),
            t_end=float(task["t_end"]),
            dt=float(task["dt"]),
        )
        config = (
            StochasticHeatEquationConfig(
                **config_fields, sigma=float(task["sigma"]), seed=int(task["seed"]),
            ) if stochastic else HeatEquationConfig(**config_fields)
        )

        solved = _measure_solve(config, brownian_path=brownian_path)
        noise = solved.get("noise", {})
        solved["reproducible"] = _check_reproducible(
            config, solved["errors"]["l2"],
            brownian_increments=noise.get("brownian_increments"),
        )
        solved["robustness_pass_rate"] = 1.0 if solved["errors"]["l2"] <= float(task["error_threshold"]) else 0.0
        solved["convergence"] = [] if stochastic else _build_convergence_data(config)
        evaluation = evaluate_result(task, solved)
        if stochastic:
            evaluation["metrics"].update({
                "error_type": "single_path_l2",
                "sigma": config.sigma,
                "seed": config.seed,
                "brownian_terminal": noise["brownian_terminal"],
            })

        attempts.append({
            "attempt": attempt,
            "dt": task["dt"],
            "nx": task["nx"],
            "l2_error": solved["errors"]["l2"],
            "pass": evaluation["pass"],
        })
        if stochastic:
            attempts[-1]["brownian_terminal"] = noise["brownian_terminal"]

        if evaluation["pass"]:
            break
        if attempt <= int(task["max_retries"]):
            _auto_correct_params(task, evaluation["failure_types"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    _write_outputs(run_id, task, evaluation, attempts, solved)

    warning = None
    try:
        _plot_solution(solved, ROOT / "runs" / run_id / "artifacts" / "solution_comparison.png")
    except Exception as exc:  # pragma: no cover - best-effort artifact generation
        warning = f"failed_to_plot_solution: {type(exc).__name__}: {exc}"

    outcome = {
        "run_id": run_id,
        "evaluation": evaluation,
        "attempts": attempts,
        "outputs": _build_output_paths(run_id, stochastic=stochastic),
    }
    if warning:
        outcome["warnings"] = [warning]
    return outcome


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
