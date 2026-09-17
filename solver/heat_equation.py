from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sin, sqrt
from typing import Dict, List


@dataclass
class HeatEquationConfig:
    alpha: float
    x_start: float
    x_end: float
    nx: int
    t_end: float
    dt: float


def stability_limit(alpha: float, dx: float) -> float:
    return (dx * dx) / (2.0 * alpha)


def analytical_solution(x: float, t: float, alpha: float, x_start: float, x_end: float) -> float:
    length = x_end - x_start
    mode = pi * (x - x_start) / length
    return exp(-alpha * (pi / length) ** 2 * t) * sin(mode)


def compute_l2_error(u_num: List[float], u_ref: List[float], dx: float) -> float:
    squared = [(a - b) ** 2 for a, b in zip(u_num, u_ref)]
    return sqrt(dx * sum(squared))


def solve_heat_equation_1d(config: HeatEquationConfig) -> Dict[str, object]:
    dx = (config.x_end - config.x_start) / config.nx
    x_grid = [config.x_start + i * dx for i in range(config.nx + 1)]
    max_dt = stability_limit(config.alpha, dx)
    stable = config.dt <= max_dt

    u = [analytical_solution(x, 0.0, config.alpha, config.x_start, config.x_end) for x in x_grid]
    u[0] = 0.0
    u[-1] = 0.0

    t = 0.0
    steps = 0
    r = config.alpha * config.dt / (dx * dx)

    while t < config.t_end:
        local_dt = min(config.dt, config.t_end - t)
        if local_dt != config.dt:
            r = config.alpha * local_dt / (dx * dx)

        next_u = u[:]
        for i in range(1, config.nx):
            next_u[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1])
        next_u[0] = 0.0
        next_u[-1] = 0.0

        u = next_u
        t += local_dt
        steps += 1

    u_ref = [analytical_solution(x, config.t_end, config.alpha, config.x_start, config.x_end) for x in x_grid]
    l2_error = compute_l2_error(u, u_ref, dx)

    return {
        "config": {
            "alpha": config.alpha,
            "x_start": config.x_start,
            "x_end": config.x_end,
            "nx": config.nx,
            "t_end": config.t_end,
            "dt": config.dt,
        },
        "grid": {"dx": dx, "x": x_grid, "time_steps": steps},
        "stability": {
            "is_stable": stable,
            "max_stable_dt": max_dt,
            "cfl": config.alpha * config.dt / (dx * dx),
        },
        "solution": {"numerical_final": u, "reference_final": u_ref},
        "errors": {"l2": l2_error},
    }
