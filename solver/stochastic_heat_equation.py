from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from math import ceil, exp, fsum, isfinite, pi, sin, sqrt
from random import Random
from typing import Dict, List, Optional, Sequence

from .heat_equation import compute_l2_error, stability_limit


@dataclass
class StochasticHeatEquationConfig:
    alpha: float
    x_start: float
    x_end: float
    nx: int
    t_end: float
    dt: float
    sigma: float = 0.5
    seed: int = 12345


class BrownianPath:
    """Sample a Brownian path at requested times using conditional bridges."""

    def __init__(self, t_end: float, seed: int):
        if not isfinite(t_end) or t_end < 0:
            raise ValueError("t_end must be finite and nonnegative")
        self.t_end = t_end
        self._rng = Random(seed)
        self._times = [0.0] if t_end == 0 else [0.0, t_end]
        self._values = {0.0: 0.0}
        if t_end > 0:
            self._values[t_end] = self._rng.gauss(0.0, sqrt(t_end))

    def value(self, t: float) -> float:
        if not isfinite(t) or not 0.0 <= t <= self.t_end:
            raise ValueError("Brownian evaluation time must lie in [0, t_end]")
        if t in self._values:
            return self._values[t]
        index = bisect_left(self._times, t)
        left, right = self._times[index - 1], self._times[index]
        weight = (t - left) / (right - left)
        mean = (1.0 - weight) * self._values[left] + weight * self._values[right]
        variance = (t - left) * (right - t) / (right - left)
        self._values[t] = mean + self._rng.gauss(0.0, sqrt(variance))
        self._times.insert(index, t)
        return self._values[t]

    def increments(self, time_grid: Sequence[float]) -> List[float]:
        values = [self.value(t) for t in time_grid]
        return [right - left for left, right in zip(values, values[1:])]


def stochastic_analytical_solution(
    x: float,
    t: float,
    alpha: float,
    sigma: float,
    brownian_value: float,
    x_start: float = 0.0,
    x_end: float = 1.0,
) -> float:
    """Evaluate the explicit Itô solution along a prescribed Brownian path."""
    if x == x_start or x == x_end:
        return 0.0
    length = x_end - x_start
    mode = pi * (x - x_start) / length
    amplitude = exp(
        (-alpha * (pi / length) ** 2 - 0.5 * sigma ** 2) * t
        + sigma * brownian_value
    )
    return amplitude * sin(mode)


def _validate_config(config: StochasticHeatEquationConfig) -> None:
    if not all(isfinite(value) for value in (
        config.alpha,
        config.x_start,
        config.x_end,
        config.t_end,
        config.dt,
        config.sigma,
    )):
        raise ValueError("SPDE parameters must be finite")
    if config.alpha <= 0 or config.dt <= 0:
        raise ValueError("alpha and dt must be positive")
    if config.x_end <= config.x_start or config.t_end < 0:
        raise ValueError("Invalid domain or terminal time")
    if isinstance(config.nx, bool) or not isinstance(config.nx, int) or config.nx < 2:
        raise ValueError("nx must be an integer of at least 2")


def solve_stochastic_heat_equation_1d(
    config: StochasticHeatEquationConfig,
    *,
    brownian_path: Optional[BrownianPath] = None,
    brownian_increments: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Solve one path using centered differences and Euler–Maruyama."""
    _validate_config(config)
    if brownian_path is not None and brownian_increments is not None:
        raise ValueError("Supply either a Brownian path or increments, not both")

    steps = max(1, ceil(config.t_end / config.dt)) if config.t_end else 0
    time_grid = [min(index * config.dt, config.t_end) for index in range(steps + 1)]
    time_grid[-1] = config.t_end
    time_grid = [
        t for index, t in enumerate(time_grid)
        if index == 0 or t > time_grid[index - 1]
    ]

    if brownian_increments is None:
        path = brownian_path or BrownianPath(config.t_end, config.seed)
        if path.t_end != config.t_end:
            raise ValueError("Brownian path terminal time does not match the task")
        increments = path.increments(time_grid)
        brownian_terminal = path.value(config.t_end)
    else:
        increments = list(brownian_increments)
        if len(increments) != len(time_grid) - 1 or not all(isfinite(dw) for dw in increments):
            raise ValueError("Provide one finite Brownian increment per time interval")
        brownian_terminal = fsum(increments)

    dx = (config.x_end - config.x_start) / config.nx
    x_grid = [config.x_start + index * dx for index in range(config.nx + 1)]
    u = [sin(pi * (x - config.x_start) / (config.x_end - config.x_start)) for x in x_grid]
    u[0] = u[-1] = 0.0

    for left, right, dw in zip(time_grid, time_grid[1:], increments):
        ratio = config.alpha * (right - left) / (dx * dx)
        next_u = u[:]
        for index in range(1, config.nx):
            next_u[index] = (
                u[index]
                + ratio * (u[index + 1] - 2.0 * u[index] + u[index - 1])
                + config.sigma * u[index] * dw
            )
        next_u[0] = next_u[-1] = 0.0
        u = next_u

    reference = [stochastic_analytical_solution(
        x,
        config.t_end,
        config.alpha,
        config.sigma,
        brownian_terminal,
        config.x_start,
        config.x_end,
    ) for x in x_grid]
    finite_solution = all(isfinite(value) for value in u)
    max_dt = stability_limit(config.alpha, dx)
    return {
        "config": {**asdict(config), "equation": "stochastic_heat_1d"},
        "grid": {"dx": dx, "x": x_grid, "time_steps": len(increments)},
        "stability": {
            "is_stable": config.dt <= max_dt and finite_solution,
            "max_stable_dt": max_dt,
            "cfl": config.alpha * config.dt / (dx * dx),
            "criterion": "deterministic_diffusion_cfl_and_finite_solution",
        },
        "solution": {"numerical_final": u, "reference_final": reference},
        "errors": {
            "l2": compute_l2_error(u, reference, dx)
            if finite_solution else float("inf")
        },
        "noise": {
            "type": "scalar_temporal_multiplicative",
            "interpretation": "ito",
            "seed": config.seed,
            "time_grid": time_grid,
            "brownian_increments": increments,
            "brownian_terminal": brownian_terminal,
        },
    }
