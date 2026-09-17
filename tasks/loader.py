from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


REQUIRED_FIELDS = {
    "equation", "domain", "initial_condition", "boundary_condition", "t_end",
    "dt", "nx", "error_threshold", "budget", "seed", "max_retries", "alpha",
}


def load_task(path: str | Path) -> Dict[str, Any]:
    task_path = Path(path)
    data = json.loads(task_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_FIELDS.difference(data.keys()))
    if missing:
        raise ValueError(f"Task missing required fields: {', '.join(missing)}")
    if data["equation"] not in {"heat_1d", "stochastic_heat_1d"}:
        raise ValueError(f"Unsupported equation: {data['equation']}")
    if data["equation"] == "stochastic_heat_1d" and "sigma" not in data:
        raise ValueError("Stochastic heat task missing required field: sigma")
    return data
