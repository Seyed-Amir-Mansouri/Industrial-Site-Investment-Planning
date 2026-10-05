"""Bridge between the Django UI and the optimizer: builds plan_capacity.py commands and reads results back."""
from __future__ import annotations

import csv
import re
import sys
from functools import lru_cache
from pathlib import Path

from django.conf import settings

PROJECT_ROOT = Path(settings.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "webui"


@lru_cache(maxsize=1)
def planner_module():
    import plan_capacity

    return plan_capacity


@lru_cache(maxsize=1)
def capex_assumptions_defaults():
    import g_investor_planning as hp

    cfg = hp.CapexAssumptions()
    return {
        "assets": list(hp.ASSETS),
        "catalog": {a: [c._asdict() for c in cfg.catalog[a]] for a in hp.ASSETS},
        "discount_rate": cfg.discount_rate,
        "budget": cfg.default_budget_eur,
        "lifetime_years": dict(cfg.lifetime_years),
    }


@lru_cache(maxsize=1)
def eligible_countries() -> list[str]:
    return planner_module().eligible_countries()


@lru_cache(maxsize=1)
def scenario_probabilities() -> dict[str, float]:
    return dict(planner_module().SCENARIO_PROBS)


def build_command(params: dict, output_prefix: Path) -> list[str]:
    cmd = [sys.executable, "plan_capacity.py"]
    if params["all_countries"]:
        cmd.append("--all")
    else:
        cmd += ["--countries", ",".join(params["countries"])]
    cmd += ["--budget", f"{params['budget']:.0f}",
            "--max-units-per-candidate", str(params["max_units_per_candidate"]),
            "--discount-rate", f"{params['discount_rate_pct'] / 100:.6f}",
            "--rep-days-per-month", str(params["rep_days_per_month"]),
            "--gap-tol", str(params["gap_tol"]),
            "--max-iters", str(params["max_iters"]),
            "--master-time-limit", str(params["master_time_limit"]),
            "--workers", str(params["workers"]),
            "--cvar-alpha", str(params["cvar_alpha"]) if params["risk_measure"] == "cvar" else "off",
            "--output", str(output_prefix)]
    if params.get("lifetime_years"):
        cmd += ["--lifetime-years", str(params["lifetime_years"])]
    if params.get("disabled_assets"):
        cmd += ["--disabled-assets", ",".join(params["disabled_assets"])]
    if params.get("scenarios") and set(params["scenarios"]) != set(scenario_probabilities()):
        cmd += ["--scenarios", ",".join(params["scenarios"])]
    return cmd


def run_dir(run_pk: int) -> Path:
    return RUNS_DIR / f"run_{run_pk}"


def output_prefix_for(run_pk: int) -> Path:
    return run_dir(run_pk) / "plan"


_DONE_RE = re.compile(r"Done in ([\d.]+)s")
_CAPEX_RE = re.compile(r"Total raw CAPEX: ([\d,]+) EUR")
_OBJ_RE = re.compile(r"Best objective.*?: (-?[\d,]+)\s*$", re.MULTILINE)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def parse_summary(output_prefix: Path, log: str) -> dict:
    capacities = _read_csv(Path(f"{output_prefix}_capacities.csv"))
    units = _read_csv(Path(f"{output_prefix}_units.csv"))
    convergence = _read_csv(Path(f"{output_prefix}_convergence.csv"))

    assets = list(capex_assumptions_defaults()["assets"])
    totals = {a: 0.0 for a in assets}
    for row in capacities:
        for a in assets:
            totals[a] += float(row[a])

    done = _DONE_RE.search(log)
    capex = _CAPEX_RE.search(log)
    objective = _OBJ_RE.search(log)
    return {
        "elapsed_seconds": float(done.group(1)) if done else None,
        "iterations": len(convergence),
        "converged": bool(re.search(r"converged \(gap", log)),
        "objective_eur": _num(objective.group(1)) if objective else None,
        "raw_capex_eur": _num(capex.group(1)) if capex else None,
        "capacities": capacities,
        "units": units,
        "convergence": [{k: _maybe_float(v) for k, v in row.items()} for row in convergence],
        "totals_mw": totals,
    }


def _maybe_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
