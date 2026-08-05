"""Representative-day sampling for ``optimize_h2_producer.solve()``'s
``rep_days_per_month=`` mode (and ``plan_h2_capacity.py --rep-days-per-month`` for the
Benders subproblems) -- picks a handful of days per month instead of solving the full
364-day year, each day carrying a WEIGHT so a sum over just the sampled hours can be
scaled up to approximate a real full-year total. See Formulation.md SS2.6.

The model's year is 364 days (52 exact weeks, ``364*24 = 8736`` hours) with no real
calendar attached -- "month" here just means one of 12 equal-ish day-index chunks, not
an actual Jan/Feb/... boundary.
"""
from __future__ import annotations

import numpy as np

TOTAL_DAYS = 364
N_MONTHS = 12


def month_boundaries(total_days: int = TOTAL_DAYS, n_months: int = N_MONTHS) -> list[int]:
    """``n_months + 1`` cut points partitioning ``[0, total_days)`` into ``n_months``
    chunks (each close to, but not necessarily exactly, ``total_days / n_months`` days,
    since 364 isn't evenly divisible by 12) -- e.g. ``[0, 30, 61, 91, ..., 364]``."""
    return [round(i * total_days / n_months) for i in range(n_months + 1)]


def representative_days(n: int, total_days: int = TOTAL_DAYS, n_months: int = N_MONTHS,
                        ) -> tuple[list[int], list[float]]:
    """``n`` evenly-spaced representative day numbers (1-indexed, ``1..total_days``) per
    month-chunk, plus each day's WEIGHT (how many real days it stands in for).

    Weights within a chunk always sum to that chunk's true length, so
    ``sum(weights) == total_days`` always, independent of ``n`` -- this is what lets a
    weighted sum over just the sampled hours approximate a real full-year total
    (``optimize_h2_producer.solve()``'s demand-conservation/RED-III-quota/objective all
    use these weights this way). If ``n`` meets or exceeds a chunk's length, every day
    in that chunk is used at weight 1 (no sampling needed, and that chunk contributes
    its true days exactly rather than an approximation).

    Returns ``(days, weights)``, same length, sorted ascending by day. Length is
    ``n * n_months`` unless clipped by a short chunk (``n >= length``, weight-1 case) or
    by two evenly-spaced picks in a short chunk rounding onto the same day (rare, only
    possible when ``n`` is just under a chunk's length)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bounds = month_boundaries(total_days, n_months)
    days: list[int] = []
    weights: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        length = hi - lo
        if n >= length:
            days.extend(range(lo + 1, hi + 1))
            weights.extend([1.0] * length)
            continue
        picks = np.unique(np.round(np.linspace(lo, hi - 1, n)).astype(int))
        w = length / len(picks)
        days.extend(int(p) + 1 for p in picks)
        weights.extend([w] * len(picks))
    return days, weights
